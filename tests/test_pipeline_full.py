"""Integration test: full pipeline DFINE -> MobileSAM -> postprocess via run_inference().

This test follows the user's requested steps:
- initialize models (expects GPU available)
- load 00157.jpg (512x512)
- build a GpuFrame and call run_inference()
- assert visible state, mask and bbox present, latency > 0

Note: this test requires a CUDA-capable environment and the model files
located in `src/core/model/` as used elsewhere in the repo.
"""

import os
import time
import numpy as np
from PIL import Image
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import torch
except ImportError:
    torch = None

from core.inference.engine.model_loader import initialize_models
from core.inference.engine.orchestrator import run_inference
from core.types import GpuFrame, FrameMeta

LOG = logging.getLogger("test.pipeline_full")

ROOT = os.path.dirname(os.path.dirname(__file__))
DFINE_PATH = os.path.join(ROOT, "src", "core", "model", "Dfine_last.pt")
MOBILESAM_PATH = os.path.join(ROOT, "src", "core", "model", "mobile_sam.pt")
IMAGE_PATH = os.path.join(ROOT, "00157.jpg")


def test_pipeline_end_to_end():
    if torch is None:
        raise ImportError("torch is required for this test")

    # Preconditions
    assert os.path.exists(DFINE_PATH), f"DFINE model not found: {DFINE_PATH}"
    assert os.path.exists(MOBILESAM_PATH), f"MobileSAM model not found: {MOBILESAM_PATH}"
    assert os.path.exists(IMAGE_PATH), f"Test image not found: {IMAGE_PATH}"

    # 1) Initialize models - prefer CUDA device
    model_paths = {
        "dfine": DFINE_PATH,
        "mobilesam": MOBILESAM_PATH,
        "precision": "auto",
    }

    models = initialize_models(model_paths, device="cuda")
    device = models.get("device")
    LOG.info("Models initialized on device: %s", device)

    # Ensure device is CUDA as requested by the plan
    assert getattr(device, "type", str(device)) == "cuda", "Expected models on CUDA device"

    # 2) Load and prepare image 640x640 RGB uint8 (model was trained on 640x640)
    with Image.open(IMAGE_PATH) as im:
        im = im.convert("RGB")
        im = im.resize((640, 640), Image.BILINEAR)
        image = np.array(im, dtype=np.uint8)

    # 3) Convert to torch.Tensor with shape [1, 3, H, W] (batch, RGB channels, height, width)
    # Note: DFINE model expects RGB input (3 channels), not grayscale
    # Transpose from (H, W, C) to (C, H, W) and add batch dimension
    # Normalize pixels to [0, 1] range
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)  # [1, 3, 640, 640]
    image_tensor = image_tensor.to("cuda")  # Transfer to CUDA

    # 4) Build a GpuFrame-like object using core.types.GpuFrame with proper tensor
    meta = FrameMeta(frame_id=1, ts=float(time.time()))
    gf = GpuFrame(tensor=image_tensor, meta=meta, stream=None)

    # 5) Run inference
    result, latency_ms = run_inference(gf)

    LOG.info("Result state=%s latency=%.3f ms", result.get("state"), latency_ms)

    # 6) Assertions as requested
    assert latency_ms is not None and latency_ms > 0.0
    assert result.get("state") == "VISIBLE", f"Expected VISIBLE, got {result.get('state')}"
    assert result.get("mask") is not None, "Mask is None"
    assert result.get("bbox") is not None, "BBox is None"

    # Optional: basic sanity on mask shape
    mask = result.get("mask")
    assert hasattr(mask, "shape"), "Mask has no shape"
    assert mask.shape[0] > 0 and mask.shape[1] > 0, "Mask appears empty"

    LOG.info("Pipeline full test passed: mask shape=%s score=%.3f", mask.shape, float(result.get("score", 0.0)))

    # 7) Visualisation: affichage de 3 images côte à côte
    # Gauche: image originale, Milieu: image avec bbox, Droite: image avec mask
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Image de gauche: originale
    axes[0].imshow(image)
    axes[0].set_title("Image originale\n640x640 RGB", fontsize=12)
    axes[0].axis('off')
    
    # Image du milieu: avec bounding box
    axes[1].imshow(image)
    bbox = result.get("bbox")
    if bbox is not None:
        # bbox format: array [x1, y1, x2, y2]
        bbox_arr = np.array(bbox).flatten()
        x1, y1, x2, y2 = int(bbox_arr[0]), int(bbox_arr[1]), int(bbox_arr[2]), int(bbox_arr[3])
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3, edgecolor='red', facecolor='none'
        )
        axes[1].add_patch(rect)
        conf = float(result.get("score", 0.0))
        axes[1].text(
            x1, y1 - 10,
            f'Conf: {conf:.4f}',
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8)
        )
    axes[1].set_title(f"Détection DFINE\nBBox: [{x1}, {y1}, {x2}, {y2}]", fontsize=12)
    axes[1].axis('off')
    
    # Image de droite: avec mask de segmentation
    # Le mask est dans les coordonnées de la ROI, il faut le replacer sur l'image complète
    axes[2].imshow(image)
    if mask is not None and bbox is not None:
        # Créer un mask complet de la taille de l'image
        full_mask = np.zeros((640, 640), dtype=bool)
        
        # Replacer le mask dans la bbox
        mask_h, mask_w = mask.shape
        # Redimensionner le mask si nécessaire pour correspondre à la bbox
        if mask_h != height or mask_w != width:
            from scipy.ndimage import zoom
            scale_y = height / mask_h
            scale_x = width / mask_w
            mask_resized = zoom(mask.astype(float), (scale_y, scale_x), order=1) > 0.5
        else:
            mask_resized = mask
        
        # Placer le mask redimensionné dans la bbox de l'image complète
        y1_clip = max(0, y1)
        y2_clip = min(640, y1 + mask_resized.shape[0])
        x1_clip = max(0, x1)
        x2_clip = min(640, x1 + mask_resized.shape[1])
        
        mask_y1 = 0 if y1 >= 0 else -y1
        mask_y2 = mask_resized.shape[0] if y2 <= 640 else mask_resized.shape[0] - (y2 - 640)
        mask_x1 = 0 if x1 >= 0 else -x1
        mask_x2 = mask_resized.shape[1] if x2 <= 640 else mask_resized.shape[1] - (x2 - 640)
        
        full_mask[y1_clip:y2_clip, x1_clip:x2_clip] = mask_resized[mask_y1:mask_y2, mask_x1:mask_x2]
        
        # Créer un overlay semi-transparent pour le mask
        mask_overlay = np.zeros((640, 640, 4))
        mask_overlay[full_mask] = [0, 1, 0, 0.6]  # Vert semi-transparent
        axes[2].imshow(mask_overlay)
    
    axes[2].set_title(f"Segmentation MobileSAM\nMask ROI: {mask.shape if mask is not None else 'N/A'}", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(ROOT, "test_pipeline_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    LOG.info("Visualisation sauvegardée dans: %s", output_path)
    plt.close()  # Fermer au lieu de show() pour éviter le blocage dans les tests
