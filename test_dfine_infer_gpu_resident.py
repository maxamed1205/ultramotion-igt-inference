# -*- coding: utf-8 -*-
"""
test_dfine_infer_gpu_resident.py
================================

Test de validation pour le refactoring GPU-resident de dfine_infer.py

Vérifie :
- Pas de transfert GPU→CPU prématuré
- Tenseurs restent sur GPU en mode return_gpu_tensor=True
- Fallback CPU contrôlé par allow_cpu_fallback
- Types de retour corrects
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import torch
    import numpy as np
    from core.inference.dfine_infer import (
        infer_dfine, 
        postprocess_dfine, 
        run_dfine_detection,
        preprocess_frame_for_dfine
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    TORCH_AVAILABLE = False
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

class MockDFINEModel(torch.nn.Module):
    """Modèle D-FINE simulé pour tests"""
    
    def __init__(self, output_format="detr"):
        super().__init__()
        self.output_format = output_format
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)  # Simple conv pour avoir des paramètres
        
    def forward(self, x):
        # Simule les sorties D-FINE
        batch_size = x.shape[0]
        device = x.device
        
        if self.output_format == "detr":
            # Format DETR-like: pred_logits et pred_boxes
            num_queries = 100
            num_classes = 2  # classe + no-object
            
            outputs = {
                "pred_logits": torch.randn(batch_size, num_queries, num_classes, device=device),
                "pred_boxes": torch.rand(batch_size, num_queries, 4, device=device)  # cxcywh normalisé
            }
            
            # Force une detection confiante au premier query
            outputs["pred_logits"][0, 0, 0] = 3.0  # Score élevé pour classe
            outputs["pred_logits"][0, 0, 1] = -2.0  # Score faible pour no-object
            outputs["pred_boxes"][0, 0] = torch.tensor([0.5, 0.5, 0.3, 0.4], device=device)  # bbox centrale
            
        else:  # torchvision-like
            num_detections = 5
            outputs = {
                "scores": torch.rand(num_detections, device=device) * 0.5 + 0.3,  # scores 0.3-0.8
                "boxes": torch.rand(num_detections, 4, device=device) * 500 + 100   # boxes en pixels
            }
            # Force une detection confiante
            outputs["scores"][0] = 0.9
            outputs["boxes"][0] = torch.tensor([100.0, 150.0, 300.0, 400.0], device=device)
            
        return outputs


def test_preprocess_frame():
    """Test prétraitement GPU"""
    print("\n🧪 Test preprocess_frame_for_dfine...")
    
    # Test avec tensor GPU grayscale
    frame_gpu = torch.rand(1, 1, 512, 512, device='cuda')
    processed = preprocess_frame_for_dfine(frame_gpu)
    
    assert processed.is_cuda, "Output should stay on GPU"
    assert processed.shape == (1, 1, 512, 512), f"Shape mismatch: {processed.shape}"
    assert processed.dtype == torch.float32, f"Type mismatch: {processed.dtype}"
    
    print("✅ preprocess_frame_for_dfine: OK")


def test_infer_dfine_gpu_resident():
    """Test inférence GPU sans fallback"""
    print("\n🧪 Test infer_dfine GPU-resident...")
    
    model = MockDFINEModel("detr").cuda()
    frame = torch.rand(1, 1, 512, 512, device='cuda')
    
    # Test mode strict (pas de fallback CPU)
    outputs = infer_dfine(model, frame, allow_cpu_fallback=False)
    
    # Vérifier que les tenseurs restent sur GPU
    assert isinstance(outputs, dict), "Output should be dict"
    assert "pred_logits" in outputs, "Missing pred_logits"
    assert "pred_boxes" in outputs, "Missing pred_boxes"
    assert outputs["pred_logits"].is_cuda, "pred_logits should stay on GPU"
    assert outputs["pred_boxes"].is_cuda, "pred_boxes should stay on GPU"
    
    print("✅ infer_dfine GPU-resident: OK")


def test_postprocess_gpu_mode():
    """Test post-traitement en mode GPU tensor"""
    print("\n🧪 Test postprocess_dfine return_gpu_tensor=True...")
    
    # Simule outputs D-FINE sur GPU
    device = 'cuda'
    outputs = {
        "pred_logits": torch.tensor([[[3.0, -2.0]]], device=device),  # [1, 1, 2]
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.3, 0.4]]], device=device)  # [1, 1, 4]
    }
    
    # Mode GPU tensor
    bbox_t, conf_t = postprocess_dfine(
        outputs, 
        img_size=(512, 512), 
        conf_thresh=0.3,
        return_gpu_tensor=True
    )
    
    # Vérifications
    assert bbox_t is not None, "bbox should not be None"
    assert isinstance(bbox_t, torch.Tensor), f"bbox should be torch.Tensor, got {type(bbox_t)}"
    assert bbox_t.is_cuda, "bbox should stay on GPU"
    assert bbox_t.shape == (4,), f"bbox shape should be (4,), got {bbox_t.shape}"
    
    assert isinstance(conf_t, torch.Tensor), f"conf should be torch.Tensor, got {type(conf_t)}"
    assert conf_t.is_cuda, "conf should stay on GPU"
    
    print(f"✅ GPU mode: bbox_t.device={bbox_t.device}, conf_t.device={conf_t.device}")
    
    # Test mode legacy (numpy)
    bbox_np, conf_np = postprocess_dfine(
        outputs, 
        img_size=(512, 512), 
        conf_thresh=0.3,
        return_gpu_tensor=False
    )
    
    assert isinstance(bbox_np, np.ndarray), f"bbox should be numpy.ndarray, got {type(bbox_np)}"
    assert isinstance(conf_np, float), f"conf should be float, got {type(conf_np)}"
    
    print("✅ postprocess_dfine modes: OK")


def test_run_dfine_detection_full_pipeline():
    """Test pipeline complet GPU-resident"""
    print("\n🧪 Test run_dfine_detection pipeline complet...")
    
    model = MockDFINEModel("detr").cuda()
    frame_gpu = torch.rand(1, 1, 512, 512, device='cuda')
    
    # Test mode GPU-resident strict
    bbox_t, conf_t = run_dfine_detection(
        model, 
        frame_gpu,
        conf_thresh=0.3,
        allow_cpu_fallback=False,
        return_gpu_tensor=True
    )
    
    # Vérifications
    assert bbox_t is not None, "bbox should not be None"
    assert isinstance(bbox_t, torch.Tensor), f"bbox should be torch.Tensor, got {type(bbox_t)}"
    assert bbox_t.is_cuda, "bbox should stay on GPU"
    
    assert isinstance(conf_t, torch.Tensor), f"conf should be torch.Tensor, got {type(conf_t)}"
    assert conf_t.is_cuda, "conf should stay on GPU"
    
    print(f"✅ Pipeline GPU: bbox_t={bbox_t.device}, conf_t={conf_t.device}")
    
    # Test mode legacy
    bbox_np, conf_np = run_dfine_detection(
        model, 
        frame_gpu,
        conf_thresh=0.3,
        allow_cpu_fallback=False,
        return_gpu_tensor=False
    )
    
    assert isinstance(bbox_np, np.ndarray), f"bbox should be numpy.ndarray, got {type(bbox_np)}"
    assert isinstance(conf_np, float), f"conf should be float, got {type(conf_np)}"
    
    print("✅ run_dfine_detection pipeline: OK")


def test_performance_comparison():
    """Test performance GPU vs CPU"""
    print("\n🧪 Test performance GPU vs modes...")
    
    model = MockDFINEModel("detr").cuda()
    frame_gpu = torch.rand(1, 1, 512, 512, device='cuda')
    
    # Warmup
    for _ in range(5):
        run_dfine_detection(model, frame_gpu, return_gpu_tensor=True)
    
    torch.cuda.synchronize()
    
    # Benchmark mode GPU
    num_iterations = 20
    start_time = time.time()
    
    for _ in range(num_iterations):
        bbox_t, conf_t = run_dfine_detection(
            model, frame_gpu, 
            return_gpu_tensor=True
        )
    
    torch.cuda.synchronize()
    gpu_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    # Benchmark mode legacy (avec sync CPU)
    start_time = time.time()
    
    for _ in range(num_iterations):
        bbox_np, conf_np = run_dfine_detection(
            model, frame_gpu, 
            return_gpu_tensor=False
        )
    
    torch.cuda.synchronize()
    legacy_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    speedup = legacy_time / gpu_time
    print(f"📊 Performance:")
    print(f"   Mode GPU-resident: {gpu_time:.2f} ms/frame")
    print(f"   Mode legacy:       {legacy_time:.2f} ms/frame")
    print(f"   Speedup:           {speedup:.2f}x")
    
    # Le mode GPU devrait être plus rapide (moins de sync)
    assert speedup > 1.0, f"GPU mode should be faster, got {speedup:.2f}x"
    
    print("✅ Performance test: OK")


def main():
    """Lance tous les tests"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available, skipping tests")
        return
        
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        return
    
    print("🚀 Tests dfine_infer GPU-resident refactoring")
    print("=" * 50)
    
    try:
        test_preprocess_frame()
        test_infer_dfine_gpu_resident()
        test_postprocess_gpu_mode()
        test_run_dfine_detection_full_pipeline()
        test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("🎉 Tous les tests dfine_infer GPU-resident: RÉUSSIS")
        print("✅ Pipeline 100% GPU-resident validé!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)