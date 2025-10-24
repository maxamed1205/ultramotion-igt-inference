from typing import Any, Optional
import logging
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - allow import when torch missing
    torch = None  # type: ignore

LOG = logging.getLogger("igt.inference")


def run_segmentation(sam_model: Any, roi: np.ndarray) -> Optional[np.ndarray]:
    """Exécute MobileSAM sur la ROI et retourne le mask binaire.

    Uses AMP autocast (fp16) when CUDA is available. Safely handles CPU-only
    environments by returning None and logging a warning.
    """
    if sam_model is None or roi is None:
        return None

    if torch is None:
        LOG.warning("torch unavailable; skipping segmentation")
        return None

    try:
        # Determine device from model parameters if possible
        try:
            device = next(sam_model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sam_model.eval()

        # Build tensor: accept HxW, HxWx1, HxWx3, or CxHxW
        roi_tensor = torch.as_tensor(roi, dtype=torch.float32, device=device)
        if roi_tensor.ndim == 2:
            roi_tensor = roi_tensor.unsqueeze(0)  # 1,H,W
        if roi_tensor.ndim == 3:
            # If last dim is channels (H,W,C), permute
            if roi_tensor.shape[-1] == 3 and roi_tensor.shape[0] != 3:
                roi_tensor = roi_tensor.permute(2, 0, 1)
            # If single channel, expand to 3
            if roi_tensor.shape[0] == 1:
                roi_tensor = roi_tensor.repeat(3, 1, 1)
        roi_tensor = roi_tensor.unsqueeze(0)  # 1,C,H,W

        # Run model under autocast FP16 when on CUDA (PyTorch>=2.5 uses torch.amp)
        if device.type == "cuda" and hasattr(torch, "amp"):
            with torch.inference_mode():
                # new API: torch.amp.autocast(device, dtype=...)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = sam_model(roi_tensor)
        else:
            with torch.inference_mode():
                outputs = sam_model(roi_tensor)

        # Extract mask from common output shapes
        mask = None
        if isinstance(outputs, dict):
            if "masks" in outputs:
                mask = outputs["masks"][0]
            elif "pred_masks" in outputs:
                mask = outputs["pred_masks"][0]
            else:
                # fallback to first value
                try:
                    mask = next(iter(outputs.values()))
                except Exception:
                    mask = outputs
        elif isinstance(outputs, (list, tuple)):
            mask = outputs[0]
        else:
            mask = outputs

        if hasattr(mask, "detach"):
            return mask.detach().cpu().numpy()
        try:
            return np.array(mask)
        except Exception:
            return None
    except Exception as e:
        LOG.exception("Segmentation failed: %s", e)
        return None
