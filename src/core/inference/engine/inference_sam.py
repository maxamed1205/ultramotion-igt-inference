from typing import Any, Optional, Tuple
import logging
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - allow import when torch missing
    torch = None  # type: ignore

LOG = logging.getLogger("igt.inference")


def run_segmentation(sam_model: Any, image: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Exécute MobileSAM sur l'image complète avec une bbox et retourne le mask binaire.

    Args:
        sam_model: SamPredictor instance wrapping the MobileSAM model
        image: Image complète uint8 RGB (H, W, 3) dans le même repère que la bbox
        bbox_xyxy: Bounding box au format [x1, y1, x2, y2] en pixels de l'image (optional)
    
    Returns:
        Binary mask (H, W) numpy array or None
    
    Uses AMP autocast (fp16) when CUDA is available. Safely handles CPU-only
    environments by returning None and logging a warning.
    
    Note: Si bbox_xyxy est None, cette fonction tombera en mode legacy (ROI).
    """
    if sam_model is None or image is None:
        return None

    if torch is None:
        LOG.warning("torch unavailable; skipping segmentation")
        return None

    # Check if we have a SamPredictor instance (new API) or raw model (legacy)
    has_predictor_api = hasattr(sam_model, 'set_image') and hasattr(sam_model, 'predict')
    
    # NEW API: Use SamPredictor with full image + bbox
    if has_predictor_api and bbox_xyxy is not None:
        try:
            # Ensure image is uint8 RGB
            if image.dtype != np.uint8:
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            else:
                image_uint8 = image
            
            # Ensure RGB (convert grayscale if needed)
            if image_uint8.ndim == 2:
                image_uint8 = np.stack([image_uint8] * 3, axis=-1)
            elif image_uint8.shape[-1] == 1:
                image_uint8 = np.repeat(image_uint8, 3, axis=-1)
            
            # Check model dtype and handle FP16 models
            # The issue: SamPredictor creates FP32 tensors, but if model is FP16, we get dtype mismatch
            # Solution: Wrap set_image/predict with autocast to ensure dtype consistency
            try:
                model = getattr(sam_model, 'model', sam_model)
                model_dtype = next(model.parameters()).dtype
                use_fp16 = model_dtype == torch.float16
                device_type = next(model.parameters()).device.type
                LOG.debug(f"SAM model dtype: {model_dtype}, device: {device_type}, use_fp16: {use_fp16}")
            except Exception as e:
                use_fp16 = False
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                LOG.debug(f"Could not detect SAM dtype: {e}, defaulting to FP32")
            
            # Set the image (computes embeddings) with appropriate autocast
            if use_fp16 and device_type == "cuda":
                # Use autocast for FP16 models
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    sam_model.set_image(image_uint8)
            else:
                sam_model.set_image(image_uint8)
            
            # Prepare bbox: ensure it's in the right format [x1, y1, x2, y2]
            bbox_np = np.array(bbox_xyxy, dtype=np.float32).flatten()
            if bbox_np.size != 4:
                LOG.warning("Invalid bbox shape: %s, falling back to legacy mode", bbox_np.shape)
                return _run_segmentation_legacy(sam_model, image)
            
            # Predict with bbox prompt (also under autocast if FP16)
            if use_fp16 and device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    masks, scores, _ = sam_model.predict(
                        box=bbox_np,
                        multimask_output=False
                    )
            else:
                masks, scores, _ = sam_model.predict(
                    box=bbox_np,
                    multimask_output=False
                )
            
            # Extract the first (and only) mask
            if masks is not None and len(masks) > 0:
                mask = masks[0]  # (H, W)
                LOG.debug(f"SAM prediction successful: mask shape={mask.shape}, score={scores[0] if len(scores) > 0 else 'N/A'}")
                return mask.astype(bool)
            else:
                LOG.warning("SAM predict returned no masks")
                return None
                
        except Exception as e:
            LOG.exception("SAM segmentation with predictor API failed: %s", e)
            return None
    
    # LEGACY API: Direct model call with ROI (fallback for compatibility)
    else:
        LOG.debug("Using legacy SAM inference (no bbox or no predictor API)")
        return _run_segmentation_legacy(sam_model, image)


def _run_segmentation_legacy(sam_model: Any, roi: np.ndarray) -> Optional[np.ndarray]:
    """Legacy SAM inference for ROI-based segmentation (original implementation)."""
    if sam_model is None or roi is None:
        return None

    if torch is None:
        LOG.warning("torch unavailable; skipping segmentation")
        return None

    try:
        # Get the underlying model if sam_model is a SamPredictor
        model = getattr(sam_model, 'model', sam_model)
        
        # Determine device from model parameters if possible
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()

        # Build tensor: accept HxW, HxWx1, HxWx3, or CxHxW
        roi_tensor = torch.as_tensor(roi, dtype=torch.float32, device=device)
        if roi_tensor.ndim == 2:
            # H,W -> 1,H,W
            roi_tensor = roi_tensor.unsqueeze(0)
        if roi_tensor.ndim == 3:
            # H,W,C -> C,H,W
            if roi_tensor.shape[-1] == 3 and roi_tensor.shape[0] != 3:
                roi_tensor = roi_tensor.permute(2, 0, 1)
            # 1,H,W -> 3,H,W
            if roi_tensor.shape[0] == 1:
                roi_tensor = roi_tensor.repeat(3, 1, 1)
            # C,H,W -> 1,C,H,W
            roi_tensor = roi_tensor.unsqueeze(0)
        if roi_tensor.ndim == 4:
            # Si déjà batché, ne rien faire
            pass

        # Ensure the model receives an image tensor in 3xHxW format (Sam.forward
        # expects each element's 'image' to be CxHxW, not batched). Build a small
        # wrapper that squeezes a leading batch dim if present and include
        # original_size so the model can postprocess masks correctly.
        H, W = roi.shape[0], roi.shape[1]
        image_for_model = roi_tensor
        # If we accidentally created a batched tensor (1,C,H,W), remove the batch
        # dimension so Sam.forward receives CxHxW per element.
        if image_for_model.ndim == 4 and image_for_model.shape[0] == 1:
            image_for_model = image_for_model.squeeze(0)

        # Run model under autocast FP16 when on CUDA (PyTorch>=2.5 uses torch.amp)
        if device.type == "cuda" and hasattr(torch, "amp"):
            with torch.inference_mode():
                # new API: torch.amp.autocast(device, dtype=...)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = sam_model([{"image": image_for_model, "original_size": (H, W)}], multimask_output=False)
        else:
            with torch.inference_mode():
                outputs = sam_model([{"image": image_for_model, "original_size": (H, W)}], multimask_output=False)

        LOG.debug(f"SAM outputs type: {type(outputs)}, is list: {isinstance(outputs, list)}")
        if isinstance(outputs, list) and len(outputs) > 0:
            LOG.debug(f"SAM outputs[0] type: {type(outputs[0])}, keys: {outputs[0].keys() if isinstance(outputs[0], dict) else 'N/A'}")

        # Extract mask from common output shapes
        mask = None
        if isinstance(outputs, dict):
            if "masks" in outputs:
                mask = outputs["masks"]
            elif "pred_masks" in outputs:
                mask = outputs["pred_masks"]
            else:
                # fallback to first value
                try:
                    mask = next(iter(outputs.values()))
                except Exception:
                    mask = outputs
        elif isinstance(outputs, (list, tuple)):
            if len(outputs) > 0:
                result_dict = outputs[0]
                if isinstance(result_dict, dict) and "masks" in result_dict:
                    mask = result_dict["masks"]
                else:
                    mask = result_dict
            else:
                mask = None
        else:
            mask = outputs

        LOG.debug(f"Extracted mask type: {type(mask)}, shape attempt: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")

        # Convert to numpy and normalize common output shapes to a 2D mask (HxW)
        try:
            if hasattr(mask, "detach"):
                m = mask.detach().cpu().numpy()
            else:
                m = np.array(mask)
        except Exception:
            return None

        # m may be BxCxHxW, CxHxW, BxHxW, or HxW. Try to reduce to HxW:
        try:
            if isinstance(m, np.ndarray):
                if m.ndim == 4:
                    # B x C x H x W -> choose first batch and first channel
                    if m.shape[0] == 0:
                        return None
                    return m[0, 0]
                if m.ndim == 3:
                    # C x H x W or B x H x W
                    # If first dim is 1 -> squeeze
                    if m.shape[0] == 1:
                        return m[0]
                    # If shape looks like channels (3, H, W), there's no mask -> return None
                    if m.shape[0] == 3:
                        # no valid binary mask
                        return None
                    # Otherwise assume B x H x W -> take first
                    return m[0]
                if m.ndim == 2:
                    return m
        except Exception:
            return None

        return None
    except Exception as e:
        LOG.exception("Segmentation failed: %s", e)
        return None
