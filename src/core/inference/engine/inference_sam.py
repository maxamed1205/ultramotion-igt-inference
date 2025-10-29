from typing import Any, Optional, Tuple
import logging
import time
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - allow import when torch missing
    torch = None  # type: ignore

LOG = logging.getLogger("igt.inference")


def run_segmentation(
    sam_model: Any,
    image: Any,
    bbox_xyxy: Optional[np.ndarray] = None,
    as_numpy: bool = False,
) -> Optional[Any]:
    """Exécute MobileSAM sur l'image complète avec une bbox et retourne le mask binaire.
    
    ⚠️ as_numpy=True doit être réservé à la visualisation ou à l'export Slicer.
    En production, utiliser as_numpy=False pour maintenir le flux GPU-resident.
    
    Args:
        sam_model: Le modèle SAM ou SamPredictor
        image: Image d'entrée (numpy array ou tensor)
        bbox_xyxy: Bounding box au format [x1, y1, x2, y2]
        as_numpy: Si True, retourne numpy array (mode legacy/visualization).
                 Si False, retourne tensor GPU (mode GPU-resident production).
    
    Returns:
        Mask binaire (numpy array si as_numpy=True, tensor GPU si as_numpy=False)
    """
    if sam_model is None or image is None:
        return None

    if torch is None:
        LOG.warning("torch unavailable; skipping segmentation")
        return None

    # Avertissement pour usage as_numpy en production
    if as_numpy:
        LOG.warning("⚠️ as_numpy=True triggers CPU conversion — reserved for visualization or export only.")

    has_predictor_api = hasattr(sam_model, 'set_image') and hasattr(sam_model, 'predict')

    # --- MODE AVEC PREDICTOR ---
    if has_predictor_api and bbox_xyxy is not None:
        try:
            # 🔹 Conversion image
            if image.dtype != np.uint8:
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            else:
                image_uint8 = image

            if image_uint8.ndim == 2:
                image_uint8 = np.stack([image_uint8] * 3, axis=-1)
            elif image_uint8.shape[-1] == 1:
                image_uint8 = np.repeat(image_uint8, 3, axis=-1)

            # 🔹 Type du modèle
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

            # --- [SAM PROMPT MODE] Full image + bbox ---
            bbox_np = np.array(bbox_xyxy, dtype=np.float32).flatten()
            if bbox_np.size != 4:
                LOG.warning("Invalid bbox shape: %s, falling back to legacy mode", bbox_np.shape)
                return _run_segmentation_legacy(sam_model, image, as_numpy)

            LOG.debug(f"[SAM DEBUG] Received bbox_xyxy={bbox_np}")

            # 🔹 Embedding complet
            if use_fp16 and device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    sam_model.set_image(image_uint8)
            else:
                sam_model.set_image(image_uint8)

            LOG.debug(f"[SAM DEBUG] Image embedding set on full image {image_uint8.shape}")

            # 🔹 Prédiction avec bbox - GPU-resident par défaut
            if use_fp16 and device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    masks, scores, _ = sam_model.predict(box=bbox_np, multimask_output=False, as_numpy=as_numpy)
            else:
                masks, scores, _ = sam_model.predict(box=bbox_np, multimask_output=False, as_numpy=as_numpy)

            # --- Résultat ---
            if masks is not None and len(masks) > 0:
                mask = masks[0]
                LOG.debug(f"SAM prediction successful: mask shape={mask.shape}, score={scores[0] if len(scores) > 0 else 'N/A'}")
                
                # Mode visualisation (legacy)
                if as_numpy:
                    LOG.debug("Returning SAM mask as numpy (CPU) — visualization mode")
                    result = mask.astype(bool) if isinstance(mask, np.ndarray) else mask.detach().cpu().numpy().astype(bool)
                    
                    # KPI instrumentation
                    try:
                        from core.monitoring.kpi import safe_log_kpi, format_kpi
                        safe_log_kpi(format_kpi({
                            "event": "sam_output",
                            "as_numpy": int(as_numpy),
                            "device": "cpu",
                            "shape": str(getattr(result, "shape", None)),
                        }))
                    except Exception:
                        LOG.debug("KPI sam_output skipped")
                    
                    return result

                # Mode production GPU-resident
                if isinstance(mask, np.ndarray):
                    # Déterminer device à partir du modèle SAM
                    try:
                        model = getattr(sam_model, 'model', sam_model)
                        device = next(model.parameters()).device
                    except Exception:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Conversion directe vers GPU
                    mask_t = torch.from_numpy(mask).to(device, non_blocking=True)
                    
                    # KPI instrumentation
                    try:
                        from core.monitoring.kpi import safe_log_kpi, format_kpi
                        safe_log_kpi(format_kpi({
                            "event": "sam_output",
                            "as_numpy": int(as_numpy),
                            "device": str(device),
                            "shape": str(mask_t.shape),
                        }))
                    except Exception:
                        LOG.debug("KPI sam_output skipped")
                    
                    return mask_t
                else:
                    # KPI instrumentation
                    try:
                        from core.monitoring.kpi import safe_log_kpi, format_kpi
                        safe_log_kpi(format_kpi({
                            "event": "sam_output",
                            "as_numpy": int(as_numpy),
                            "device": str(getattr(mask, "device", "cpu")),
                            "shape": str(getattr(mask, "shape", None)),
                        }))
                    except Exception:
                        LOG.debug("KPI sam_output skipped")
                    
                    return mask  # déjà tensor GPU
            else:
                LOG.warning("SAM predict returned no masks")
                return None

        except Exception as e:
            LOG.exception("SAM segmentation with predictor API failed: %s", e)
            return None

    # --- LEGACY MODE ---
    LOG.debug("Using legacy SAM inference (no bbox or no predictor API)")
    return _run_segmentation_legacy(sam_model, image, as_numpy)


def _run_segmentation_legacy(sam_model: Any, roi: np.ndarray, as_numpy: bool = False) -> Optional[Any]:
    """Legacy SAM inference for ROI-based segmentation (original implementation).
    
    ⚠️ as_numpy=True doit être réservé à la visualisation ou à l'export Slicer.
    En production, utiliser as_numpy=False pour maintenir le flux GPU-resident.
    
    Args:
        sam_model: Le modèle SAM 
        roi: ROI numpy array 
        as_numpy: Si True, retourne numpy array (visualization). Si False, tensor GPU.
        
    Returns:
        Mask binaire (numpy array si as_numpy=True, tensor GPU si as_numpy=False)
    """
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

        # Convert to numpy and normalize common output shapes to a 2D mask (HxW) OR keep as GPU tensor
        if mask is None:
            return None

        # Instrumentation KPI pour debug visuel
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi
            safe_log_kpi(format_kpi({
                "ts": time.time(),
                "event": "sam_mask_output",
                "as_numpy": int(as_numpy),
                "device": str(getattr(mask, "device", "cpu")),
            }))
        except Exception:
            pass

        if hasattr(mask, "detach"):
            # GPU-resident
            if not as_numpy:
                return mask  # ne pas .detach() -> préserver graphe pour fine-tuning éventuel
            else:
                LOG.debug("SAM legacy mode: exporting mask to numpy (CPU)")
                return mask.detach().cpu().numpy()
        else:
            # non-torch : fallback numpy
            try:
                m = np.array(mask)
            except Exception:
                return None

        # Réduction dimensionnelle seulement pour le mode as_numpy=True
        if as_numpy:
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
        else:
            # GPU tensor : on ne le réduit pas ici, le ResultPacket ou postprocess s'en charge
            return mask

        return None
    except Exception as e:
        LOG.exception("Segmentation failed: %s", e)
        return None
