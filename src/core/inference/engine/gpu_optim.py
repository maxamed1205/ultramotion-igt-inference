from typing import Any, Dict
import logging

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - allow tests without torch
    torch = None  # type: ignore
    nn = None  # type: ignore

LOG = logging.getLogger("igt.inference")


def _is_mobilesam(model: Any) -> bool:
    """Heuristic to detect a MobileSAM model instance.

    Returns True if the model has characteristic MobileSAM attributes or
    class names (image_encoder, prompt_encoder, mask_decoder) or if its
    class name contains 'Sam'/'SAM'/'Mobile'.
    """
    if model is None:
        return False
    # common attribute-based check
    try:
        if hasattr(model, "image_encoder") and hasattr(model, "prompt_encoder") and hasattr(model, "mask_decoder"):
            return True
    except Exception:
        # be conservative on errors
        pass
    # class-name heuristic
    try:
        name = type(model).__name__
        if any(x in name.lower() for x in ("sam", "mobile")):
            return True
    except Exception:
        pass
    return False


def _optimize_mobilesam(model: Any, device_obj: "torch.device", precision: str) -> Dict[str, bool]:
    """Apply safe, non-destructive optimizations for MobileSAM models.

    - set memory_format to channels_last where supported
    - disable checkpoint flags if present (use_checkpoint)
    - enable backend flags (cudnn.benchmark, allow_tf32) when CUDA

    Returns a dict with booleans describing what was applied.
    This function is a no-op if the model doesn't look like MobileSAM.
    """
    applied = {
        "channels_last": False,
        "checkpoint_off": False,
        "cudnn_benchmark": False,
        "allow_tf32": False,
    }

    if not _is_mobilesam(model):
        return applied

    # channels_last: set model to use channels_last memory format if possible
    try:
        for m in model.modules():
            try:
                if hasattr(m, "to"):
                    m.to(memory_format=torch.channels_last)
            except Exception:
                pass
        applied["channels_last"] = True
    except Exception:
        LOG.debug("channels_last optimization failed for MobileSAM", exc_info=True)

    # disable checkpoint flags if present
    try:
        if hasattr(model, "use_checkpoint"):
            try:
                setattr(model, "use_checkpoint", False)
                applied["checkpoint_off"] = True
            except Exception:
                LOG.debug("Failed to set model.use_checkpoint=False", exc_info=True)
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "use_checkpoint"):
            try:
                setattr(model.image_encoder, "use_checkpoint", False)
                applied["checkpoint_off"] = True
            except Exception:
                LOG.debug("Failed to set image_encoder.use_checkpoint=False", exc_info=True)
    except Exception:
        LOG.debug("Checkpoint toggle failed", exc_info=True)

    # backend flags
    try:
        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.benchmark = True
                applied["cudnn_benchmark"] = True
            except Exception:
                LOG.debug("Failed to set cudnn.benchmark=True", exc_info=True)
        if device_obj.type == "cuda":
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    applied["allow_tf32"] = True
            except Exception:
                LOG.debug("Failed to set allow_tf32", exc_info=True)
    except Exception:
        LOG.debug("Backend flag setup failed", exc_info=True)

    try:
        LOG.info("MobileSAM optimizations applied: %s", applied)
    except Exception:
        pass

    # Post-optimization preparation: clear grads, enforce channels_last and eval
    try:
        try:
            for p in model.parameters():
                try:
                    # Clear any attached gradients to avoid retention
                    p.grad = None
                except Exception:
                    pass
        except Exception:
            pass
        try:
            model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass
    except Exception:
        LOG.debug("Post-optimization prep failed", exc_info=True)

    return applied


def resolve_precision(req: str, device_obj: "torch.device") -> str:
    """Résout dynamiquement la précision effective ('fp32', 'fp16', 'bf16').

    Règles:
    - device CPU -> 'fp32'
    - req == 'auto' -> fp16 si CC >= 7, sinon fp32
    - req == 'bf16' -> vérifie le support bf16 sinon fallback fp32
    - sinon retourne la requête
    """
    if device_obj.type == "cpu":
        return "fp32"
    if req == "auto":
        try:
            if torch is None:
                return "fp32"
            cc = torch.cuda.get_device_capability(device_obj) if hasattr(torch.cuda, "get_device_capability") else (6, 0)
            major = cc[0] if isinstance(cc, (list, tuple)) else 6
            return "fp16" if major >= 7 else "fp32"
        except Exception:
            return "fp32"
    if req == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        LOG.warning("bf16 not supported; fallback to fp32")
        return "fp32"
    return req
