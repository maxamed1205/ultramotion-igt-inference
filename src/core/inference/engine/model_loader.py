import io
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - tests may run without torch
    torch = None  # type: ignore
    nn = None  # type: ignore

from core.inference.MobileSAM.mobilesam_loader import build_mobilesam_model  # noqa
from core.inference.engine.gpu_optim import _optimize_mobilesam, resolve_precision

LOG = logging.getLogger("igt.inference")

# Global cache and lock for loaded models
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_MODEL_LOCK = threading.Lock()


def initialize_models(model_paths: Dict[str, str], device: str = "cuda") -> Dict[str, Any]:
    """Charge et initialise les modèles D-FINE et MobileSAM.

    Args:
        model_paths: dictionnaire contenant les chemins {'dfine': ..., 'mobilesam': ...}
        device: 'cuda' ou 'cpu'

    Returns:
        dict {'dfine': model_dfine, 'mobilesam': model_sam}
    """
    if not isinstance(model_paths, dict) or "dfine" not in model_paths or "mobilesam" not in model_paths:
        raise ValueError("model_paths must contain 'dfine' and 'mobilesam'")

    if torch is None:
        raise ImportError("torch is required to initialize models")

    # Resolve device availability
    req_device = device or "cpu"
    if req_device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA unavailable, fallback to CPU")
        req_device = "cpu"

    torch_device = torch.device(req_device)

    # Resolve precision
    precision_req = model_paths.get("precision", "auto")

    precision_effective = resolve_precision(precision_req, torch_device)

    # Build cache key
    cache_key = f"{torch_device.type}:{precision_effective}:{model_paths['dfine']}:{model_paths['mobilesam']}"

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            LOG.debug("Using cached models for %s", cache_key)
            return _MODEL_CACHE[cache_key]

    def make_stub_model(name: str):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                # Accept 3-channel images to better mimic vision models
                self.conv = nn.Conv2d(3, 1, 3, padding=1)

            def forward(self, x):
                return torch.sigmoid(self.conv(x))

        m = Dummy()
        try:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        except Exception:
            LOG.debug("Failed to set eval/grad flags on Dummy stub")
        LOG.warning("Using Dummy stub model for %s", name)
        return m

    def load_model_async(path: str):
        """Charge un modèle depuis `path` avec gestion spéciale.

        - si le path indique MobileSAM -> utilise build_mobilesam_model(checkpoint=path)
        - sinon tente torch.load; si on obtient un state_dict, essaie de reconstruire
          D-FINE via `d_fine.dfine.build_model` (si disponible) puis charger les poids
        - en cas d'erreur, renvoie un stub via make_stub_model
        """
        try:
            if path and "mobilesam" in path.lower():
                try:
                    return build_mobilesam_model(checkpoint=path)
                except Exception:
                    LOG.exception("build_mobilesam_model failed for %s", path)
                    return make_stub_model(path)

            with open(path, "rb") as f:
                buf = f.read()
            bio = io.BytesIO(buf)
            # Use weights_only=True on PyTorch >= 2.5 to avoid FutureWarning
            try:
                obj = torch.load(bio, map_location="cpu", weights_only=True)
            except TypeError:
                # Older PyTorch versions don't accept weights_only
                obj = torch.load(bio, map_location="cpu")

            if isinstance(obj, torch.nn.Module):
                return obj

            if isinstance(obj, dict):
                state = None
                for k in ("state_dict", "model_state_dict", "weights", "net"):
                    if k in obj:
                        state = obj[k]
                        break
                if state is None:
                    try:
                        import collections

                        if all(hasattr(v, "dtype") for v in obj.values()):
                            state = obj
                    except Exception:
                        state = None

                if state is not None and isinstance(state, dict):
                    try:
                        try:
                            from d_fine.dfine import build_model as build_dfine_model  # type: ignore
                        except Exception:
                            build_dfine_model = None  # type: ignore

                        if build_dfine_model is not None:
                            df = build_dfine_model()
                            try:
                                df.load_state_dict(state)
                            except Exception:
                                LOG.debug("Loaded state_dict could not be applied to rebuilt DFINE; returning stub")
                                return make_stub_model(path)
                            return df
                        else:
                            LOG.debug("d_fine.build_model not available; cannot reconstruct DFINE from state_dict")
                            return make_stub_model(path)
                    except Exception:
                        LOG.exception("Failed to reconstruct model from state dict for %s", path)
                        return make_stub_model(path)

            LOG.warning("Loaded object from %s is not a Module or dict; using stub", path)
            return make_stub_model(path)
        except Exception:
            LOG.exception("Failed to load model from %s", path)
            return make_stub_model(path)

    # Parallel CPU loading
    t_start = time.perf_counter()
    dfine_load_ms = sam_load_ms = 0.0
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_dfine = executor.submit(load_model_async, model_paths["dfine"])
        fut_sam = executor.submit(load_model_async, model_paths["mobilesam"])
        try:
            t0 = time.perf_counter()
            dfine = fut_dfine.result()
            t1 = time.perf_counter()
            dfine_load_ms = (t1 - t0) * 1000.0
        except Exception:
            LOG.exception("Exception while loading dfine model")
            dfine = make_stub_model("dfine")
            dfine_load_ms = 0.0
        try:
            t0 = time.perf_counter()
            sam = fut_sam.result()
            t1 = time.perf_counter()
            sam_load_ms = (t1 - t0) * 1000.0
        except Exception:
            LOG.exception("Exception while loading mobilesam model")
            sam = make_stub_model("mobilesam")
            sam_load_ms = 0.0

    try:
        with torch.inference_mode():
            for m in (dfine, sam):
                try:
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    m.float()
                except Exception:
                    LOG.debug("Model %s does not support parameters iteration", type(m))
    except Exception:
        LOG.debug("torch.inference_mode not available, skipping")

    compiled_flag = bool(model_paths.get("compile", False))
    ts_flag = bool(model_paths.get("torchscript", False))

    if torch_device := torch.device(req_device) if hasattr(torch, 'device') else torch.device(req_device):
        pass

    if torch_device.type == "cuda":
        try:
            stream_dfine = torch.cuda.Stream()
            stream_sam = torch.cuda.Stream()
            with torch.inference_mode():
                with torch.cuda.stream(stream_dfine):
                    dfine = dfine.to(torch_device, non_blocking=True)
                with torch.cuda.stream(stream_sam):
                    sam = sam.to(torch_device, non_blocking=True)
            torch.cuda.synchronize()

            if precision_effective == "fp16":
                try:
                    dfine = dfine.half()
                    sam = sam.half()
                except Exception:
                    LOG.debug("Half cast failed")
            if precision_effective == "bf16":
                try:
                    dfine = dfine.bfloat16()
                    sam = sam.bfloat16()
                except Exception:
                    LOG.debug("BFloat16 cast failed")
        except Exception:
            LOG.exception("GPU transfer failed, falling back to CPU models")
            dfine = dfine.to("cpu") if hasattr(dfine, "to") else dfine
            sam = sam.to("cpu") if hasattr(sam, "to") else sam
            torch_device = torch.device("cpu")
    else:
        stream_dfine = None
        stream_sam = None

    mobilesam_channels_last = False
    mobilesam_checkpoint_off = False
    mobilesam_warmup = False

    try:
        if sam is not None:
            try:
                if torch_device.type == "cuda" and stream_sam is not None:
                    with torch.cuda.stream(stream_sam):
                        applied = _optimize_mobilesam(sam, torch_device, precision_effective)
                else:
                    applied = _optimize_mobilesam(sam, torch_device, precision_effective)

                mobilesam_channels_last = bool(applied.get("channels_last"))
                mobilesam_checkpoint_off = bool(applied.get("checkpoint_off"))
                try:
                    LOG.info(
                        f"MobileSAM ready: channels_last={mobilesam_channels_last}, "
                        f"fp16={(precision_effective=='fp16')}, checkpoint_off={mobilesam_checkpoint_off}"
                    )
                except Exception:
                    pass

                if torch_device.type == "cuda" and hasattr(sam, "image_encoder"):
                    try:
                        if stream_sam is not None:
                            with torch.cuda.stream(stream_sam):
                                with torch.inference_mode():
                                    if precision_effective == "fp16":
                                        try:
                                            # Use new torch.amp API on recent PyTorch (>=2.5)
                                            if hasattr(torch, "amp"):
                                                with torch.amp.autocast("cuda", dtype=torch.float16):
                                                    dummy = torch.zeros((1, 3, 1024, 1024), device=torch_device, dtype=torch.float32)
                                                    _ = sam.image_encoder(dummy)
                                            else:
                                                # Fallback to older autocast behaviour
                                                try:
                                                    from torch.cuda.amp import autocast

                                                    with autocast():
                                                        dummy = torch.zeros((1, 3, 1024, 1024), device=torch_device, dtype=torch.float32)
                                                        _ = sam.image_encoder(dummy)
                                                except Exception:
                                                    dummy = torch.zeros((1, 3, 1024, 1024), device=torch_device, dtype=torch.float32)
                                                    _ = sam.image_encoder(dummy)
                                        except Exception:
                                            dummy = torch.zeros((1, 3, 1024, 1024), device=torch_device, dtype=torch.float32)
                                            _ = sam.image_encoder(dummy)
                                    else:
                                        dummy = torch.zeros((1, 3, 1024, 1024), device=torch_device, dtype=torch.float32)
                                        _ = sam.image_encoder(dummy)
                                mobilesam_warmup = True
                        else:
                            mobilesam_warmup = False
                    except RuntimeError as e:
                        LOG.warning("MobileSAM warm-up skipped due to runtime error: %s", e)
                        mobilesam_warmup = False
            except Exception:
                LOG.exception("MobileSAM optimization/warmup failed")
    except Exception:
        LOG.debug("MobileSAM optimization block failed", exc_info=True)

    try:
        if ts_flag:
            for name, mdl in [("dfine", dfine), ("mobilesam", sam)]:
                try:
                    scripted = torch.jit.script(mdl)
                    if name == "dfine":
                        dfine = scripted
                    else:
                        sam = scripted
                except Exception:
                    LOG.warning("TorchScript failed for %s", name)

        if compiled_flag and hasattr(torch, "compile"):
            for name, mdl in [("dfine", dfine), ("mobilesam", sam)]:
                try:
                    compiled = torch.compile(mdl, mode="reduce-overhead")
                    if name == "dfine":
                        dfine = compiled
                    else:
                        sam = compiled
                except Exception:
                    LOG.debug("torch.compile failed for %s", name)
    except Exception as e:
        LOG.debug("Optional optimization block failed: %s", e)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0

    try:
        from core.monitoring.kpi import safe_log_kpi, format_kpi
        gpu_mem_MB = 0.0
        if torch_device.type == "cuda":
            try:
                gpu_mem_MB = torch.cuda.memory_allocated(torch_device) / 1e6
            except Exception:
                gpu_mem_MB = -1.0
        LOG.debug("GPU memory allocated after init: %.1f MB", gpu_mem_MB)

        kdata = {
            "ts": time.time(),
            "event": "init_models",
            "device": str(torch_device),
            "precision": precision_effective,
            "compiled": compiled_flag,
            "torchscripted": ts_flag,
            "dfine_load_ms": round(dfine_load_ms, 2),
            "sam_load_ms": round(sam_load_ms, 2),
            "total_ms": round(total_ms, 2),
            "gpu_mem_MB": round(gpu_mem_MB, 1),
            "dfine_path": model_paths["dfine"],
            "mobilesam_path": model_paths["mobilesam"],
            "mobilesam_channels_last": bool(mobilesam_channels_last),
            "mobilesam_checkpoint_off": bool(mobilesam_checkpoint_off),
            "mobilesam_warmup": bool(mobilesam_warmup),
        }
        safe_log_kpi(format_kpi(kdata))
    except Exception:
        LOG.debug("Failed to emit KPI init_models")

    out = {
        "device": torch_device,
        "dfine": dfine,
        "mobilesam": sam,
        "meta": {
            "dfine_path": model_paths["dfine"],
            "mobilesam_path": model_paths["mobilesam"],
            "precision": precision_effective,
            "compiled": compiled_flag,
            "torchscripted": ts_flag,
            "ts": time.time(),
        },
        "streams": {
            "dfine": stream_dfine,
            "mobilesam": stream_sam,
        },
    }

    with _MODEL_LOCK:
        _MODEL_CACHE[cache_key] = out

    return out
