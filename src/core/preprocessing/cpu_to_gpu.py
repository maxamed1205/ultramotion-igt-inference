"""
CPU → GPU Transfer Utility (Process B)
======================================

💡 Mise à jour 20/10/2025 — version “single copy_async”
-------------------------------------------------------

Ce module correspond au **Process B** dans la pipeline Ultramotion (A→B→C→D) :

    A. Acquisition         → service/plus_client.py (PlusServer → RawFrame)
    👉 B. Préprocessing     → core/preprocessing/cpu_to_gpu.py (RawFrame → GpuFrame)
    C. Inférence           → core/inference/detection_and_engine.py (GpuFrame → ResultPacket)
    D. Sortie vers Slicer  → service/slicer_server.py (ResultPacket → 3D Slicer)

Rôle du module
--------------
Assurer **un seul transfert CPU→GPU par frame**, de manière **asynchrone** et **non bloquante**,
avant l’inférence D-FINE + MobileSAM. Ce module effectue :

1. la conversion numpy → torch.Tensor,  
2. la normalisation et mise en forme (channels_first, dtype),  
3. l’allocation en mémoire “pinned” (fixée CPU pour transfert rapide),  
4. la copie asynchrone vers le GPU (copy_async sur un stream dédié).

Une fois la frame convertie en `GpuFrame`, elle est déposée dans `Queue_GPU`
et utilisée directement par `detection_and_engine.py` sans nouvelle copie.

Avantages :
-----------
- ✅ Une seule copie CPU→GPU pour tout le pipeline.
- ✅ Transfert asynchrone (copy + compute se chevauchent).
- ✅ Compatible CUDA streams pour D-FINE / MobileSAM.
- ✅ Facile à synchroniser en aval via `stream.wait_stream()`.

Flux schématique :
------------------
    RawFrame (numpy)
         │
         ▼
    prepare_frame_for_gpu()
         │
     copy_async CUDA stream
         │
         ▼
    GpuFrame(tensor, meta, stream)
         │
         ▼
    Queue_GPU  → detection_and_engine.py

Fonctions principales :
-----------------------
- `prepare_frame_for_gpu(frame, device)` : prépare et transfert CPU→GPU (asynchrone)
- `transfer_to_gpu_async(tensor, stream)` : copie GPU dédiée (optionnelle)
- `process_one_frame()` : consomme une frame RT et alimente Queue_GPU

Ce fichier définit les squelettes des fonctions, à implémenter dans la phase GPU
du Process B. Aucun calcul réel n’est effectué ici.
"""

from typing import Any, Optional, List, Tuple
import logging
import time
import math
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - runtime may not have torch in some test envs
    torch = None

from core.types import RawFrame, GpuFrame
from core.queues.buffers import (
    get_queue_gpu,
    try_dequeue,
    enqueue_nowait_gpu,
)


LOG = logging.getLogger("igt.gpu")
LOG_KPI = logging.getLogger("igt.kpi")


# Module-level transfer runtime / pinned pool state
# A persistent transfer stream to avoid creating one-per-frame
# Module-level transfer runtime / pinned pool state
# A persistent transfer stream to avoid creating one-per-frame
_TRANSFER_STREAM: Optional[Any] = None
# Circular pool of pinned CPU buffers (torch.Tensor)
_PIN_POOL: List[Any] = []
_PIN_POOL_IDX: int = 0
_PIN_POOL_SHAPE: Optional[Tuple[int, int, int, int]] = None  # (N,C,H,W)
# Remember configured pool size (default 2)
_PIN_POOL_SIZE: int = 2


def init_transfer_runtime(device: str = "cuda", pool_size: int = 2, shape_hint: Optional[Tuple[int, int]] = None) -> None:
    """Initialize the persistent transfer stream and optional pinned buffer pool.

    Idempotent: calling multiple times is safe.

    Raises ImportError if torch is not importable.
    If CUDA is unavailable or device does not start with 'cuda' -> no-op (CPU fallback).
    """
    global _TRANSFER_STREAM, _PIN_POOL, _PIN_POOL_SHAPE, _PIN_POOL_IDX, _PIN_POOL_SIZE

    if torch is None:
        raise ImportError("torch required for init_transfer_runtime")

    _PIN_POOL_SIZE = int(pool_size) if pool_size is not None else 2

    # If device is not CUDA or CUDA unavailable, leave runtime as no-op
    use_cuda = str(device).lower().startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        LOG.info("init_transfer_runtime: CUDA not available, running in CPU-only fallback")
        _TRANSFER_STREAM = None
        _PIN_POOL = []
        _PIN_POOL_SHAPE = None
        _PIN_POOL_IDX = 0
        return

    # Create a persistent transfer stream if needed
    if _TRANSFER_STREAM is None:
        try:
            _TRANSFER_STREAM = torch.cuda.Stream(device=device)
        except Exception:
            # fallback to default stream if device string fails
            _TRANSFER_STREAM = torch.cuda.Stream()
    # Optionally create pool
    if shape_hint is not None:
        H, W = shape_hint
        _PIN_POOL_SHAPE = (1, 1, int(H), int(W))
        _PIN_POOL = []
        for _ in range(_PIN_POOL_SIZE):
            try:
                t = torch.empty(_PIN_POOL_SHAPE, dtype=torch.float32).pin_memory()
            except Exception:
                # pin_memory might fail; create non-pinned fallback
                t = torch.empty(_PIN_POOL_SHAPE, dtype=torch.float32)
            _PIN_POOL.append(t)
        _PIN_POOL_IDX = 0
        LOG.debug("init_transfer_runtime: created pinned pool size=%d shape=%s device=%s", _PIN_POOL_SIZE, _PIN_POOL_SHAPE, device)
    else:
        LOG.debug("init_transfer_runtime: stream ready; pinned pool deferred (no shape hint) device=%s", device)
    # Emit KPI about the runtime init (best-effort)
    try:
        from core.monitoring.kpi import safe_log_kpi, format_kpi
        kdata = {
            "ts": time.time(),
            "event": "init_transfer_runtime",
            "device": device,
            "pool_size": _PIN_POOL_SIZE,
            "has_shape_hint": bool(shape_hint),
            "use_cuda": bool(use_cuda),
        }
        safe_log_kpi(format_kpi(kdata))
    except Exception:
        LOG.debug("init_transfer_runtime: KPI emission failed (non-blocking)")


def ensure_pinned_buffer(shape_nchw: Tuple[int, int, int, int]) -> Any:
    """Return a reusable pinned buffer of the requested shape.

    If CUDA is unavailable, return a plain CPU tensor (not pinned).
    If the existing pool is empty or the requested shape differs, recreate the pool.
    The returned tensor is contiguous, dtype float32.
    """
    global _PIN_POOL, _PIN_POOL_IDX, _PIN_POOL_SHAPE, _PIN_POOL_SIZE

    if torch is None:
        raise ImportError("torch required for ensure_pinned_buffer")

    # CPU-only fallback
    if not torch.cuda.is_available():
        # keep internal shape consistent even on CPU fallback
        _PIN_POOL_SHAPE = tuple(shape_nchw)
        return torch.empty(shape_nchw, dtype=torch.float32)

    # Recreate pool if shape mismatch
    if not _PIN_POOL or _PIN_POOL_SHAPE != tuple(shape_nchw):
        LOG.debug("ensure_pinned_buffer: recreating pool for shape=%s (old=%s)", shape_nchw, _PIN_POOL_SHAPE)
        _PIN_POOL = []
        _PIN_POOL_SHAPE = tuple(shape_nchw)
        for _ in range(max(1, int(_PIN_POOL_SIZE))):
            try:
                t = torch.empty(_PIN_POOL_SHAPE, dtype=torch.float32).pin_memory()
            except Exception:
                t = torch.empty(_PIN_POOL_SHAPE, dtype=torch.float32)
            _PIN_POOL.append(t)
        _PIN_POOL_IDX = 0

    idx = _PIN_POOL_IDX
    buf = _PIN_POOL[idx]
    _PIN_POOL_IDX = (_PIN_POOL_IDX + 1) % len(_PIN_POOL)
    # ensure contiguous
    if not buf.is_contiguous():
        buf = buf.contiguous()
    return buf


def warmup_transfer_once(H: int, W: int, device: str = "cuda") -> None:
    """Warm up the CUDA driver by performing a single transfer using the usual path.

    No-op when CUDA is unavailable. Exceptions are swallowed and logged at DEBUG level.
    Emits KPI event 'warmup_copy_async' with duration (ms) when possible.
    """
    if torch is None:
        LOG.debug("warmup_transfer_once: torch not available; skipping warmup")
        return
    if not torch.cuda.is_available() or not str(device).lower().startswith("cuda"):
        LOG.debug("warmup_transfer_once: CUDA not available or device not cuda; skipping warmup")
        return

    try:
        init_transfer_runtime(device=device, pool_size=2, shape_hint=(H, W))
        import numpy as _np
        from core.types import FrameMeta, RawFrame

        dummy = _np.zeros((H, W), dtype=_np.uint8)
        meta = FrameMeta(frame_id=0, ts=time.time())
        rf = RawFrame(image=dummy, meta=meta)
        t0 = time.time()
        try:
            gf = prepare_frame_for_gpu(rf, device=device, config=None)
            # Ensure the copy is finished
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        finally:
            t1 = time.time()
            ms = (t1 - t0) * 1000.0
            try:
                from core.monitoring.kpi import safe_log_kpi, format_kpi
                kdata = {"ts": time.time(), "event": "warmup_copy_async", "duration_ms": round(ms, 2), "H": int(H), "W": int(W)}
                safe_log_kpi(format_kpi(kdata))
            except Exception:
                LOG.debug("warmup_transfer_once: KPI emission failed")
    except Exception as e:
        LOG.debug("warmup_transfer_once failed: %s", e)
        return


# ======================================================================
# 1. Préparation et transfert CPU → GPU
# ======================================================================

def prepare_frame_for_gpu(frame: RawFrame, device: str = "cuda", config: Optional[dict] = None, test_mode: bool = False) -> GpuFrame:
    """
    Prépare une frame CPU (numpy) pour le GPU, via un **seul transfert asynchrone**.

    Étapes attendues (à implémenter) :
      1. Validation de la forme et du dtype (float32, [C,H,W]).
      2. Normalisation des intensités (ex : /255.0 ou mean/std).
      3. Allocation mémoire CPU en pinned memory.
      4. Transfert asynchrone vers GPU via `torch.cuda.Stream`.

    Args:
        frame: RawFrame contenant l'image CPU (numpy array ou buffer équivalent).
        device: cible du transfert ('cuda', 'cuda:0', etc.)
        config: dictionnaire optionnel (normalisation, dtype, scale, etc.)
        test_mode: si True, active les vérifications supplémentaires et nettoyage 
                  explicite des références NumPy pour éliminer les risques de persistance

    Returns:
        GpuFrame : objet contenant le tensor GPU, les métadonnées et le stream CUDA associé.

    Notes :
        - L’objectif est d’assurer que **toute la pipeline C (inférence)** travaille
          sur ce tensor unique sans reconversion CPU→GPU.
        - Le transfert asynchrone permet le chevauchement avec le calcul du frame précédent.
    """
    # Refer to module-level runtime/pool variables early to satisfy Python scoping rules
    global _TRANSFER_STREAM, _PIN_POOL_SIZE

    # Defaults
    cfg = {
        "device": None,
        "normalize": {"mode": "unit", "mean": 0.0, "std": 1.0, "clip": [0.0, 1.0]},
        "kpi": {"enabled": True},
    }
    if config:
        # shallow merge - sufficient for our needs
        cfg.update({k: v for k, v in config.items() if k != "normalize"})
        if "normalize" in config:
            cfg["normalize"] = {**cfg["normalize"], **config["normalize"]}

    if cfg.get("device"):
        device = cfg["device"]

    # Validate torch availability
    if torch is None:
        LOG.error("Torch not available: prepare_frame_for_gpu requires torch")
        raise ImportError("torch is required for prepare_frame_for_gpu")

    # Refer to module-level runtime/pool variables
    global _TRANSFER_STREAM, _PIN_POOL_SIZE

    img = getattr(frame, "image", None)
    if img is None:
        LOG.error("RawFrame has no image attribute")
        raise ValueError("frame.image is required")

    if not isinstance(img, np.ndarray):
        LOG.error("frame.image is not a numpy.ndarray: %s", type(img))
        raise TypeError("frame.image must be a numpy.ndarray")

    # Validate dtype
    if img.dtype not in (np.uint8, np.float32, np.float64):
        LOG.debug("Casting image dtype %s to float32", img.dtype)
        img = img.astype(np.float32)

    # Validate ndim and shape -> expect HxW or HxWx1 or CxHxW
    if img.ndim == 2:
        H, W = img.shape
        img_proc = img.reshape((1, H, W))
    elif img.ndim == 3:
        # possible shapes: H x W x 1 or 1 x H x W or C x H x W
        if img.shape[2] == 1:
            H, W = img.shape[0], img.shape[1]
            img_proc = np.transpose(img, (2, 0, 1)).reshape((1, H, W))
        elif img.shape[0] == 1 or img.shape[0] == 3:
            # interpret as C x H x W
            C = img.shape[0]
            if C != 1:
                LOG.error("Invalid channel count C=%s (expected 1)", C)
                raise ValueError("Only single-channel images supported (C==1)")
            H, W = img.shape[1], img.shape[2]
            img_proc = img.reshape((1, H, W))
        else:
            # fallback: try H x W x C
            C = img.shape[2]
            if C != 1:
                LOG.error("Invalid channel count in trailing axis: %s", C)
                raise ValueError("Only single-channel images supported")
            H, W = img.shape[0], img.shape[1]
            img_proc = np.transpose(img, (2, 0, 1)).reshape((1, H, W))
    else:
        LOG.error("Unsupported image ndim=%d", img.ndim)
        raise ValueError("Unsupported image ndim")

    # Size warnings
    if H < 128 or W < 128 or H > 2048 or W > 2048:
        LOG.warning("Image size HxW=%dx%d outside recommended range", H, W)

    meta = frame.meta

    # Normalization on CPU
    t_norm0 = time.time()
    arr = img_proc.astype(np.float32, copy=False)
    norm_cfg = cfg["normalize"]
    mode = norm_cfg.get("mode", "unit")

    # Detect whether user explicitly provided mean/std in the incoming config
    norm_mean_std_provided = False
    zscore_per_frame = False
    if config and isinstance(config, dict) and "normalize" in config and isinstance(config["normalize"], dict):
        norm_mean_std_provided = ("mean" in config["normalize"]) and ("std" in config["normalize"]) 
        zscore_per_frame = bool(config["normalize"].get("zscore_per_frame", False))

    try:
        if mode == "unit":
            if img.dtype == np.uint8:
                arr = arr / 255.0
            else:
                # ensure values in [0,1]
                arr = np.clip(arr, 0.0, 1.0)
        elif mode == "zscore":
            # Only apply zscore when mean/std explicitly provided, or when
            # zscore_per_frame is requested (discouraged).
            if not norm_mean_std_provided and not zscore_per_frame:
                LOG.warning("zscore sans mean/std explicites — normalisation ignorée (stabilité temporelle)")
            else:
                if zscore_per_frame:
                    LOG.warning("zscore_per_frame=True requested — computing per-frame mean/std (discouraged)")
                    mean = float(arr.mean())
                    std = float(arr.std())
                else:
                    mean = float(norm_cfg.get("mean", 0.0))
                    std = float(norm_cfg.get("std", 1.0))
                if std == 0:
                    std = 1.0
                arr = (arr - mean) / std
        elif mode == "none":
            pass
        else:
            LOG.debug("Unknown normalize.mode=%s, skipping", mode)
    except Exception as e:
        LOG.error("Normalization failed: %s", e)
        raise

    clip = norm_cfg.get("clip")
    if clip and isinstance(clip, (list, tuple)) and len(clip) == 2:
        arr = np.clip(arr, float(clip[0]), float(clip[1]))

    # Add batch dim -> [1,1,H,W]
    arr = arr.reshape((1, 1, H, W))
    t_norm1 = time.time()
    norm_ms = (t_norm1 - t_norm0) * 1000.0
    # Decide fast-path: use Torch vectorized ops for uint8 + simple unit/none normalization
    orig_dtype = img.dtype
    use_cuda = torch.cuda.is_available() and str(device).lower().startswith("cuda")
    fastpath_torch = (orig_dtype == np.uint8) and (mode in ("unit", "none"))
    
    # Track which transfer path is used for KPI
    transfer_path = "unknown"

    # Prepare CPU tensor (possibly using pinned pool) depending on fastpath
    t_pin0 = time.time()
    stream = None
    ten_cpu = None
    try:
        if fastpath_torch:
            # Torch-first path: avoid numpy divisions for uint8
            # We will attempt to fill the pinned buffer directly (avoid intermediate copy_)
            transfer_path = "fastpath_torch"
            if use_cuda:
                if _TRANSFER_STREAM is None:
                    try:
                        init_transfer_runtime(device=device, pool_size=_PIN_POOL_SIZE, shape_hint=None)
                    except Exception:
                        LOG.debug("init_transfer_runtime failed during prepare_frame_for_gpu")
                try:
                    buf = ensure_pinned_buffer((1, 1, H, W))
                    # Try to obtain a numpy view into the CPU tensor memory and write there
                    try:
                        buf_np = buf.numpy()
                        # Ensure source has the same shape as the buffer
                        src = img_proc
                        if src.shape != buf_np.shape:
                            try:
                                src = src.reshape(buf_np.shape)
                            except Exception:
                                src = np.ascontiguousarray(src)
                                src = src.reshape(buf_np.shape)

                        # SINGLE-PASS: use NumPy to write normalized values directly into the
                        # pinned buffer, avoiding subsequent torch.div_/clamp_ passes.
                        if mode == "unit":
                            if LOG.isEnabledFor(logging.DEBUG):
                                t_test0 = time.perf_counter()
                            # multiply into buf_np, casting unsafe allows uint8->float32 direct write
                            np.multiply(src, 1.0 / 255.0, out=buf_np, casting="unsafe")
                            if LOG.isEnabledFor(logging.DEBUG):
                                t_test1 = time.perf_counter()
                                LOG.debug("fastpath torch: normalize fused in %.3f ms", (t_test1 - t_test0) * 1000)
                        else:
                            # direct copy-as-float32
                            if src.dtype != np.float32:
                                buf_np[...] = src.astype(np.float32)
                            else:
                                buf_np[...] = src

                        if clip and isinstance(clip, (list, tuple)) and len(clip) == 2:
                            np.clip(buf_np, float(clip[0]), float(clip[1]), out=buf_np)

                        # Explicitly clean NumPy reference to prevent persistence risks
                        if test_mode:
                            # In test mode, explicitly delete the reference
                            del buf_np
                        
                        ten_cpu = buf
                    except Exception:
                        # Fallback to older method if numpy view not possible
                        transfer_path = "fastpath_fallback"
                        t = torch.from_numpy(np.ascontiguousarray(img_proc)).to(dtype=torch.float32)
                        if t.ndim == 3:
                            t = t.unsqueeze(1)
                        elif t.ndim == 2:
                            t = t.view(1, 1, H, W)
                        if mode == "unit":
                            t.div_(255.0)
                        if clip and isinstance(clip, (list, tuple)) and len(clip) == 2:
                            t.clamp_(float(clip[0]), float(clip[1]))
                        buf.copy_(t, non_blocking=False)
                        ten_cpu = buf
                except Exception:
                    LOG.debug("Pinned pool write failed, falling back to non-pinned tensor")
                    transfer_path = "fastpath_no_pinned"
                    t = torch.from_numpy(np.ascontiguousarray(img_proc)).to(dtype=torch.float32)
                    if mode == "unit":
                        t.div_(255.0)
                    if t.ndim == 3:
                        t = t.unsqueeze(1)
                    elif t.ndim == 2:
                        t = t.view(1, 1, H, W)
                    ten_cpu = t
            else:
                # No CUDA: operate on regular CPU tensor
                transfer_path = "fastpath_cpu_only"
                t = torch.from_numpy(np.ascontiguousarray(img_proc)).to(dtype=torch.float32)
                if mode == "unit":
                    t.div_(255.0)
                if t.ndim == 3:
                    t = t.unsqueeze(1)
                elif t.ndim == 2:
                    t = t.view(1, 1, H, W)
                ten_cpu = t
        else:
            # Numpy-first fallback (existing code path): arr already normalized and shaped
            transfer_path = "numpy_fallback"
            arr_c = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr_c)
            if use_cuda:
                # lazy init stream/runtime
                if _TRANSFER_STREAM is None:
                    try:
                        init_transfer_runtime(device=device, pool_size=_PIN_POOL_SIZE, shape_hint=None)
                    except Exception:
                        LOG.debug("init_transfer_runtime failed during prepare_frame_for_gpu")
                try:
                    buf = ensure_pinned_buffer((1, 1, H, W))
                    buf.copy_(t, non_blocking=False)
                    ten_cpu = buf
                except Exception:
                    LOG.debug("Pinned pool copy failed in numpy path; using original tensor")
                    ten_cpu = t
            else:
                ten_cpu = t
    except Exception as e:
        LOG.debug("prepare_frame_for_gpu: tensor preparation failed: %s", e)
        raise
    t_pin1 = time.time()
    pin_ms = (t_pin1 - t_pin0) * 1000.0

    # Transfer async
    t_copy0 = time.time()

    # Helper to emit KPI once at the end. Captures t_copy0 and t_norm0.
    def _emit_kpi(ts_end_copy, norm_ms_val, pin_ms_val, device_used_val, H_val, W_val, frame_id_val, transfer_path_val):
        try:
            copy_ms_val = (ts_end_copy - t_copy0) * 1000.0
            total_ms_val = (ts_end_copy - t_norm0) * 1000.0
            if cfg.get("kpi", {}).get("enabled", True):
                from core.monitoring.kpi import safe_log_kpi, format_kpi
                kdata = {
                    "ts": time.time(),
                    "event": "copy_async",
                    "device": device_used_val,
                    "transfer_path": transfer_path_val,
                    "H": int(H_val),
                    "W": int(W_val),
                    "norm_ms": round(norm_ms_val, 2),
                    "pin_ms": round(pin_ms_val, 2),
                    "copy_ms": round(copy_ms_val, 2),
                    "total_ms": round(total_ms_val, 2),
                    "frame": frame_id_val,
                }
                safe_log_kpi(format_kpi(kdata))
        except Exception:
            LOG.debug("Failed to emit KPI copy_async")

    device_used = "cpu"
    try:
        if use_cuda:
            # use persistent transfer stream
            if _TRANSFER_STREAM is None:
                try:
                    init_transfer_runtime(device=device, pool_size=_PIN_POOL_SIZE, shape_hint=None)
                except Exception:
                    LOG.debug("init_transfer_runtime failed creating persistent stream")
            stream = _TRANSFER_STREAM
            try:
                with torch.cuda.stream(stream):
                    ten_gpu = ten_cpu.to(device=device, non_blocking=True)
            except Exception:
                # fallback: try default stream
                try:
                    with torch.cuda.stream(torch.cuda.Stream()):
                        ten_gpu = ten_cpu.to(device=device, non_blocking=True)
                except Exception:
                    raise
            device_used = str(ten_gpu.device)
        else:
            ten_gpu = ten_cpu
            device_used = "cpu"
    except torch.cuda.OutOfMemoryError:
        LOG.warning("CUDA OOM during copy_async; falling back to CPU (frame %s)", getattr(meta, "frame_id", None))
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        ten_gpu = ten_cpu
        stream = None
        device_used = "cpu"

    # End of transfer phase: compute copy timings
    t_copy1 = time.time()
    copy_ms = (t_copy1 - t_copy0) * 1000.0

    # Build GpuFrame
    # Validate tensor invariants
    try:
        assert ten_gpu.dtype == torch.float32
        assert ten_gpu.ndim == 4 and ten_gpu.shape[0] == 1 and ten_gpu.shape[1] == 1
        devstr = str(ten_gpu.device)
        assert ("cuda" in devstr) or ("cpu" in devstr)
    except AssertionError as e:
        LOG.error("Prepared tensor failed invariants: %s", e)
        raise

    # KPI emission (single place)
    _emit_kpi(t_copy1, norm_ms, pin_ms, device_used, H, W, getattr(meta, "frame_id", None), transfer_path)

    if LOG.isEnabledFor(logging.DEBUG):
        total_ms = (t_copy1 - t_norm0) * 1000.0
        LOG.debug(
            "Prepared tensor NCHW=%s frame_id=%s on device=%s (pin_ms=%.2f, copy_ms=%.2f, total_ms=%.2f)",
            ten_gpu.shape,
            getattr(meta, "frame_id", "?"),
            device_used,
            pin_ms,
            copy_ms,
            total_ms,
        )

    # Debug: indicate which path was chosen for this frame
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "prepare_frame_for_gpu: fastpath=%s device=%s HxW=%dx%d",
            "torch" if fastpath_torch else "numpy",
            device,
            H,
            W,
        )

    return GpuFrame(tensor=ten_gpu, meta=meta, stream=stream)


def transfer_to_gpu_async(tensor: Any, stream_transfer: Optional[Any] = None, device: str = "cuda") -> Any:
    """
    Effectue un transfert asynchrone CPU→GPU sur un stream CUDA fourni.

    Args:
        tensor: tensor CPU (torch.Tensor sur device='cpu').
        stream_transfer: stream CUDA dédié au transfert.
        device: cible GPU ('cuda', 'cuda:0', etc.).

    Returns:
        tensor GPU (torch.Tensor sur le device cible).

    Comportement attendu :
        - Utiliser `with torch.cuda.stream(stream_transfer):`
          puis `tensor.to(device, non_blocking=True)`.
        - Retourner le tensor GPU pour réutilisation immédiate.
    """
    if torch is None:
        raise ImportError("torch required for transfer_to_gpu_async")

    # Accept numpy arrays for ergonomics
    if isinstance(tensor, np.ndarray):
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(tensor))
        except Exception as e:
            LOG.debug("Failed to convert numpy to tensor: %s", e)
            raise

    # After possible conversion, ensure we have a torch.Tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor or numpy.ndarray")

    target = str(device).lower() if device is not None else "cpu"

    # If target is not cuda or CUDA not available -> CPU path (no-op)
    if (not target.startswith("cuda")) or (not torch.cuda.is_available()):
        # KPI: no-op (explicit)
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi
            kdata = {
                "ts": time.time(),
                "event": "transfer_async_noop",
                "device_target": target,
                "bytes": int(tensor.element_size() * tensor.numel()) if hasattr(tensor, "element_size") else 0,
                "shape": tuple(tensor.shape),
                "stream_used": int(bool(stream_transfer)),
                "already_on_device": int(getattr(tensor, "is_cuda", False) and str(getattr(tensor, "device", "")).lower() == target),
                "copy_ms": 0.0,
            }
            safe_log_kpi(format_kpi(kdata))
        except Exception:
            # KPI is optional
            pass
        return tensor

    # Already on the desired CUDA device -> no-op
    if tensor.is_cuda and str(tensor.device).lower() == target:
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi
            kdata = {
                "ts": time.time(),
                "event": "transfer_async_noop",
                "device_target": target,
                "bytes": int(tensor.element_size() * tensor.numel()),
                "shape": tuple(tensor.shape),
                "stream_used": int(bool(stream_transfer)),
                "already_on_device": 1,
                "copy_ms": 0.0,
            }
            safe_log_kpi(format_kpi(kdata))
        except Exception:
            pass
        return tensor

    # Ensure CPU tensor is contiguous
    if not tensor.is_contiguous():
        LOG.debug("transfer_to_gpu_async: making tensor contiguous")
        try:
            tensor = tensor.contiguous()
        except Exception:
            LOG.debug("contiguous() failed; proceeding with original tensor")

    # Try pinning memory if tensor is on CPU and CUDA available
    if (not tensor.is_cuda) and torch.cuda.is_available():
        try:
            tensor = tensor.pin_memory()
            LOG.debug("transfer_to_gpu_async: pin_memory applied")
        except Exception:
            LOG.debug("pin_memory failed or unavailable; continuing without it")

    # If we're doing a cross-device GPU->GPU copy, log it (debug)
    if tensor.is_cuda and str(tensor.device).lower() != target:
        LOG.debug("transfer_to_gpu_async: cross-device GPU copy %s -> %s", str(tensor.device), target)

    # Compute bytes for KPI
    try:
        bytes_count = int(tensor.element_size() * tensor.numel())
    except Exception:
        bytes_count = 0

    # Perform non-blocking transfer, optionally on provided stream
    t0 = time.time()
    try:
        if stream_transfer is None:
            ten_gpu = tensor.to(device=target, non_blocking=True)
        else:
            with torch.cuda.stream(stream_transfer):
                ten_gpu = tensor.to(device=target, non_blocking=True)
    except torch.cuda.OutOfMemoryError:
        LOG.warning("CUDA OOM during transfer_to_gpu_async")
        # Emit KPI for OOM
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi
            kdata = {
                "ts": time.time(),
                "event": "transfer_async_oom",
                "device_target": target,
                "bytes": bytes_count,
                "shape": tuple(tensor.shape),
                "stream_used": int(bool(stream_transfer)),
                "already_on_device": 0,
            }
            safe_log_kpi(format_kpi(kdata))
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise

    t1 = time.time()
    copy_ms = (t1 - t0) * 1000.0

    # Emit KPI for successful transfer
    try:
        from core.monitoring.kpi import safe_log_kpi, format_kpi
        kdata = {
            "ts": time.time(),
            "event": "transfer_async",
            "device_target": target,
            "bytes": bytes_count,
            "shape": tuple(ten_gpu.shape),
            "stream_used": int(bool(stream_transfer)),
            "already_on_device": 0,
            "copy_ms": round(copy_ms, 2),
        }
        safe_log_kpi(format_kpi(kdata))
    except Exception:
        pass

    # Debug log summary for successful copies only
    LOG.debug(
        "transfer_to_gpu_async: shape=%s bytes=%d target=%s stream_used=%s copy_ms=%.2f",
        tuple(ten_gpu.shape),
        bytes_count,
        target,
        bool(stream_transfer),
        copy_ms,
    )

    return ten_gpu

