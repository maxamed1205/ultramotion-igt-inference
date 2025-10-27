from typing import Any, Dict, Optional, Tuple
import logging
import time
import numpy as np

from core.types import GpuFrame, ResultPacket
from core.queues.buffers import get_queue_gpu, get_queue_out, try_dequeue, enqueue_nowait_out
from core.inference.engine.inference_dfine import run_detection
from core.inference.engine.inference_sam import run_segmentation
from core.inference.engine.postprocess import compute_mask_weights

LOG = logging.getLogger("igt.inference")
LOG_KPI = logging.getLogger("igt.kpi")


def prepare_inference_inputs(frame_t: np.ndarray, dfine_model: Any, sam_model: Any, tau_conf: float = 0.0001) -> Dict[str, Any]:
    """Orchestration complète des étapes 0 → 3.

    0. Passe l'image dans D-FINE → bbox/conf.
    1. Si conf < τ_conf → renvoie state_hint='LOST'.
    2. Crop ROI autour de la bbox.
    3. Passe la ROI dans MobileSAM.
    4. Calcule les pondérations spatiales (W_edge/W_in/W_out).

    Returns:
        dictionnaire prêt pour visibility_fsm.evaluate_visibility().
    """
    # Stable output shape required by the pipeline
    out: Dict[str, Any] = {
        "state_hint": "LOST",
        "bbox": None,
        "conf": 0.0,
        "mask": None,
        "weights": None,
        "timestamp": None,
    }

    try:
        # Try to extract an optional timestamp from the frame-like object
        ts = None
        try:
            ts = getattr(getattr(frame_t, "meta", None), "ts", None)
        except Exception:
            ts = None
        out["timestamp"] = ts

        LOG.debug("Starting prepare_inference_inputs: tau_conf=%s frame_shape=%s", tau_conf, getattr(frame_t, "shape", None))

        # 1) Run D-FINE to obtain bbox and confidence
        try:
            bbox_t, conf_t = run_detection(dfine_model, frame_t)
        except Exception as e:
            LOG.exception("D-FINE detection failed: %s", e)
            out.update({"state_hint": "LOST", "conf": 0.0})
            return out

        out["bbox"] = bbox_t
        out["conf"] = float(conf_t) if conf_t is not None else 0.0

        if bbox_t is None or (conf_t is None) or (conf_t < tau_conf):
            LOG.debug("DFINE result: no bbox or below threshold (conf=%s, tau=%s). state=LOST", conf_t, tau_conf)
            out["state_hint"] = "LOST"
            return out

        # 2) Convert bbox to integer pixel coordinates (x1,y1,x2,y2)
        try:
            b = np.array(bbox_t, dtype=np.float32).flatten()
            if b.size != 4:
                raise ValueError("bbox must have 4 elements")
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            LOG.debug("DFINE bbox coordinates: x1=%d, y1=%d, x2=%d, y2=%d (conf=%.3f)", x1, y1, x2, y2, conf_t)
        except Exception as e:
            LOG.exception("Invalid bbox format from DFINE: %s", e)
            out["state_hint"] = "LOST"
            return out

        # 3) Extract full image as uint8 RGB for SAM
        full_image = None
        try:
            arr = frame_t
            if hasattr(frame_t, "tensor"):
                arr = getattr(frame_t, "tensor")

            # Convert to numpy uint8 RGB (H, W, 3)
            if hasattr(arr, "detach"):  # PyTorch tensor
                if arr.ndim == 4 and arr.shape[0] == 1:
                    # [1, C, H, W] -> [H, W, C]
                    arr_np = arr[0].permute(1, 2, 0).detach().cpu().numpy()
                    # Denormalize if needed: if max <= 1.0, assume normalized
                    if arr_np.max() <= 1.0:
                        arr_np = (arr_np * 255).astype(np.uint8)
                    else:
                        arr_np = arr_np.astype(np.uint8)
                    
                    # Convert to RGB if grayscale
                    if arr_np.shape[-1] == 1:
                        arr_np = np.repeat(arr_np, 3, axis=-1)
                    
                    full_image = arr_np
                else:
                    LOG.warning("Unexpected torch tensor shape: %s", arr.shape)
            elif isinstance(arr, np.ndarray):
                # Handle various numpy array formats
                if arr.ndim == 2:
                    # H x W grayscale -> RGB
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                    full_image = np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3:
                    if arr.shape[-1] in (1, 3):
                        # H x W x C
                        if arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)
                        if arr.shape[-1] == 1:
                            full_image = np.repeat(arr, 3, axis=-1)
                        else:
                            full_image = arr
                    else:
                        # C x H x W -> H x W x C
                        arr = np.transpose(arr, (1, 2, 0))
                        if arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)
                        if arr.shape[-1] == 1:
                            full_image = np.repeat(arr, 3, axis=-1)
                        else:
                            full_image = arr
                elif arr.ndim == 4:
                    # B x C x H x W or B x H x W x C
                    first = arr[0]
                    if first.shape[0] in (1, 3):
                        # C x H x W
                        arr = np.transpose(first, (1, 2, 0))
                    else:
                        arr = first
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                    if arr.shape[-1] == 1:
                        full_image = np.repeat(arr, 3, axis=-1)
                    else:
                        full_image = arr
        except Exception as e:
            LOG.exception("Failed to extract full image for SAM: %s", e)
            full_image = None

        if full_image is None:
            LOG.debug("Failed to extract full image; returning LOST")
            out["state_hint"] = "LOST"
            return out

        # 4) Run MobileSAM with full image + bbox (new predictor API)
        try:
            LOG.debug(f"[MASK DEBUG] Calling run_segmentation with bbox={bbox_t}")
            mask = run_segmentation(sam_model, full_image, bbox_xyxy=bbox_t)
        except Exception as e:
            LOG.exception("SAM segmentation failed: %s", e)
            mask = None

        if mask is None:
            LOG.debug("SAM returned no mask; state=LOST")
            out["state_hint"] = "LOST"
            out["mask"] = None
            out["weights"] = None
            return out

        # 5) Compute spatial weights
        try:
            # Normalize mask to binary float array
            mask_arr = np.array(mask)
            mask_bin = (mask_arr > 0.5).astype(np.uint8)
            W_edge, W_in, W_out = compute_mask_weights(mask_bin)
        except Exception as e:
            LOG.exception("compute_mask_weights failed: %s", e)
            out["state_hint"] = "LOST"
            out["mask"] = mask if mask is not None else None
            out["weights"] = None
            return out

        # 6) Build the successful output
        out.update({
            "state_hint": "VISIBLE",
            "bbox": np.array([x1, y1, x2, y2], dtype=np.int32),
            "conf": float(conf_t),
            "mask": mask_bin,
            "weights": (W_edge, W_in, W_out),
        })

        LOG.debug("DFINE OK, SAM OK, mask_shape=%s, state=VISIBLE", getattr(mask_bin, "shape", None))
        return out

    except Exception as e:
        # Never raise — return a stable dict describing the failure
        LOG.exception("Unexpected error in prepare_inference_inputs: %s", e)
        return out


def run_inference(frame_tensor: GpuFrame, stream_infer: Any = None) -> Tuple[ResultPacket, float]:
    """Exécute (mock) l’inférence GPU et retourne un ResultPacket minimal."""
    # Contract:
    # - Input: GpuFrame-like object with .tensor (ndarray/torch.Tensor) and .meta (FrameMeta)
    # - Output: tuple(result_packet: dict-like, latency_ms: float)
    t0 = time.perf_counter()
    try:
        # Extract image/tensor and meta safely
        arr = getattr(frame_tensor, "tensor", frame_tensor)
        meta = getattr(frame_tensor, "meta", None)
        frame_id = getattr(getattr(frame_tensor, "meta", None), "frame_id", None)
        ts = getattr(getattr(frame_tensor, "meta", None), "ts", None)

        # Attempt to find initialized models from model_loader cache if available
        dfine_model = None
        sam_model = None
        try:
            from core.inference.engine import model_loader as _ml

            cache = getattr(_ml, "_MODEL_CACHE", None)
            if cache:
                # pick the first cached entry (most recent/only)
                first = next(iter(cache.values()), None)
                if isinstance(first, dict):
                    dfine_model = first.get("dfine")
                    sam_model = first.get("mobilesam") or first.get("sam")
        except Exception:
            # best-effort: models may be provided elsewhere; keep None if not found
            dfine_model = dfine_model
            sam_model = sam_model

        # Prepare inputs / run DFINE -> MobileSAM -> postprocess
        prepared = prepare_inference_inputs(arr, dfine_model, sam_model)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Ensure mandatory keys and build final ResultPacket-like dict
        state = prepared.get("state_hint", "LOST")
        mask = prepared.get("mask", None)
        bbox = prepared.get("bbox", None)
        score = float(prepared.get("conf", 0.0) or 0.0)
        weights = prepared.get("weights", None)

        result = {
            "frame_id": frame_id,
            "timestamp": ts if ts is not None else time.time(),
            "bbox": bbox,
            "mask": mask,
            "score": score,
            "state": state,
            "weights": weights,
            # latency embedded for downstream KPI/inspection
            "latency_ms": round(latency_ms, 3),
        }

        # Logging + KPI (best-effort)
        try:
            LOG.debug("Inference completed: frame=%s state=%s conf=%.3f latency=%.1f ms", frame_id, state, score, latency_ms)
        except Exception:
            try:
                LOG.debug("Inference completed: state=%s", state)
            except Exception:
                pass

        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi

            kdata = {
                "ts": time.time(),
                "event": "infer_frame",
                "frame": frame_id,
                "latency_ms": f"{latency_ms:.3f}",
                "conf": f"{score:.3f}",
                "state": state,
            }
            safe_log_kpi(format_kpi(kdata))
        except Exception:
            LOG.debug("Failed to emit KPI infer_frame")

        # Return result and measured latency
        return result, latency_ms

    except Exception as e:
        # Always measure latency even on error
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        LOG.exception("run_inference failed: %s", e)

        # Minimal stable ResultPacket-like dict on error
        err = {
            "state": "ERROR",
            "bbox": None,
            "mask": None,
            "score": 0.0,
            "timestamp": getattr(getattr(frame_tensor, "meta", None), "ts", time.time()),
            "frame_id": getattr(getattr(frame_tensor, "meta", None), "frame_id", None),
            "latency_ms": round(latency_ms, 3),
        }
        return err, latency_ms


def fuse_outputs(mask: Any, score: float, state: str) -> ResultPacket:
    """Fusionne les sorties et renvoie un ResultPacket standardisé."""
    # Build a minimal ResultPacket-compatible dict. We prefer a simple dict
    # because the pipeline historically accepts both dataclass and dict.
    try:
        pkt = {
            "mask": mask,
            "score": float(score or 0.0),
            "state": state or "LOST",
            "timestamp": time.time(),
        }
        return pkt  # type: ignore[return-value]
    except Exception:
        return {"mask": None, "score": 0.0, "state": "ERROR", "timestamp": time.time()}  # type: ignore[return-value]


def process_inference_once(models: Any = None) -> None:
    """Consomme une GpuFrame, exécute une inférence (mock) et place le résultat en sortie."""
    # Dequeue a GpuFrame (non-blocking). If none available, just return.
    q_gpu = get_queue_gpu()
    gf = try_dequeue(q_gpu)
    if gf is None:
        return

    # Extract simple metadata for logging/KPI
    frame_id = getattr(getattr(gf, "meta", None), "frame_id", None)
    ts = getattr(getattr(gf, "meta", None), "ts", None)

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug("Dequeued GpuFrame for inference: %s", frame_id)

    try:
        # Determine optional stream if present on the GpuFrame
        stream = getattr(gf, "stream", None)

        # Run the full inference pipeline (DFINE -> SAM -> postprocess)
        result, latency_ms = run_inference(gf, stream)

        # Logging
        try:
            LOG.info("Inference done: frame=%s state=%s latency=%.1f ms", frame_id, result.get("state"), float(latency_ms or 0.0))
        except Exception:
            LOG.info("Inference done: state=%s", result.get("state"))

        # Emit KPI for the inference loop
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi

            kdata = {
                "ts": time.time(),
                "event": "infer_loop",
                "frame": frame_id,
                "latency_ms": f"{latency_ms:.3f}",
                "state": result.get("state"),
            }
            safe_log_kpi(format_kpi(kdata))
        except Exception:
            try:
                # Fallback to KPI logger
                LOG_KPI.info("infer_loop frame=%s latency_ms=%.3f state=%s", frame_id, float(latency_ms or 0.0), result.get("state"))
            except Exception:
                LOG.debug("Failed to emit KPI infer_loop")

        # Enqueue the ResultPacket to the output queue (best-effort)
        try:
            q_out = get_queue_out()
            ok = enqueue_nowait_out(q_out, result)  # type: ignore[arg-type]
            if not ok:
                LOG.warning("Out queue full, result for frame %s dropped", frame_id)
        except Exception as e:
            LOG.exception("Failed to enqueue result: %s", e)

    except Exception as e:
        # Never let exceptions from a single frame crash the loop.
        LOG.exception("process_inference_once failed for frame %s: %s", frame_id, e)
        return
