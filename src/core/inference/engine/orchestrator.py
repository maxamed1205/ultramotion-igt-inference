from typing import Any, Dict, Optional, Tuple
import logging
import time
import numpy as np
import torch

from core.types import GpuFrame, ResultPacket
from core.queues.buffers import get_queue_gpu, get_queue_out, try_dequeue, enqueue_nowait_out
from core.inference.engine.inference_dfine import run_detection
from core.inference.engine.inference_sam import run_segmentation
from core.inference.engine.postprocess import compute_mask_weights
from core.monitoring import monitor

LOG = logging.getLogger("igt.inference")
LOG_KPI = logging.getLogger("igt.kpi")


def prepare_inference_inputs(frame_t: np.ndarray, dfine_model: Any, sam_model: Any, tau_conf: float = 0.0001, sam_as_numpy: bool = False) -> Dict[str, Any]:
    """Orchestration complète des étapes 0 → 3.

    0. Passe l'image dans D-FINE → bbox/conf.
    1. Si conf < τ_conf → renvoie state_hint='LOST'.
    2. Crop ROI autour de la bbox.
    3. Passe la ROI dans MobileSAM.
    4. Calcule les pondérations spatiales (W_edge/W_in/W_out).

    Args:
        frame_t: Input frame tensor or array
        dfine_model: Detection model
        sam_model: Segmentation model  
        tau_conf: Confidence threshold
        sam_as_numpy: If True, converts tensors to numpy for SAM (legacy mode).
                     If False, keeps tensors on GPU for SAM (GPU-resident mode).

    Returns:
        dictionnaire prêt pour visibility_fsm.evaluate_visibility().
    """
    # Garde contre mode legacy non intentionnel
    if sam_as_numpy:
        LOG.warning("SAM running in legacy CPU mode (as_numpy=True)")
    
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

        # 1) Run D-FINE to obtain bbox and confidence (GPU-resident mode)
        try:
            # ⏱️ Mesure de la durée d’inférence D-FINE
            t_dfine0 = time.perf_counter()
            bbox_t, conf_t = run_detection(
                dfine_model,
                frame_t,
                allow_cpu_fallback=False,    # Mode strict GPU-resident
                return_gpu_tensor=True,      # Garde tenseurs sur GPU
                stream=None,                 # Use default stream
                conf_thresh=tau_conf         # Pass threshold directly
            )
            t_dfine1 = time.perf_counter()

            # 🧩 Enregistre la durée dans le monitor
            try:
                from core.monitoring import monitor
                monitor.record_interstage("cpu_gpu_to_proc", (t_dfine1 - t_dfine0) * 1000.0)
            except Exception:
                LOG.debug("Failed to record D-FINE latency")

        except Exception as e:
            LOG.exception("D-FINE detection failed: %s", e)
            out.update({"state_hint": "LOST", "conf": 0.0})
            return out


        out["bbox"] = bbox_t
        # Conversion GPU→CPU seulement pour scalar conf si nécessaire
        if hasattr(conf_t, 'item'):
            conf_scalar = float(conf_t.item())  # Sync GPU→CPU pour scalar uniquement
        else:
            conf_scalar = float(conf_t) if conf_t is not None else 0.0
        out["conf"] = conf_scalar

        if bbox_t is None or (conf_t is None) or (conf_scalar < tau_conf):
            LOG.debug("DFINE result: no bbox or below threshold (conf=%s, tau=%s). state=LOST", conf_scalar, tau_conf)
            out["state_hint"] = "LOST"
            return out

        # 2) Convert bbox to integer pixel coordinates (x1,y1,x2,y2) - GPU-resident
        try:
            # Assure que bbox_t est un tensor GPU
            if isinstance(bbox_t, torch.Tensor):
                if not bbox_t.is_cuda:
                    bbox_t = bbox_t.cuda()  # Move to GPU if needed
                b = bbox_t.flatten()  # Reste sur GPU
            else:
                # Legacy numpy array → convertir en tensor GPU
                b = torch.as_tensor(bbox_t, dtype=torch.float32, device="cuda").flatten()
                bbox_t = b.view(4)  # Update bbox_t to be GPU tensor
                
            if b.numel() != 4:
                raise ValueError("bbox must have 4 elements")
            
            # Extraction minimale CPU seulement pour logs (4 scalars)
            x1, y1, x2, y2 = [int(v) for v in b.tolist()]  # Sync ponctuelle pour scalars
            LOG.debug("DFINE bbox coordinates: x1=%d, y1=%d, x2=%d, y2=%d (conf=%.3f)", x1, y1, x2, y2, conf_scalar)
            
            # KPI monitoring device
            LOG.debug(f"bbox_device={bbox_t.device}, bbox_dtype={bbox_t.dtype}, conf={conf_scalar:.3f}")
        except Exception as e:
            LOG.exception("Invalid bbox format from DFINE: %s", e)
            out["state_hint"] = "LOST"
            return out

        # 3) Extract full image for SAM - GPU-resident path or legacy CPU path
        full_image = None
        arr = getattr(frame_t, "tensor", frame_t)

        if hasattr(arr, "detach") and not sam_as_numpy:
            # ✅ GPU-resident path : SAM reçoit directement le tensor CUDA
            try:
                if arr.ndim == 4 and arr.shape[0] == 1:
                    # [1, C, H, W] -> [H, W, C] on GPU
                    full_image = arr[0].permute(1, 2, 0).contiguous()
                    
                    # Normalisation (0–1) si besoin
                    if full_image.dtype != torch.float32:
                        full_image = full_image.to(torch.float32)
                    if full_image.max() > 1.0:
                        full_image = full_image / 255.0
                    
                    # Convert to RGB if grayscale (on GPU)
                    if full_image.shape[-1] == 1:
                        full_image = full_image.repeat(1, 1, 3)
                        
                    LOG.debug("GPU-resident path: tensor shape=%s, device=%s", full_image.shape, full_image.device)
                else:
                    LOG.warning("Unexpected tensor shape for GPU SAM input: %s", arr.shape)
                    full_image = None
            except Exception as e:
                LOG.exception("Failed GPU SAM path: %s", e)
                full_image = None

        else:
            # 🧩 Legacy CPU path (DEPRECATED: sam_as_numpy=False by default)
            # Only executed when explicitly requested for visualization/export
            LOG.warning("🚨 Legacy CPU path activated - performance degraded (sam_as_numpy=True)")
            try:
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
                LOG.debug("Legacy CPU path: array shape=%s, type=%s", 
                         getattr(full_image, 'shape', 'None'), type(full_image))
            except Exception as e:
                LOG.exception("Failed CPU SAM path: %s", e)
                full_image = None

        if full_image is None:
            LOG.debug("Failed to extract full image; returning LOST")
            out["state_hint"] = "LOST"
            return out

        # 4) Run MobileSAM with full image + bbox (new predictor API)
        try:
            # KPI avant SAM - monitoring GPU continuity
            try:
                from core.monitoring.kpi import safe_log_kpi, format_kpi
                safe_log_kpi(format_kpi({
                    "ts": time.time(),
                    "event": "sam_call_start",
                    "bbox_device": str(bbox_t.device if hasattr(bbox_t, 'device') else "cpu"),
                    "image_device": str(getattr(full_image, "device", "cpu")),
                    "sam_as_numpy": int(sam_as_numpy)
                }))
            except Exception:
                pass
            
            # Vérification GPU avant appel SAM
            if not isinstance(bbox_t, torch.Tensor) or not bbox_t.is_cuda:
                LOG.warning("Converting bbox to GPU tensor for SAM")
                bbox_t = torch.as_tensor(bbox_t, device="cuda", dtype=torch.float32)
                
            LOG.debug(f"SAM receives bbox tensor on {bbox_t.device}, dtype={bbox_t.dtype}")
            LOG.debug(f"[MASK DEBUG] Calling run_segmentation with bbox={bbox_t}")
            
            # ⏱️ Mesure de la durée d’inférence MobileSAM
            t_sam0 = time.perf_counter()
            mask = run_segmentation(sam_model, full_image, bbox_xyxy=bbox_t, as_numpy=sam_as_numpy)
            t_sam1 = time.perf_counter()

            # 🧩 Enregistre la durée MobileSAM dans le monitor
            try:
                from core.monitoring import monitor
                monitor.record_interstage("proc_to_gpu_cpu", (t_sam1 - t_sam0) * 1000.0)
            except Exception:
                LOG.debug("Failed to record SAM latency")

            # KPI après SAM - monitoring résultat  
            try:
                safe_log_kpi(format_kpi({
                    "ts": time.time(),
                    "event": "sam_call_end",
                    "mask_device": str(getattr(mask, "device", "cpu")),
                    "mask_type": str(type(mask))
                }))
            except Exception:
                pass

                
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
            # KPI device monitoring pour diagnostic
            try:
                LOG_KPI.info(f"Mask converted to CPU for compute_mask_weights (device={getattr(mask, 'device', 'cpu')})")
            except Exception:
                pass
                
            # ⚠️ TODO Phase 2 : porter compute_mask_weights() sur GPU (torch ops)
            # Cela supprimera ce dernier transfert CPU.
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
    """
    Exécute l’inférence GPU complète (D-FINE → MobileSAM → postprocess),
    mesure les latences inter-étapes et renvoie un ResultPacket stable.

    Étapes monitorées :
      1️⃣ RX → CPU→GPU        : transfert du frame vers GPU
      2️⃣ CPU→GPU → PROC      : exécution DFINE + SAM
      3️⃣ PROC → GPU→CPU      : récupération du mask depuis GPU
      4️⃣ GPU→CPU → TX        : préparation + envoi du résultat

    Retour :
        (result_packet: dict compatible ResultPacket, latency_ms: float)
    """
    import torch
    t0_total = time.perf_counter()  # horodatage global

    try:
        # ===============================================================
        # 🔹 Extraction du frame et des métadonnées
        # ===============================================================
        arr = getattr(frame_tensor, "tensor", frame_tensor)
        meta = getattr(frame_tensor, "meta", None)
        frame_id = getattr(meta, "frame_id", None)
        ts = getattr(meta, "ts", None)

        # ===============================================================
        # 🔹 1️⃣ RX → CPU→GPU : transfert CPU→GPU
        # ===============================================================
        t0_rx_gpu = time.perf_counter()
        try:
            if isinstance(arr, np.ndarray):
                # passage CPU→GPU explicite si pas déjà Tensor CUDA
                arr = torch.as_tensor(arr, dtype=torch.float32, device="cuda", non_blocking=True)
            elif isinstance(arr, torch.Tensor) and not arr.is_cuda:
                arr = arr.to(device="cuda", non_blocking=True)
        except Exception as e:
            LOG.warning("Transfert CPU→GPU échoué : %s", e)
        t1_rx_gpu = time.perf_counter()
        monitor.record_interstage("rx_to_cpu_gpu", (t1_rx_gpu - t0_rx_gpu) * 1000.0)

        # ===============================================================
        # 🔹 Récupération des modèles depuis le cache global
        # ===============================================================
        dfine_model = None
        sam_model = None
        try:
            from core.inference.engine import model_loader as _ml
            cache = getattr(_ml, "_MODEL_CACHE", None)
            if cache:
                first = next(iter(cache.values()), None)
                if isinstance(first, dict):
                    dfine_model = first.get("dfine")
                    sam_model = first.get("mobilesam") or first.get("sam")
        except Exception:
            LOG.debug("Model cache introuvable ou vide")

        # ===============================================================
        # 🔹 2️⃣ CPU→GPU → PROC : exécution D-FINE + SAM
        # ===============================================================
        t0_proc = time.perf_counter()
        prepared = prepare_inference_inputs(arr, dfine_model, sam_model)
        t1_proc = time.perf_counter()
        monitor.record_interstage("cpu_gpu_to_proc", (t1_proc - t0_proc) * 1000.0)

        # ===============================================================
        # 🔹 3️⃣ PROC → GPU→CPU : récupération du mask
        # ===============================================================
        t0_mask = time.perf_counter()
        mask = prepared.get("mask", None)
        # Forcer conversion si le mask est encore sur GPU (cas PyTorch)
        try:
            if hasattr(mask, "is_cuda") and mask.is_cuda:
                mask = mask.detach().cpu().numpy()
        except Exception:
            pass
        t1_mask = time.perf_counter()
        monitor.record_interstage("proc_to_gpu_cpu", (t1_mask - t0_mask) * 1000.0)

        # ===============================================================
        # 🔹 4️⃣ GPU→CPU → TX : préparation du ResultPacket
        # ===============================================================
        t0_tx = time.perf_counter()

        bbox = prepared.get("bbox", None)
        state = prepared.get("state_hint", "LOST")
        score = float(prepared.get("conf", 0.0) or 0.0)
        weights = prepared.get("weights", None)

        latency_ms = (t1_mask - t0_total) * 1000.0

        result = {
            "frame_id": frame_id,
            "timestamp": ts if ts is not None else time.time(),
            "bbox": bbox,
            "mask": mask,
            "score": score,
            "state": state,
            "weights": weights,
            "latency_ms": round(latency_ms, 3),
        }

        t1_tx = time.perf_counter()
        monitor.record_interstage("gpu_cpu_to_tx", (t1_tx - t0_tx) * 1000.0)

        # ===============================================================
        # 🔹 Logging & KPI
        # ===============================================================
        try:
            LOG.debug(
                "Inference completed: frame=%s state=%s conf=%.3f latency=%.1f ms",
                frame_id, state, score, latency_ms
            )
        except Exception:
            LOG.debug("Inference completed (frame=%s)", frame_id)

        # Émission KPI structurée (pour kpi.log + dashboard)
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
        except Exception as e:
            LOG.debug("Failed to emit KPI infer_frame: %s", e)

        return result, latency_ms

    # ===============================================================
    # 🔹 Gestion d'erreurs robuste
    # ===============================================================
    except Exception as e:
        t1_fail = time.perf_counter()
        latency_ms = (t1_fail - t0_total) * 1000.0
        LOG.exception("run_inference failed: %s", e)

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
