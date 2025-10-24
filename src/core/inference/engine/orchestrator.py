from typing import Any, Dict, Optional, Tuple
import logging
import time
import numpy as np

from core.types import GpuFrame, ResultPacket
from core.queues.buffers import get_queue_gpu, get_queue_out, try_dequeue, enqueue_nowait_out

LOG = logging.getLogger("igt.inference")
LOG_KPI = logging.getLogger("igt.kpi")


def prepare_inference_inputs(frame_t: np.ndarray, dfine_model: Any, sam_model: Any, tau_conf: float = 0.5) -> Dict[str, Any]:
    """Orchestration complète des étapes 0 → 3.

    0. Passe l’image dans D-FINE → bbox/conf.
    1. Si conf < τ_conf → renvoie state_hint='LOST'.
    2. Crop ROI autour de la bbox.
    3. Passe la ROI dans MobileSAM.
    4. Calcule les pondérations spatiales (W_edge/W_in/W_out).

    Returns:
        dictionnaire prêt pour visibility_fsm.evaluate_visibility().
    """
    raise NotImplementedError


def run_inference(frame_tensor: GpuFrame, stream_infer: Any = None) -> Tuple[ResultPacket, float]:
    """Exécute (mock) l’inférence GPU et retourne un ResultPacket minimal."""
    raise NotImplementedError


def fuse_outputs(mask: Any, score: float, state: str) -> ResultPacket:
    """Fusionne les sorties et renvoie un ResultPacket standardisé."""
    raise NotImplementedError


def process_inference_once(models: Any = None) -> None:
    """Consomme une GpuFrame, exécute une inférence (mock) et place le résultat en sortie."""
    q_gpu = get_queue_gpu()
    gf = try_dequeue(q_gpu)
    if gf is None:
        return

    if LOG.isEnabledFor(logging.DEBUG):
        fid = getattr(getattr(gf, "meta", None), "frame_id", None)
        LOG.debug("Dequeued GpuFrame for inference: %s", fid)

    # Simulation minimale d’inférence
    t0 = time.time()
    result: ResultPacket = {
        "frame_id": getattr(getattr(gf, "meta", None), "frame_id", None),
        "mask": None,
        "score": 1.0,
        "state": "OK",
        "timestamp": getattr(getattr(gf, "meta", None), "ts", None),
    }  # type: ignore[assignment]
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0

    try:
        from core.monitoring.kpi import safe_log_kpi, format_kpi
        msg = format_kpi({"ts": t1, "event": "infer_event", "frame": result.get("frame_id"), "latency_ms": f"{latency_ms:.1f}"})
        safe_log_kpi(msg)
    except Exception:
        LOG.debug("Failed to emit KPI infer_event")

    try:
        q_out = get_queue_out()
        ok = enqueue_nowait_out(q_out, result)  # type: ignore[arg-type]
        if not ok:
            LOG.warning("Out queue full, result for frame %s dropped", result.get("frame_id"))
    except Exception as e:
        LOG.exception("Failed to enqueue result: %s", e)
