# ⚠️ LEGACY MODULE — to be deprecated after Gateway v3 integration
# This module is kept only for offline pipelines and local GPU tests.
# All live RX/TX buffering is now handled by core.queues.adaptive.
# TODO [v3]: Migrate remaining uses in detection_and_engine and dfine_infer.
# TODO [v3]: Move this file to core/legacy/queues/buffers_legacy.py once migration complete.

"""
core/queues/buffers.py (legacy)
================================

📆 Mise à jour : 2025-10-21
📦 Statut : LEGACY — conservé pour tests locaux et pipelines hors-gateway

Ce module gère désormais uniquement deux buffers pour les pipelines
locaux/legacy : `Queue_GPU` et `Queue_Out`. Les opérations temps réel
et adaptatives (mailbox, drops, resizing) sont effectuées par
`core.queues.adaptive` via la passerelle `service.gateway`.

⚙️ Drop-oldest policy now limited to Queue_GPU and Queue_Out.

─────────────────────────────────────────────────────────────
🧭 Migration plan (v3)
─────────────────────────────────────────────────────────────
• Déplacer ce module sous `core/legacy/queues/`.
• Supprimer dès que `detection_and_engine` et `dfine_infer` utilisent
    `AdaptiveDeque` / `GpuPool` fournis par la passerelle.
• `core.queues.gpu_buffers` (phase 4) prendra le relais des buffers GPU.

Le module reste fonctionnel et stable pour les tests locaux. Aucune
modification fonctionnelle n'est prévue ici : il sert uniquement de
pont jusqu'à la migration complète.
"""

from typing import Dict, Any, Optional, Tuple, Iterable, Union, TypedDict
import queue
import time
import logging

from core.types import GpuFrame, ResultPacket

LOG = logging.getLogger("igt.queues")
LOG_KPI = logging.getLogger("igt.kpi")


# Typed aliases for queues (runtime uses queue.Queue)
# QGPU: runtime queue for GpuFrame (queue.Queue)
# QOut: runtime queue for ResultPacket (queue.Queue)
QGPU = queue.Queue
QOut = queue.Queue


# Internal typed registry for the four queues
# Use a TypedDict so static checkers can validate access by exact names
class RegistryType(TypedDict):
    Queue_GPU: QGPU
    Queue_Out: QOut

# At runtime this dict will be populated by init_queues
_QUEUES: RegistryType = {}


# Metrics / counters (module-level) for observability
# increments when an item is dropped by policy
drops_gpu: int = 0
drops_out: int = 0

# last backpressure timestamp per queue
last_bp_gpu_ts: Optional[float] = None
last_bp_out_ts: Optional[float] = None


# NOTE: list-based archive helper removed — live system uses AdaptiveDeque


def _drop_oldest_policy_queue(q: queue.Queue, now: Optional[float] = None, max_lag_ms: int = 500) -> Dict[str, int]:
    """Drop-oldest policy for queue.Queue (legacy helper).

    This helper is intended only for the remaining local buffers
    (`Queue_GPU` and `Queue_Out`) in legacy/local pipeline modes. It
    temporarily empties the queue, removes items older than `max_lag_ms`,
    then reinserts the recent items preserving order.

    Returns a dict: {'removed': n, 'remaining': m}.
    """
    if now is None:
        now = time.time()

    temp = []
    removed = 0
    try:
        while True:
            item = q.get_nowait()
            temp.append(item)
    except queue.Empty:
        pass

    # remove oldest exceeding threshold
    while temp and (now - temp[0].meta.ts) * 1000.0 > max_lag_ms:
        temp.pop(0)
        removed += 1

    # put remaining back
    for item in temp:
        try:
            q.put_nowait(item)
        except queue.Full:
            removed += 1

    return {"removed": removed, "remaining": len(temp)}


def enqueue_nowait_gpu(q_gpu: QGPU, item: GpuFrame) -> bool:
    """Non-blocking enqueue into GPU queue. If full, do not block.

    Returns False if queue full.
    """
    global drops_gpu, last_bp_gpu_ts
    try:
        q_gpu.put_nowait(item)
        return True
    except queue.Full:
        # attempt drop-oldest once to make room
        stats = _drop_oldest_policy_queue(q_gpu, now=time.time())
        removed = int(stats.get("removed", 0))
        if removed > 0:
            drops_gpu += removed
            last_bp_gpu_ts = time.time()
            try:
                from core.monitoring.kpi import increment_drops

                try:
                    increment_drops("gpu.drop_total", removed, emit=True)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                from core.monitoring.kpi import safe_log_kpi, format_kpi

                kmsg = format_kpi({"ts": time.time(), "event": "drop_event", "removed": removed, "qsize": q_gpu.qsize()})
                safe_log_kpi(kmsg)
            except Exception:
                LOG.debug("Failed to emit KPI drop_event for GPU")
        try:
            q_gpu.put_nowait(item)
            return True
        except queue.Full:
            return False


def enqueue_nowait_out(q_out: QOut, item: ResultPacket) -> bool:
    """Non-blocking enqueue into Out queue. If full, attempt one drop-oldest, else fail."""
    global drops_out, last_bp_out_ts
    try:
        q_out.put_nowait(item)
        return True
    except queue.Full:
        stats = _drop_oldest_policy_queue(q_out, now=time.time())
        removed = int(stats.get("removed", 0))
        if removed > 0:
            drops_out += removed
            last_bp_out_ts = time.time()
            try:
                from core.monitoring.kpi import increment_drops

                try:
                    increment_drops("out.drop_total", removed, emit=True)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                from core.monitoring.kpi import safe_log_kpi, format_kpi

                kmsg = format_kpi({"ts": time.time(), "event": "drop_event", "removed": removed, "qsize": q_out.qsize()})
                safe_log_kpi(kmsg)
            except Exception:
                LOG.debug("Failed to emit KPI drop_event for OUT")
        try:
            q_out.put_nowait(item)
            return True
        except queue.Full:
            return False


def try_dequeue(q: Any):
    """Try to dequeue from a queue.Queue.

    Returns the item or None if empty.
    """
    if isinstance(q, list):
        if q:
            return q.pop(0)
        return None
    try:
        return q.get_nowait()
    except queue.Empty:
        return None


# `get_queue_raw` removed: Queue_Raw is now managed by the Gateway AdaptiveDeque




def get_queue_gpu() -> QGPU:
    return _QUEUES["Queue_GPU"]


def get_queue_out() -> QOut:
    return _QUEUES["Queue_Out"]


def init_queues(config: Dict) -> RegistryType:
    """Initialise les 4 queues et les stocke dans `_QUEUES`.

        Policy (simplifiée) :
            - Queue_GPU  : queue.Queue(maxsize=3)
            - Queue_Out  : queue.Queue(maxsize=3)

    Returns:
        mapping name->queue-like (RegistryType)
    """
    # Queues thread-safe
    _QUEUES["Queue_GPU"] = queue.Queue(maxsize=3)
    _QUEUES["Queue_Out"] = queue.Queue(maxsize=3)

    return _QUEUES


def get_queue(name: str) -> Any:
    """Retourne la queue identifiée par `name`.

    Lève KeyError si la queue n'existe pas.
    """
    if name not in _QUEUES:
        raise KeyError(f"Queue '{name}' non initialisée. Appeler init_queues(config) d'abord.")
    return _QUEUES[name]


def collect_queue_metrics() -> Dict[str, Any]:
    """Collect simple metrics for all queues.

    Returns a mapping per queue name with: size, maxsize (or None), drops, last_backpressure_ts.
    """
    metrics = {}
    for name in ("Queue_GPU", "Queue_Out"):
        q = _QUEUES.get(name)
        if q is None:
            metrics[name] = {"size": None, "maxsize": None, "drops": None, "last_backpressure_ts": None}
            continue
        if isinstance(q, list):
            size = len(q)
            maxsize = None
        else:
            size = q.qsize()
            try:
                maxsize = q.maxsize
                if maxsize <= 0:
                    maxsize = None
            except Exception:
                maxsize = None

        if name == "Queue_GPU":
            drops = drops_gpu
            last = last_bp_gpu_ts
        elif name == "Queue_Out":
            drops = drops_out
            last = last_bp_out_ts
        else:
            drops = None
            last = None

        metrics[name] = {"size": size, "maxsize": maxsize, "drops": drops, "last_backpressure_ts": last}

    return metrics


# ℹ️ Adaptive resizing is now handled by core.queues.adaptive.adjust_queue_size

# ✅ buffers.py legacy version stabilized (v2)
