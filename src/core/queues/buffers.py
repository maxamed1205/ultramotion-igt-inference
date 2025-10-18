"""Queues et buffers centralisés

Description
-----------
Ce module centralise la création et l'accès aux différentes queues utilisées
par la pipeline :
- Queue_Raw   : stockage/archive des frames brutes
- Queue_RT_dyn: queue temps réel adaptative (priorité faible latence)
- Queue_GPU   : queue de tensors prêts pour GPU
- Queue_Out   : queue des résultats à envoyer (mask/score)

Responsabilités
----------------
- initialiser les queues selon la configuration,
- fournir des helpers d'accès (get_queue),
- permettre l'ajustement adaptatif (resize) par le monitor.

Dépendances attendues
---------------------
- queue.Queue ou multiprocessing.Queue selon besoin (process vs thread)

Schéma des queues (types & politique)
------------------------------------
Chaque queue transporte un type unique (contrat dataclasses dans `core.types`):

Queue_Raw    -> RawFrame
    Rôle: archive complète; taille conseillée: non bornée / disque
    Politique: aucune suppression automatique (écriture séquentielle)

Queue_RT_dyn -> RawFrame
    Rôle: temps réel vers GPU; maxsize=3
    Politique: drop-oldest si lag_ms > 500

Queue_GPU    -> GpuFrame
    Rôle: tampon VRAM entre B et C; maxsize=2-3
    Politique: drop-oldest (ne jamais bloquer B)

Queue_Out    -> ResultPacket
    Rôle: tampon pour l'envoi IGT; maxsize=2-3
    Politique: drop-oldest (ne jamais bloquer C)

Définition du lag (utilisée par la politique):
    lag_ms = (now() - raw.meta.ts) * 1000
Si lag_ms > 500 en entrant dans Queue_RT_dyn -> éjecter les plus anciens
jusqu'à retomber < 500 ms.

Fonctions principales
---------------------
 - init_queues(config) -> dict
 - get_queue(name) -> queue object
 - adaptive_queue_resize(queues, policy)

Note
----
Seules les signatures sont fournies ici.
"""

from typing import Dict, Any, Optional, Tuple, Iterable, Union, TypedDict
import queue
import time
import logging

from core.types import RawFrame, GpuFrame, ResultPacket

LOG = logging.getLogger("igt.queues")
LOG_KPI = logging.getLogger("igt.kpi")


# Typed aliases for queues (runtime uses queue.Queue or list for archive)
# QRaw: archive of RawFrame (list)
# QRTDyn: runtime queue for RawFrame (queue.Queue)
# QGPU: runtime queue for GpuFrame (queue.Queue)
# QOut: runtime queue for ResultPacket (queue.Queue)
QRaw = list[RawFrame]
QRTDyn = queue.Queue
QGPU = queue.Queue
QOut = queue.Queue


# Internal typed registry for the four queues
# Use a TypedDict so static checkers can validate access by exact names
class RegistryType(TypedDict):
    Queue_Raw: QRaw
    Queue_RT_dyn: QRTDyn
    Queue_GPU: QGPU
    Queue_Out: QOut

# At runtime this dict will be populated by init_queues
_QUEUES: RegistryType = {}


# Metrics / counters (module-level) for observability
# increments when an item is dropped by policy
drops_rt: int = 0
drops_gpu: int = 0
drops_out: int = 0

# last backpressure timestamp per queue
last_bp_rt_ts: Optional[float] = None
last_bp_gpu_ts: Optional[float] = None
last_bp_out_ts: Optional[float] = None


def drop_oldest_policy_list(q: list, now: Optional[float] = None, max_lag_ms: int = 500) -> None:
    """Version liste de la politique drop-oldest (usage tests / archive).

    Args:
        q: liste de RawFrame où q[0] est le plus ancien.
        now: timestamp de référence (time.time()). Si None, utilise time.time().
        max_lag_ms: seuil en millisecondes au-delà duquel on éjecte.
    """
    if now is None:
        now = time.time()

    while q and (now - q[0].meta.ts) * 1000.0 > max_lag_ms:
        q.pop(0)


def drop_oldest_policy_queue(q: queue.Queue, now: Optional[float] = None, max_lag_ms: int = 500) -> Dict[str, int]:
    """Version thread-safe de la politique drop-oldest pour `queue.Queue`.

    Cette opération vide temporairement la queue, filtre les éléments trop
    vieux, puis remet les éléments récents dans la queue dans le même ordre.

    Retourne:
        dict contenant 'removed' (nb supprimés) et 'remaining' (nb remis).

    Note:
        - Utilise get_nowait/put_nowait et est atomique vis-à-vis des autres
          actions concurrentes seulement dans la mesure où le consommateur
          est coordonné par le monitor/producer.
        - Pour des garanties strictes en prod, envisager l'utilisation du
          lock interne (q.mutex) ou d'une stratégie lockée par le monitor.
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

    # retire les plus anciens dépassant max_lag_ms
    while temp and (now - temp[0].meta.ts) * 1000.0 > max_lag_ms:
        old = temp.pop(0)
        removed += 1
        if LOG.isEnabledFor(logging.DEBUG):
            try:
                LOG.debug("drop_oldest_policy_queue removing frame %s (ts=%s)", getattr(old, "meta", None) and getattr(old.meta, "frame_id", None), getattr(old, "meta", None) and getattr(old.meta, "ts", None))
            except Exception:
                # avoid any exception in logging path
                pass

    # remets le reste dans la queue
    for item in temp:
        try:
            q.put_nowait(item)
        except queue.Full:
            # en cas d'overflow improbable, on incrémente removed
            removed += 1

    return {"removed": removed, "remaining": len(temp)}


def apply_rt_backpressure(q_rt: QRTDyn, now: Optional[float] = None, max_lag_ms: int = 500) -> Dict[str, int]:
    """Apply drop-oldest backpressure on a runtime RT queue.

    Returns stats dict with keys: removed, remaining.
    Updates module-level drops_rt and last_bp_rt_ts.
    """
    global drops_rt, last_bp_rt_ts
    if now is None:
        now = time.time()

    stats = drop_oldest_policy_queue(q_rt, now=now, max_lag_ms=max_lag_ms)
    removed = int(stats.get("removed", 0))
    if removed > 0:
        drops_rt += removed
        last_bp_rt_ts = now
        try:
            from core.monitoring.kpi import increment_drops

            try:
                increment_drops("rt.drop_total", removed, emit=True)
            except Exception:
                pass
        except Exception:
            pass
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi

            kmsg = format_kpi({"ts": now, "event": "drop_event", "removed": removed, "qsize": q_rt.qsize()})
            safe_log_kpi(kmsg)
        except Exception:
            # KPI logging must not fail the path
            LOG.debug("Failed to emit KPI drop_event")
    return stats


def enqueue_nowait_rt(q_rt: QRTDyn, item: RawFrame) -> bool:
    """Non-blocking enqueue into RT queue.

    Returns True if enqueued, False if queue was full and item dropped.
    The caller may choose to call apply_rt_backpressure before retrying.
    """
    try:
        q_rt.put_nowait(item)
        return True
    except queue.Full:
        return False


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
        stats = drop_oldest_policy_queue(q_gpu, now=time.time())
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
        stats = drop_oldest_policy_queue(q_out, now=time.time())
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
    """Try to dequeue from a queue.Queue or pop from list archive.

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


def get_queue_raw() -> QRaw:
    return _QUEUES["Queue_Raw"]


def get_queue_rt_dyn() -> QRTDyn:
    return _QUEUES["Queue_RT_dyn"]


def get_queue_gpu() -> QGPU:
    return _QUEUES["Queue_GPU"]


def get_queue_out() -> QOut:
    return _QUEUES["Queue_Out"]


def init_queues(config: Dict) -> RegistryType:
    """Initialise les 4 queues et les stocke dans `_QUEUES`.

    Policy (simplifiée) :
      - Queue_Raw  : queue sans taille max (utilisation de list pour archive)
      - Queue_RT_dyn: queue.Queue(maxsize=3)
      - Queue_GPU  : queue.Queue(maxsize=3)
      - Queue_Out  : queue.Queue(maxsize=3)

    Returns:
        mapping name->queue-like (RegistryType)
    """
    # Archive raw en list (écriture séquentielle, pas de block)
    _QUEUES["Queue_Raw"] = []

    # Queues thread-safe pour le reste
    _QUEUES["Queue_RT_dyn"] = queue.Queue(maxsize=3)
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
    for name in ("Queue_Raw", "Queue_RT_dyn", "Queue_GPU", "Queue_Out"):
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

        if name == "Queue_Raw":
            drops = None
            last = None
        elif name == "Queue_RT_dyn":
            drops = drops_rt
            last = last_bp_rt_ts
        elif name == "Queue_GPU":
            drops = drops_gpu
            last = last_bp_gpu_ts
        else:
            drops = drops_out
            last = last_bp_out_ts

        metrics[name] = {"size": size, "maxsize": maxsize, "drops": drops, "last_backpressure_ts": last}

    return metrics


def adaptive_queue_resize(queues: Dict[str, Any], policy: Dict) -> None:
    """Placeholder pour futur redimensionnement adaptatif.

    Pour l'instant, la logique est déléguée à `drop_oldest_policy` et
    au monitor qui décide quand appeler cette fonction.
    """
    # Minimal: no-op for now
    return None
