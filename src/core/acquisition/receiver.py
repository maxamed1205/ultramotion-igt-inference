# TODO [Phase 5 - Logging Integration]
# - Logger KPI ajouté pour métriques acquisition (fps, latency, drops)
# - Publier dans logs/kpi.log via igt.kpi
# - Conserver logs contextuels (frame_id, ts, queue_size)
# - Adapter niveaux: INFO/WARNING/ERROR selon gravité
# - Compatible PerfFilter (désactivation propre en mode perf)

"""Receiver module — Thread A

Rôle : recevoir les images et poses via IGTLink, les pousser dans
Queue_Raw + Queue_RT_dyn.

Contient les signatures des fonctions sans implémentation.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import time
import logging
import threading

from typing import Tuple as _Tuple

from core.types import FrameMeta, RawFrame
from core.queues.buffers import get_queue_rt_dyn, enqueue_nowait_rt, apply_rt_backpressure
from core.monitoring.monitor import log_kpi_tick

LOG = logging.getLogger("igt.receiver")
KPI = logging.getLogger("igt.kpi")


class ReceiverThread(threading.Thread):
    """Thread A — Acquisition en continu (PlusServer → Queue_RT_dyn).

    Récupère les images via IGTGateway.receive_image() et les pousse dans
    `Queue_RT_dyn` via `enqueue_to_rt()`.

    Note: boucle utilise un bref sleep pour rester CPU-friendly lorsque
    la source n'est pas prête. Quand `pyigtl` sera utilisé, remplacer par
    un mécanisme bloquant sur socket (select/epoll) pour éviter polling.
    """

    def __init__(self, gateway: object, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.gateway = gateway
        self.stop_event = stop_event

    def run(self) -> None:
        LOG.info("ReceiverThread started.")

        # KPI aggregation is handled centrally by core.monitoring.monitor

        while not self.stop_event.is_set():
            try:
                frame = None
                # recv_start measures network/IGTLink receive latency
                recv_start = time.time()
                try:
                    frame = self.gateway.receive_image()
                except Exception as e:
                    # transient gateway errors should not break the loop
                    LOG.error("IGTGateway receive_image error: %r", e)
                    frame = None

                # processing timestamp after receiving and constructing frame
                t0 = time.time()

                if frame:
                    try:
                        # Explicit enqueue policy: try non-blocking, apply backpressure once, retry
                        # measure enqueue latency
                        enqueue_latency_ms = 0.0
                        enqueue_start = time.time()
                        q = get_queue_rt_dyn()
                        if not enqueue_nowait_rt(q, frame):
                            # apply backpressure (drop oldest frames) then retry once
                            apply_rt_backpressure(q, now=time.time(), max_lag_ms=500)
                            enqueue_nowait_rt(q, frame)
                        enqueue_latency_ms = (time.time() - enqueue_start) * 1000.0

                        # contextual debug log with frame id and queue size
                        try:
                            qsize = q.qsize() if not isinstance(q, list) else len(q)
                        except Exception:
                            qsize = -1

                        if LOG.isEnabledFor(logging.DEBUG):
                            LOG.debug(
                                "Frame pushed (id=%s ts=%.3f queue=%d)",
                                getattr(frame.meta, "frame_id", -1),
                                getattr(frame.meta, "ts", -1.0),
                                qsize,
                            )

                        # KPI emission delegated to central monitor. Emit a per-frame
                        # datapoint (latency) and let the monitor perform aggregation.
                        proc_latency_ms = (t0 - recv_start) * 1000.0
                        total_latency_ms = (time.time() - recv_start) * 1000.0
                        # combine processing and enqueue latency for KPI
                        total_proc_ms = proc_latency_ms + enqueue_latency_ms
                        try:
                            # fps values are computed centrally; pass 0.0 placeholders
                            # for now and provide the measured per-frame processing latency.
                            log_kpi_tick(0.0, 0.0, total_proc_ms, gpu_util=0.0)
                        except Exception:
                            # Do not raise on KPI emission failure; keep contextual log
                            if LOG.isEnabledFor(logging.DEBUG):
                                LOG.debug("log_kpi_tick failed for frame_id=%s", getattr(frame.meta, "frame_id", -1))
                    except Exception as e:
                        # Enqueue error: receiver should not implement drop logic here.
                        # Drops and counters are handled centrally by core.queues.buffers
                        # (apply_rt_backpressure). Keep a debug log and continue.
                        LOG.debug("ReceiverThread enqueue error: %r", e)
            except Exception as e:
                # protect the loop from unexpected exceptions
                LOG.debug("ReceiverThread error: %r", e)
            # small sleep to avoid busy-wait when no data
            # TODO [Phase 7] Remplacer time.sleep(0.001) par select.select() quand IGTGateway utilisera pyigtl.Client
            time.sleep(0.001)
        LOG.info("ReceiverThread stopped.")


def start_receiver_thread(config: Dict) -> None:
    """Démarre le thread d'acquisition.

    Args:
        config: dictionnaire de configuration (ports, host, etc.).
    """
    # create stop event and gateway lazily to avoid import-time issues
    stop_event = threading.Event()

    plus_host = config.get("plus_host", "127.0.0.1")
    plus_port = int(config.get("plus_port", 18944))
    slicer_port = int(config.get("slicer_port", 18945))

    # instantiate gateway (import locally)
    try:
        from service.igthelper import IGTGateway
    except Exception:
        try:
            from igthelper import IGTGateway  # fallback
        except Exception:
            IGTGateway = None

    if IGTGateway is not None:
        gateway = IGTGateway(plus_host, plus_port, slicer_port)
        try:
            gateway.start()
        except Exception:
            LOG.debug("IGTGateway.start() failed or is a stub")
    else:
        # use centralized stub gateway for simulation/testing
        from simulation.mock_gateway import StubGateway

        gateway = StubGateway()

    receiver = ReceiverThread(gateway, stop_event)
    receiver.start()
    LOG.info("Receiver thread started (PlusServer=%s:%d)", plus_host, plus_port)
    return receiver, stop_event


# _igt_callback moved to core.acquisition.decode.decode_igt_image


def stop_receiver_thread() -> None:
    """Arrête proprement le thread d'acquisition."""
    raise NotImplementedError(
        "Use stop_receiver_thread(receiver, stop_event) with the objects returned by start_receiver_thread"
    )


def stop_receiver_thread_ex(receiver: threading.Thread, stop_event: threading.Event, timeout: float = 0.1) -> None:
    """Stop the running ReceiverThread instance.

    Args:
        receiver: thread object returned by start_receiver_thread
        stop_event: threading.Event instance returned by start_receiver_thread
        timeout: join timeout in seconds
    """
    try:
        stop_event.set()
        receiver.join(timeout=timeout)
        LOG.info("Receiver thread stopped cleanly.")
    except Exception as e:
        LOG.debug("Error stopping receiver thread: %r", e)



