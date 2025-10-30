"""
aggregator.py
--------------
Fusionne les événements de même frame_id et calcule les latences inter-étapes.
Version alignée avec les structures de core/types.py et schemas.py.
"""

from collections import deque
from threading import RLock
from . import logger
from .schemas import (
    FrameMetaLite,
    GPUTransferLite,
    InterStageLatencyLite,
    FrameAggregate,
    DashboardSnapshotLite,
    TimelineLite,
)


class FrameAggregator:
    """Fusionne les événements issus de pipeline.log et kpi.log
    et maintient un historique glissant des frames agrégées."""

    def __init__(self, max_history: int = 300):
        self.frames: dict[int, FrameAggregate] = {}
        self.history: deque[FrameAggregate] = deque(maxlen=max_history)
        self.lock = RLock()

    # ------------------------------------------------------------------ #
    def update(self, parsed: dict):
        """Met à jour les données agrégées à partir d'une ligne parsée."""
        if not parsed or "frame_id" not in parsed:
            return

        fid = int(parsed["frame_id"])
        event = parsed.get("event")
        with self.lock:
            frame = self.frames.get(fid)

            # ─── Création initiale si nouvelle frame ───
            if frame is None:
                meta = FrameMetaLite(frame_id=fid, ts=parsed.get("ts", 0.0))
                frame = FrameAggregate(frame_id=fid, meta=meta)
                # initialise les timestamps RX/PROC/TX à None
                frame.rx = None
                frame.proc = None
                frame.tx = None
                self.frames[fid] = frame

            # 1️⃣ RX / PROC / TX
            if event in ("rx", "proc", "tx"):
                frame_ts = parsed.get("ts")
                if frame_ts:
                    setattr(frame, event, frame_ts)
                if getattr(frame, "rx", None) and getattr(frame, "proc", None) and getattr(frame, "tx", None):
                    try:
                        frame.latency_rxproc = (frame.proc - frame.rx) * 1000.0
                        frame.latency_proctx = (frame.tx - frame.proc) * 1000.0
                        frame.latency_rxtx = (frame.tx - frame.rx) * 1000.0
                    except Exception as e:
                        logger.debug(f"[AGG] Erreur calcul latences frame#{fid}: {e}")

            # 2️⃣ GPU copy_async
            elif event == "copy_async" and "latencies" in parsed:
                l = parsed["latencies"]
                frame.gpu_transfer = GPUTransferLite(
                    frame_id=fid,
                    norm_ms=l.get("norm_ms", 0.0),
                    pin_ms=l.get("pin_ms", 0.0),
                    copy_ms=l.get("copy_ms", 0.0),
                    total_ms=l.get("cpu_gpu", l.get("total_ms", 0.0)),
                )

            # 3️⃣ Inter-stage latencies
            elif event == "interstage" and "latencies" in parsed:
                lat = parsed["latencies"]
                frame.interstage = InterStageLatencyLite(
                    frame_id=fid,
                    rx_cpu=lat.get("rx_cpu"),
                    cpu_gpu=lat.get("cpu_gpu"),
                    proc_gpu=lat.get("proc_gpu"),
                    gpu_cpu=lat.get("gpu_cpu"),
                    cpu_tx=lat.get("cpu_tx"),
                    total=lat.get("total"),
                )

            # 4️⃣ Frame complète
            if self._is_frame_complete(frame):
                self._finalize_frame(fid, frame)

    # ------------------------------------------------------------------ #
    def _is_frame_complete(self, frame: FrameAggregate) -> bool:
        """Frame complète quand RX/PROC/TX + copy_async + interstage dispo."""
        has_cpu_path = all([getattr(frame, "rx", None), getattr(frame, "proc", None), getattr(frame, "tx", None)])
        has_gpu_transfer = frame.gpu_transfer is not None
        has_interstage = frame.interstage is not None and frame.interstage.total is not None
        return has_cpu_path and has_gpu_transfer and has_interstage

    # ------------------------------------------------------------------ #
    def _finalize_frame(self, fid: int, frame: FrameAggregate):
        frame.ts_wall = frame.meta.ts
        total = frame.interstage.total if frame.interstage else None
        logger.debug(f"[AGG] Frame #{fid} complète : lat_tot={total}")
        self.history.append(frame)
        self.frames.pop(fid, None)

    def get_latest(self):
        with self.lock:
            return self.history[-1] if self.history else None

    def get_history(self, n: int = 100):
        with self.lock:
            return list(self.history)[-n:]

    # ------------------------------------------------------------------ #
    def as_snapshot(self, profiler_stats: dict | None = None) -> DashboardSnapshotLite:
        with self.lock:
            latest = self.history[-1] if self.history else None
            timeline = self._build_timeline()
            return DashboardSnapshotLite(
                latest=latest,
                timeline=timeline,
                collector_ms_avg=(profiler_stats or {}).get("avg_ms"),
                collector_ms_p95=(profiler_stats or {}).get("p95_ms"),
                health="OK" if latest else "EMPTY",
            )

    def _build_timeline(self) -> TimelineLite:
        timeline = TimelineLite()
        for frame in list(self.history):
            fid = frame.frame_id
            if frame.interstage:
                i = frame.interstage
                timeline.frames.append(fid)
                timeline.total_ms.append(i.total or 0.0)
                timeline.rx_cpu.append(i.rx_cpu or 0.0)
                timeline.cpu_gpu.append(i.cpu_gpu or 0.0)
                timeline.proc_gpu.append(i.proc_gpu or 0.0)
                timeline.gpu_cpu.append(i.gpu_cpu or 0.0)
                timeline.cpu_tx.append(i.cpu_tx or 0.0)
            elif frame.latency_rxtx is not None:
                timeline.frames.append(fid)
                timeline.total_ms.append(frame.latency_rxtx or 0.0)
        return timeline
