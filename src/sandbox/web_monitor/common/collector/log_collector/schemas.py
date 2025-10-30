"""
schemas.py — LogCollector data schemas
--------------------------------------

Structures légères dérivées de core.types, conçues pour :
- représenter les frames et métriques lues dans les logs,
- être sérialisables JSON (pour WebSocket / Dashboard),
- rester cohérentes avec les structures du pipeline temps réel.

Elles ne dépendent pas de numpy ni torch.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import time
from datetime import datetime

from . import logger

# ─────────────────────────────────────────────────────────────
#  Utilitaires de timestamps
# ─────────────────────────────────────────────────────────────

def _now_ts() -> float:
    return time.time()

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


# ─────────────────────────────────────────────────────────────
#  Pose (miroir minimal de core.types.Pose)
# ─────────────────────────────────────────────────────────────

@dataclass
class PoseLite:
    valid: bool = True
    matrix_shape: tuple = (4, 4)

    def to_dict(self):
        return {"valid": self.valid, "matrix_shape": self.matrix_shape}


# ─────────────────────────────────────────────────────────────
#  Meta (miroir de core.types.FrameMeta)
# ─────────────────────────────────────────────────────────────

@dataclass
class FrameMetaLite:
    frame_id: int
    ts: float
    device_name: str = "Image"
    spacing: tuple = (1.0, 1.0, 1.0)
    orientation: str = "UN"
    coord_frame: str = "Echographique"
    pose: PoseLite = field(default_factory=PoseLite)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ts": self.ts,
            "device_name": self.device_name,
            "spacing": self.spacing,
            "orientation": self.orientation,
            "coord_frame": self.coord_frame,
            "pose": self.pose.to_dict(),
        }


# ─────────────────────────────────────────────────────────────
#  Métriques GPU Transfer (kpi.log → event=copy_async)
# ─────────────────────────────────────────────────────────────

@dataclass
class GPUTransferLite:
    frame_id: int
    norm_ms: float
    pin_ms: float
    copy_ms: float
    total_ms: float
    ts_log: float = field(default_factory=_now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
#  Latences inter-étapes (Pipeline GPU-résident)
# ─────────────────────────────────────────────────────────────

@dataclass
class InterStageLatencyLite:
    frame_id: int
    rx_cpu: Optional[float] = None
    cpu_gpu: Optional[float] = None
    proc_gpu: Optional[float] = None
    gpu_cpu: Optional[float] = None
    cpu_tx: Optional[float] = None
    total: Optional[float] = None
    ts_log: float = field(default_factory=_now_ts)
    ts_iso: str = field(default_factory=_now_iso)

    def compute_total(self):
        if self.total is None:
            vals = [v for v in [self.rx_cpu, self.cpu_gpu, self.proc_gpu, self.gpu_cpu, self.cpu_tx] if v]
            self.total = sum(vals) if vals else 0.0

    def to_dict(self) -> Dict[str, Any]:
        self.compute_total()
        return {
            "frame_id": self.frame_id,
            "rx_cpu": self.rx_cpu,
            "cpu_gpu": self.cpu_gpu,
            "proc_gpu": self.proc_gpu,
            "gpu_cpu": self.gpu_cpu,
            "cpu_tx": self.cpu_tx,
            "total": self.total,
            "ts_log": self.ts_log,
            "ts_iso": self.ts_iso,
        }


# ─────────────────────────────────────────────────────────────
#  FrameAggregate — vue fusionnée (RX/PROC/TX + interstage + GPU)
# ─────────────────────────────────────────────────────────────

@dataclass
class FrameAggregate:
    frame_id: int
    meta: FrameMetaLite
    interstage: Optional[InterStageLatencyLite] = None
    gpu_transfer: Optional[GPUTransferLite] = None
    latency_rxproc: Optional[float] = None
    latency_proctx: Optional[float] = None
    latency_rxtx: Optional[float] = None
    rx: Optional[float] = None
    proc: Optional[float] = None
    tx: Optional[float] = None
    ts_wall: float = field(default_factory=_now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "meta": self.meta.to_dict(),
            "interstage": self.interstage.to_dict() if self.interstage else None,
            "gpu_transfer": self.gpu_transfer.to_dict() if self.gpu_transfer else None,
            "latency_rxproc": self.latency_rxproc,
            "latency_proctx": self.latency_proctx,
            "latency_rxtx": self.latency_rxtx,
            "ts_wall": self.ts_wall,
        }


# ─────────────────────────────────────────────────────────────
#  Timeline & snapshot global pour WS/dashboard
# ─────────────────────────────────────────────────────────────

@dataclass
class TimelineLite:
    frames: List[int] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)
    rx_cpu: List[float] = field(default_factory=list)
    cpu_gpu: List[float] = field(default_factory=list)
    proc_gpu: List[float] = field(default_factory=list)
    gpu_cpu: List[float] = field(default_factory=list)
    cpu_tx: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class DashboardSnapshotLite:
    latest: Optional[FrameAggregate] = None
    timeline: TimelineLite = field(default_factory=TimelineLite)
    gpu_transfer_stats: Dict[str, float] = field(default_factory=dict)
    fps_rx: Optional[float] = None
    fps_tx: Optional[float] = None
    health: str = "UNKNOWN"
    collector_ms_avg: Optional[float] = None
    collector_ms_p95: Optional[float] = None

    def to_dict(self):
        return {
            "latest": self.latest.to_dict() if self.latest else None,
            "timeline": self.timeline.to_dict(),
            "gpu_transfer_stats": self.gpu_transfer_stats,
            "fps_rx": self.fps_rx,
            "fps_tx": self.fps_tx,
            "health": self.health,
            "collector_ms_avg": self.collector_ms_avg,
            "collector_ms_p95": self.collector_ms_p95,
        }
