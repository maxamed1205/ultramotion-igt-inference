"""Types partagés pour la pipeline A->B->C->D.

Ce module définit les dataclasses servant de contrat entre les étapes
de la pipeline. Les objets sont légers, typés et conçus pour être
manipulés dans les queues (RawFrame, GpuFrame, ResultPacket).

Règles :
- les queues transportent uniquement ces objets (RawFrame -> GpuFrame -> ResultPacket),
- Pydantic/validation uniquement hors du hot-path temps réel,
- préférence dataclass pour lisibilité et facilité d'usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Any, TYPE_CHECKING
import time
import logging
import numpy as np

LOG = logging.getLogger("igt.types")

# Avoid importing torch at runtime in the hot-path. Import it only for
# type-checking to keep runtime lightweight.
if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from torch import Tensor  # type: ignore


StateLiteral = Literal["VISIBLE", "RELOCALIZING", "LOST"]


# matrice identité partagée (créée une seule fois pour éviter copies inutiles)
_DEFAULT_IDENTITY = np.eye(4, dtype=np.float32)


@dataclass(slots=True)
class Pose:
    """Représentation d'une pose homogène 4x4.

    Champs:
      - matrix: np.ndarray dtype float32 shape (4,4)
      - valid: bool indiquant si la pose est considérée comme valide

    Méthodes futures (squelettes):
      - as_tensor(device): conversion paresseuse vers torch.Tensor (optionnelle)
      - from_igt(transform_message): construire une Pose depuis un message IGT
      - from_translation_rotation(t, q): construire depuis translation+quaternion
      - to_dict(): sérialiser en dict léger

    Notes de performance:
      - __post_init__ effectue uniquement des assertions légères en mode debug
        (si __debug__ est vrai) pour éviter le coût en production.
    """

    # use a copy of the shared identity to avoid accidental shared-mutation
    matrix: np.ndarray = field(default_factory=lambda: _DEFAULT_IDENTITY.copy())
    valid: bool = True

    def __post_init__(self) -> None:
        # validations légères uniquement en mode debug
        if __debug__:
            assert isinstance(self.matrix, np.ndarray), "Pose.matrix doit être un numpy.ndarray"
            assert self.matrix.shape == (4, 4), "Pose.matrix doit être de forme (4,4)"
            assert self.matrix.dtype == np.float32, "Pose.matrix doit être dtype float32"

    def as_tensor(self, device: str = "cpu"):
        """Conversion paresseuse vers torch.Tensor.

        Args:
            device: cible pour le tensor ('cpu' ou 'cuda:0', ...)

        Returns:
            torch.Tensor représentant la matrice sur le device demandé.

        Note: import torch uniquement si nécessaire. Implémentation future.
        """
        raise NotImplementedError

    @classmethod
    def from_igt(cls, transform_message: object) -> "Pose":
        """Construire une Pose depuis un message OpenIGTLink.

        Args:
            transform_message: message reçu via IGT (format dépendant de l'implémentation)

        Returns:
            Pose

        Implémentation future: parser le message et remplir matrix/valid.
        """
        # Optimized minimal implementation: accept an object that exposes
        # a `matrix` attribute (4x4 iterable). Return a contiguous float32 array.
        try:
            mat = getattr(transform_message, "matrix", None)
            if mat is None:
                return cls(valid=False)
            arr = np.asarray(mat, dtype=np.float32)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            return cls(matrix=arr, valid=True)
        except Exception as e:
            LOG.debug("Pose.from_igt failed: %r", e)
            return cls(valid=False)

    @classmethod
    def from_translation_rotation(cls, t: Tuple[float, float, float], q: Tuple[float, float, float, float]) -> "Pose":
        """Construire une Pose depuis translation (t) et quaternion (q).

        Args:
            t: translation (x,y,z)
            q: quaternion (x,y,z,w)

        Returns:
            Pose

        Implémentation future: convertir t+q en matrice 4x4.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Sérialiser la pose en dictionnaire léger (pour logging/config).

        Implémentation future.
        """
        return {"valid": bool(self.valid), "matrix_shape": self.matrix.shape}


@dataclass(slots=True)
class FrameMeta:
    """Métadonnées attachées à chaque frame.
    Champs:
        - frame_id: identifiant entier unique par frame
        - ts: timestamp Unix en secondes (float)
        - pose: `Pose` (contient la matrice homogène 4x4 float32 et un flag valid)
        - spacing: tuple (sx, sy)
        - device_name: nom du device/source IGT
    """

    frame_id: int
    ts: float
    pose: Pose = field(default_factory=Pose)
    # spacing as (sx, sy, sz). Accept legacy 2-tuple for compatibility.
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: str = "UN"
    coord_frame: str = "Echographique"
    device_name: str = "Image"

    def __post_init__(self) -> None:
        # lightweight validation and compatibility handling
        if __debug__:
            # normalize spacing length-2 to length-3 for backward compatibility
            if len(self.spacing) == 2:
                sx, sy = self.spacing
                self.spacing = (float(sx), float(sy), 1.0)
            assert len(self.spacing) == 3, "spacing doit être (sx, sy, sz)"
            assert isinstance(self.ts, float), "ts doit être float"
            assert isinstance(self.device_name, str)
            # permit a set of orientations; be permissive in production
            if self.orientation not in ("UN", "LF", "RF", "HF"):
                # don't raise in production, just log in debug
                LOG.debug("Non-standard orientation: %s", self.orientation)

    @property
    def trace_id(self) -> str:
        """Legacy trace id kept for backward compatibility (frame_id only)."""
        return f"{self.frame_id}"

    @property
    def device_trace_id(self) -> str:
        """Device-scoped trace id e.g. 'Image#123'."""
        return f"{self.device_name}#{self.frame_id}"

    def summary(self) -> str:
        return f"{self.device_name}#{self.frame_id} [{self.orientation}] {self.ts:.3f}s"

    @classmethod
    def from_igt(cls, msg: object) -> "FrameMeta":
        """Construct FrameMeta from an IGTLink-like message object.

        The `msg` is expected to expose `.metadata` (dict) and optional
        `.timestamp` / `.device_name`. This is permissive and applies
        reasonable defaults when fields are missing.
        """
        md = getattr(msg, "metadata", {}) or {}
        # spacing may be present as list/tuple
        spacing = md.get("spacing") or md.get("Spacing") or (1.0, 1.0, 1.0)
        # orientation / coord frame
        orientation = md.get("orientation") or md.get("ImageOrientation") or "UN"
        coord_frame = md.get("coord_frame") or md.get("CoordinateFrame") or "Echographique"
        ts = float(getattr(msg, "timestamp", getattr(msg, "ts", time.time())))
        device_name = getattr(msg, "device_name", md.get("DeviceName", "Image"))
        frame_id = int(md.get("FrameNumber", md.get("frame_number", 0)))
        pose = Pose.from_igt(msg)
        return cls(frame_id=frame_id, ts=ts, pose=pose, spacing=tuple(spacing), orientation=orientation, coord_frame=coord_frame, device_name=device_name)

    def to_igt_dict(self) -> dict:
        """Convert FrameMeta to a lightweight dict compatible with pyigtl metadata."""
        return {
            "DeviceName": self.device_name,
            "Timestamp": float(self.ts),
            "ImageOrientation": self.orientation,
            "CoordinateFrame": self.coord_frame,
            "Spacing": tuple(self.spacing),
            "FrameNumber": int(self.frame_id),
        }


# Simple circular pool of FrameMeta for hot-path reuse.
# This is an optional, low-cost micro-optimization to reduce GC
# pressure in the receiver when many frames arrive per second.
_POOL: list[FrameMeta] = [FrameMeta(i, 0.0) for i in range(64)]
_IDX: int = 0


def reuse_frame_meta() -> FrameMeta:
    """Return a FrameMeta instance from a small circular pool.

    Note: the returned object is reused and will be mutated by callers.
    Use only in hot-path code where allocations are the bottleneck and
    you control concurrent access.
    """
    global _IDX
    m = _POOL[_IDX]
    _IDX = (_IDX + 1) % len(_POOL)
    return m


@dataclass
class RawFrame:
    """Frame brute en CPU (hot-path initial).

    image: numpy.ndarray attendu en uint8 (H x W)
    meta: FrameMeta
    """

    image: np.ndarray
    meta: FrameMeta


@dataclass
class GpuFrame:
    """Représentation d'une frame prête pour le GPU.

    Champs :
      - tensor : torch.Tensor optionnel (shape typique 1x1xH x W).
        Le tensor peut être None si la frame n’a pas encore été transférée vers le GPU.
        L'import de torch n'est effectué qu'en mode typing (TYPE_CHECKING),
        afin d'éviter tout coût inutile dans le hot-path temps réel.
      - meta : FrameMeta associée à cette frame.
    """


    # At runtime this is usually a torch.Tensor, but we annotate as Any to
    # avoid importing torch during normal execution when it's not available.
    tensor: Optional["Tensor"]
    meta: FrameMeta
    # Optional CUDA stream associated with the transfer. None for CPU fallback.
    stream: Optional[object] = None


@dataclass
class ResultPacket:
    """Packet résultat destiné à la sortie/Slicer.

    mask: numpy.ndarray (H x W uint8, 0/255). Le mask doit être spatialement
    aligné avec `meta.spacing` (mêmes dimensions / résolution).
    score: float (0..1)
    state: StateLiteral
    meta: FrameMeta
    """

    mask: np.ndarray
    score: float
    state: StateLiteral
    meta: FrameMeta
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None


@dataclass
class IGTLinkContract:
    """Minimal representation of an IGTLink header useful for fast parsing.

    Fields chosen to be compact and match the minimal header used in this
    project (device_name, message_type, timestamp, body_size, crc, version).
    """
    device_name: str
    message_type: str
    timestamp: float
    body_size: int
    crc: int = 0
    version: int = 2

    @classmethod
    def from_bytes(cls, data: bytes) -> "IGTLinkContract":
        """Decode a minimal IGTLink-like header (first 56-64 bytes).

        This is permissive: fields missing/short buffers are handled safely.
        """
        # ensure at least the minimal slice lengths
        name = data[0:20].decode("ascii", errors="ignore").rstrip("\x00") if len(data) >= 20 else ""
        msg_type = data[20:32].decode("ascii", errors="ignore").rstrip("\x00") if len(data) >= 32 else ""
        ts = 0.0
        body_size = 0
        try:
            if len(data) >= 40:
                ts = float(int.from_bytes(data[32:40], "big"))
            if len(data) >= 48:
                body_size = int.from_bytes(data[40:48], "big")
        except Exception:
            ts = 0.0
            body_size = 0
        return cls(device_name=name, message_type=msg_type, timestamp=ts, body_size=body_size)

    def to_bytes(self) -> bytes:
        """Make a minimal header bytes blob (not a full spec serialization).

        Used primarily for tests and compatibility helpers.
        """
        name = self.device_name.encode("ascii", errors="ignore")[:20].ljust(20, b"\x00")
        mtype = self.message_type.encode("ascii", errors="ignore")[:12].ljust(12, b"\x00")
        ts = int(self.timestamp).to_bytes(8, "big")
        bsize = int(self.body_size).to_bytes(8, "big")
        crc = int(self.crc).to_bytes(8, "big")
        return name + mtype + ts + bsize + crc
