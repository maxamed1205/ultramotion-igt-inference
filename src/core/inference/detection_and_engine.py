"""
Module d’inférence combinée D-FINE + MobileSAM (Process C)
==========================================================

💡 Mise à jour 20/10/2025 — intégration du pré-processing “FSM mask-aware”
---------------------------------------------------------------------------
Ce module gère désormais **les étapes 0 → 3** du pipeline de visibilité :

    0. Détection globale via D-FINE (bbox_t, conf_t)
    1. Gating macro : si conf_t < τ_conf → LOST direct
    2. Crop ROI autour de bbox_t
    3. Segmentation fine via MobileSAM
    4. Calcul des pondérations spatiales (W_edge, W_in, W_out)

Il constitue le moteur d’inférence (Process C) entre la réception GPU
et le module FSM (`core.state_machine.visibility_fsm`).

Flux typique :
    RawFrame (CPU) → cpu_to_gpu.py → GpuFrame (torch.Tensor)
    → detection_and_engine.prepare_inference_inputs()
    → visibility_fsm.evaluate_visibility()
    → ResultPacket → Gateway._outbox → Slicer

Rôle général
------------
- charger et initialiser les modèles IA (D-FINE et MobileSAM)
- exécuter l’inférence D-FINE → bbox/conf
- effectuer le crop ROI et la segmentation MobileSAM
- générer les pondérations spatiales utilisées pour les scores S₁–S₄
- fournir un dictionnaire normalisé pour le module FSM

Entrées attendues
-----------------
frame_tensor : image 2D échographique (numpy | torch)
model_paths  : dict avec au moins les clés {"dfine", "mobilesam"}

Sortie type
-----------
{
    "state_hint": "VISIBLE" | "LOST",
    "bbox_t": <tuple[int,int,int,int]> | None,
    "conf_t": <float>,
    "roi": <ndarray> | None,
    "mask_t": <ndarray> | None,
    "W_edge": <ndarray> | None,
    "W_in": <ndarray> | None,
    "W_out": <ndarray> | None,
}

Implémentations à prévoir
-------------------------
- initialize_models()        → chargement GPU des deux modèles
- run_detection()            → inférence D-FINE (bbox/conf)
- run_segmentation()         → inférence MobileSAM
- compute_mask_weights()     → calcul morphologique des pondérations
- prepare_inference_inputs() → orchestration globale (étapes 0 → 3)

Le reste du Process C (run_inference/fuse_outputs/process_inference_once)
reste valide pour la compatibilité pipeline et KPI.
"""

from typing import Any, Dict, Tuple, Optional
import logging
import numpy as np
import time

from core.types import GpuFrame, ResultPacket
from core.queues.buffers import get_queue_gpu, get_queue_out, try_dequeue, enqueue_nowait_out

LOG = logging.getLogger("igt.inference")
LOG_KPI = logging.getLogger("igt.kpi")


# ============================================================
# 1. Chargement et initialisation des modèles
# ============================================================

def initialize_models(model_paths: Dict[str, str], device: str = "cuda") -> Dict[str, Any]:
    """Charge et initialise les modèles D-FINE et MobileSAM.

    Args:
        model_paths: dictionnaire contenant les chemins {'dfine': ..., 'mobilesam': ...}
        device: 'cuda' ou 'cpu'

    Returns:
        dict {'dfine': model_dfine, 'mobilesam': model_sam}
    """
    raise NotImplementedError


# ============================================================
# 2. Étapes d’inférence mask-aware
# ============================================================

def run_detection(dfine_model: Any, frame_tensor: Any) -> Tuple[Tuple[int, int, int, int], float]:
    """Exécute le modèle D-FINE et renvoie (bbox_t, conf_t)."""
    raise NotImplementedError


def run_segmentation(sam_model: Any, roi: np.ndarray) -> Optional[np.ndarray]:
    """Exécute MobileSAM sur la ROI et retourne le mask binaire."""
    raise NotImplementedError


def compute_mask_weights(mask_t: np.ndarray, width_edge: int = 3, width_out: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les trois pondérations spatiales W_edge, W_in, W_out à partir du mask."""
    raise NotImplementedError


def prepare_inference_inputs(frame_t: np.ndarray, dfine_model: Any, sam_model: Any, tau_conf: float = 0.5) -> Dict[str, Any]:
    """
    Orchestration complète des étapes 0 → 3.

    0. Passe l’image dans D-FINE → bbox/conf.
    1. Si conf < τ_conf → renvoie state_hint='LOST'.
    2. Crop ROI autour de la bbox.
    3. Passe la ROI dans MobileSAM.
    4. Calcule les pondérations spatiales (W_edge/W_in/W_out).

    Returns:
        dictionnaire prêt pour visibility_fsm.evaluate_visibility().
    """
    raise NotImplementedError


# ============================================================
# 3. Compatibilité Process C (inférence générique)
# ============================================================

def run_inference(frame_tensor: GpuFrame, stream_infer: Any = None) -> Tuple[ResultPacket, float]:
    """Exécute (mock) l’inférence GPU et retourne un ResultPacket minimal."""
    raise NotImplementedError


def fuse_outputs(mask: Any, score: float, state: str) -> ResultPacket:
    """Fusionne les sorties et renvoie un ResultPacket standardisé."""
    raise NotImplementedError


# ============================================================
# 4. Routine de boucle unique (mock actuelle)
# ============================================================

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
