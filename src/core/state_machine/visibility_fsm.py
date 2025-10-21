"""
Finite-State Machine de visibilité (S1..S4 / C1..C3, version mask-aware, option B).

Ce module orchestre la logique de visibilité :
  - reçoit les données préparées par detection_and_engine.py
    (D-FINE + MobileSAM + pondérations spatiales),
  - calcule les scores S₁–S₄ (intra-frame, toujours actifs),
  - calcule les scores C₁–C₃ uniquement pendant la phase RELOCALIZING,
  - applique la logique de transition entre les états :
        VISIBLE  → RELOCALIZING → LOST  (et inversement),
  - renvoie l’état courant et les scores pour supervision et logs KPI.

Principe de calcul (Option B)
-----------------------------
    - En état **VISIBLE**  : seuls les scores S₁–S₄ sont calculés.
    - En état **RELOCALIZING** : S₁–S₄ + C₁–C₃ (pour valider le recalage).
    - En état **LOST**     : aucun score, attente d’une détection fiable.

Entrées :
    prepared   : dict issu de detection_and_engine.prepare_inference_inputs()
                 (bbox_t, conf_t, roi, mask_t, W_edge, W_in, W_out, state_hint)
    frame_prev : image précédente (pour C2, S4)
    pose_t, pose_prev : poses 3D de la sonde
    thresholds : dict des seuils FSM (τ_conf, θ_on/off, θ_cont)
    current_state : état courant avant mise à jour ("VISIBLE", "RELOCALIZING", "LOST")

Sorties :
    - new_state : {"VISIBLE", "RELOCALIZING", "LOST"}
    - scores    : {S1..S4, [C1..C3], S_total, conf_t}
"""

from typing import Any, Dict, Tuple, Optional

import logging
import numpy as np
import torch

LOG = logging.getLogger("igt.fsm")
LOG_KPI = logging.getLogger("igt.kpi")

# ======================================================================
# 1. Évaluation FSM — version GPU full streams
# ======================================================================

def evaluate_visibility(
    prepared: Dict[str, Any],
    frame_prev: Optional[np.ndarray],
    pose_t: Optional[np.ndarray],
    pose_prev: Optional[np.ndarray],
    thresholds: Dict[str, float],
    current_state: str,
    device: str = "cuda",
) -> Tuple[str, Dict[str, float]]:
    """
    Exécute les calculs S₁–S₄ et C₁–C₃ sur GPU (multi-stream CUDA), puis met à jour le FSM.

    Étapes :
      0. Gating macro (conf_t)
      1. Lancement des kernels S₁–S₄ sur 4 streams
      2. Si relocalisation : lancement des kernels C₁–C₃ sur 3 streams supplémentaires
      3. Synchronisation et agrégation
      4. Application de la logique FSM
    """

    conf_t = float(prepared.get("conf_t", 0.0))
    tau_conf = thresholds.get("tau_conf", 0.5)

    if prepared.get("state_hint") == "LOST" or conf_t < tau_conf:
        return "LOST", {"conf_t": conf_t}

    # === Récupération des données ===
    bbox_t = prepared["bbox_t"]
    roi = prepared["roi"]
    mask_t = prepared["mask_t"]
    W_edge, W_in, W_out = prepared["W_edge"], prepared["W_in"], prepared["W_out"]

    # ==================================================================
    # Bloc 1 — S₁–S₄ : lancés sur 4 streams GPU parallèles
    # ==================================================================
    streams_S = [torch.cuda.Stream(device=device) for _ in range(4)]

    with torch.cuda.stream(streams_S[0]):
        S1 = compute_S1_sharpness(roi, W_edge)
    with torch.cuda.stream(streams_S[1]):
        S2 = compute_S2_snr(roi, W_in, W_out)
    with torch.cuda.stream(streams_S[2]):
        S3 = compute_S3_continuity(mask_t, roi)
    with torch.cuda.stream(streams_S[3]):
        S4 = compute_S4_stability(roi, frame_prev, pose_t, pose_prev, W_edge)

    # Attente synchronisation S-streams
    for s in streams_S:
        s.synchronize()

    S_total = 0.25 * (S1 + S2 + S3 + S4)

    # ==================================================================
    # Bloc 2 — C₁–C₃ : seulement si RELOCALIZING (3 streams GPU)
    # ==================================================================
    C1 = C2 = C3 = None
    C_mean = None

    if current_state == "RELOCALIZING":
        streams_C = [torch.cuda.Stream(device=device) for _ in range(3)]
        with torch.cuda.stream(streams_C[0]):
            C1 = compute_C1_geometric_stability(bbox_t, mask_t)
        with torch.cuda.stream(streams_C[1]):
            C2 = compute_C2_visual_coherence(roi, frame_prev, mask_t)
        with torch.cuda.stream(streams_C[2]):
            C3 = compute_C3_spatial_depth(mask_t, pose_t, pose_prev)

        # Synchronisation C-streams
        for s in streams_C:
            s.synchronize()

        C_mean = float(np.mean([C1, C2, C3]))

    # ==================================================================
    # Bloc 3 — Logique FSM
    # ==================================================================

    # ⚠️ TODO [Phase 3 - Scheduler adaptatif]
    # Si conf_t reste élevé sur N frames consécutives et que la cible est stable (S_total>θ_on),
    # permettre de "skipper" temporairement la détection D-FINE pendant k frames.
    # L'état reste VISIBLE, seule la segmentation MobileSAM est actualisée.
    # À implémenter via un compteur de stabilité (heartbeat D-FINE toutes les K frames).

    θ_on = thresholds.get("theta_on", 0.6)
    θ_off = thresholds.get("theta_off", 0.4)
    θ_cont = thresholds.get("theta_cont", 0.5)

    if S_total < θ_off:
        new_state = "LOST"

    elif current_state == "VISIBLE" and S_total < θ_off:
        new_state = "RELOCALIZING"

    elif current_state == "RELOCALIZING":
        if C_mean is not None and C_mean > θ_cont and S_total > θ_on:
            new_state = "VISIBLE"
        elif S_total < θ_off or conf_t < tau_conf:
            new_state = "LOST"
        else:
            new_state = "RELOCALIZING"

    elif current_state == "LOST" and conf_t > tau_conf:
        new_state = "RELOCALIZING"

    else:
        new_state = current_state

    # ==================================================================
    # Bloc 4 — Agrégation finale des scores
    # ==================================================================
    scores = {
        "S1": float(S1),
        "S2": float(S2),
        "S3": float(S3),
        "S4": float(S4),
        "S_total": float(S_total),
        "C1": float(C1) if C1 is not None else None,
        "C2": float(C2) if C2 is not None else None,
        "C3": float(C3) if C3 is not None else None,
        "C_mean": float(C_mean) if C_mean is not None else None,
        "conf_t": float(conf_t),
    }

    return new_state, scores


# ======================================================================
# 2. Interfaces GPU (kernels à implémenter)
# ======================================================================

def compute_S1_sharpness(image: np.ndarray, W_edge: np.ndarray) -> float:
    """GPU kernel : variance du Laplacien pondéré (mesure de netteté)."""
    raise NotImplementedError


def compute_S2_snr(image: np.ndarray, W_in: np.ndarray, W_out: np.ndarray) -> float:
    """GPU kernel : rapport signal/bruit local (intensité intérieure vs extérieure)."""
    raise NotImplementedError


def compute_S3_continuity(mask: np.ndarray, image: np.ndarray) -> float:
    """GPU kernel : continuité du trait osseux via squelette binaire."""
    raise NotImplementedError


def compute_S4_stability(
    frame_t: np.ndarray,
    frame_prev: Optional[np.ndarray],
    pose_t: Optional[np.ndarray],
    pose_prev: Optional[np.ndarray],
    W_edge: np.ndarray,
) -> float:
    """GPU kernel : cohérence temporelle et flou de mouvement (FFT ou corrélation)."""
    raise NotImplementedError


def compute_C1_geometric_stability(bbox_t, mask_t) -> float:
    """GPU kernel : cohérence géométrique (IoU bbox vs mask et stabilité inter-frame)."""
    raise NotImplementedError


def compute_C2_visual_coherence(frame_t, frame_prev, mask_t) -> float:
    """GPU kernel : cohérence visuelle inter-frame (SSIM/NCC pondéré par mask)."""
    raise NotImplementedError


def compute_C3_spatial_depth(mask_t, pose_t, pose_prev) -> float:
    """GPU kernel : cohérence spatiale / profondeur 3D du squelette osseux."""
    raise NotImplementedError