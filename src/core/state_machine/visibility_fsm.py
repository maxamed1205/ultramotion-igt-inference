"""Visibility finite-state machine (S1..S4 / C1..C3).

Rôle : calcul S1–S4 / C1–C3, hystérésis, décision (VISIBLE / LOST / RELOCALIZING).
"""

from typing import Any, Dict, Tuple


def evaluate_visibility(frame: Any, pose: Any) -> Tuple[str, Dict[str, float]]:
    """Évalue la visibilité pour une frame donnée.

    Returns:
        (state, scores) où state est un string ('VISIBLE','LOST','RELOCALIZING')
        et scores contient les métriques S/C.
    """
    raise NotImplementedError


def update_state_machine(current_state: str, metrics: Dict[str, float]) -> str:
    """Met à jour la machine d'état en appliquant hysteresis/anti-rebond.

    Args:
        current_state: état courant
        metrics: métriques calculées par evaluate_visibility

    Returns:
        nouvel état
    """
    raise NotImplementedError


def is_frame_valid_for_gpu(state: str) -> bool:
    """Indique si une frame dans l'état donné doit être envoyée au GPU."""
    raise NotImplementedError
