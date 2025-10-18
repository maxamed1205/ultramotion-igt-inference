"""Émetteur d'événements pour la passerelle.

Petit utilitaire qui garde un seul callback et l'appelle de manière sécurisée.
"""
import logging
from typing import Any, Callable, Optional

LOG = logging.getLogger("igt.gateway.events")  # Définit un logger pour ce module


class EventEmitter:
    def __init__(self) -> None:
        self._callback: Optional[Callable[[str, Any], None]] = None  # Contient la fonction de rappel enregistrée (callback unique)

    def on_event(self, callback: Callable[[str, Any], None]) -> None:  # Enregistre une fonction à appeler lors de la réception d'un événement
        self._callback = callback  # Stocke la fonction dans l'attribut interne

    def emit(self, name: str, payload: Any) -> None:  # Émet un événement avec un nom et un contenu associé
        if not self._callback:  # Si aucun callback n'est enregistré, on ne fait rien
            return
        try:
            self._callback(name, payload)  # Appelle le callback avec le nom et le contenu de l'événement
        except Exception:
            LOG.exception("Le callback d'événement a échoué : %s", name)  # Enregistre l'erreur sans bloquer le reste du programme
