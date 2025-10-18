"""Simule la passerelle OpenIGTLink pour les tests et le développement.

Ce module fournit la classe `MockIGTGateway` qui génère des images aléatoires
et permet de tester localement sans matériel réel (PlusServer / Slicer).

Attention: simulation — ne pas utiliser ce module en production.
"""

from typing import Dict, Tuple, Iterator
import time
import numpy as np
import logging

LOG = logging.getLogger("igt.simulation")


class MockIGTGateway:
    """Passerelle simulée pour tests.

    Attributs:
        _running: bool indiquant si la simulation est active.
        _frame_counter: compteur d'images générées.
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._frame_counter: int = 0

    def start(self) -> None:
        """Démarre la génération d'images simulées.

        Cette méthode définit l'état interne à `running`. La génération réelle
        se fait via `generate_images()`.
        """
        self._running = True
        LOG.info("MockIGTGateway start() — simulation démarrée")

    def stop(self) -> None:
        """Arrête la génération d'images.

        Met simplement `_running` à False. Ne tue pas de threads car ce
        squelette reste mono-threaded pour la simplicité.
        """
        self._running = False
        LOG.info("MockIGTGateway stop() — simulation arrêtée")

    def generate_images(self, interval_s: float = 0.05) -> Iterator[Tuple[np.ndarray, Dict]]:
        """Génère un itérateur d'images simulées.

        Args:
            interval_s: intervalle entre deux images (s).

        Yields:
            Tuple (image, meta) où `image` est un numpy.ndarray 480x640 uint8
            et `meta` contient au moins le timestamp.

        Remarques:
            - Cette fonction est volontairement simple et synchrone pour
              faciliter les tests.
            - # TODO: permettre un callback pour l'envoi asynchrone.
        """
        while self._running:
            self._frame_counter += 1
            img = (np.random.rand(480, 640) * 255).astype('uint8')
            meta = {"timestamp": time.time(), "frame_id": self._frame_counter}
            yield img, meta
            time.sleep(interval_s)

    def send_mask(self, mask_array: np.ndarray, meta: Dict) -> None:
        """Simule l'envoi d'un masque vers Slicer.

        Pour l'instant, affiche seulement la forme du masque et les clés meta.
        """
        LOG.info("MockIGTGateway send_mask() — mask shape: %s, meta keys: %s", getattr(mask_array, "shape", None), list(meta.keys()))


class StubGateway:
    """Stub Gateway — utilisée quand IGTGateway n'est pas disponible.

    Fournit une interface compatible (méthode receive_image) mais ne renvoie
    jamais de frame. Sert pour les tests ou l'exécution sans PlusServer.
    """

    def receive_image(self):
        return None
