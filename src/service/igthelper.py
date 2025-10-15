"""Minimal helper for OpenIGTLink I/O using pyigtl.

This file contains a very small wrapper class `IGTGateway` that:
- connects as a client to a PlusServer OpenIGTLink endpoint to receive IMAGE
  messages,
- opens a server socket for Slicer to connect and receive the published masks.

Note: This is a lightweight skeleton for development. Replace/expand with
robust error handling and full message parsing for production.
"""

import threading      # ➜ Gestion du multi-threading (prévu pour extensions futures)
import time           # ➜ Utilisé pour les temporisations et timestamps
import numpy as np     # ➜ Génération et manipulation d’images simulées

try:
    import pyigtl     # ➜ Bibliothèque Python pour le protocole OpenIGTLink
except Exception:
    pyigtl = None     # ➜ Fallback si pyigtl n’est pas installé (mode simulation)


class IGTGateway:
    def __init__(self, plus_host: str, plus_port: int, slicer_port: int):
        self.plus_host = plus_host     # ➜ Adresse IP du serveur PlusServer
        self.plus_port = plus_port     # ➜ Port OpenIGTLink utilisé par PlusServer (entrée images)
        self.slicer_port = slicer_port # ➜ Port local pour écouter Slicer (sortie masques)
        self._running = False          # ➜ Drapeau d’exécution de la passerelle
        self._frame_counter = 0        # ➜ Compteur de frames simulées/envoyées

    def start(self):
        self._running = True
        # ➜ Active la passerelle (mock), en vrai : connexion client + serveur OpenIGTLink
        print(f"IGTGateway starting: plus={self.plus_host}:{self.plus_port} slicer_listen={self.slicer_port}")

    def stop(self):
        self._running = False
        # ➜ Stoppe la passerelle et les threads associés (mock)
        print("IGTGateway stopped")

    def image_generator(self):
        """Yield tuple (frame_id, (image_array, meta_dict)).

        This mock generator produces a grayscale image every 0.05s. Replace with
        actual pyigtl IMAGE message parsing when pyigtl is installed.
        """
        while self._running:  # ➜ Boucle active tant que la passerelle tourne
            self._frame_counter += 1                      # ➜ Incrémente l’ID d’image
            img = (np.random.rand(480, 640) * 255).astype('uint8')  # ➜ Génère une image 480x640 aléatoire
            meta = {"timestamp": time.time()}             # ➜ Ajoute un timestamp à la métadonnée
            yield self._frame_counter, (img, meta)        # ➜ Retourne l’image simulée et ses métadonnées
            time.sleep(0.05)                              # ➜ Fréquence ≈ 20 FPS (0.05s par frame)

    def send_mask(self, mask_array: np.ndarray, meta: dict):
        # ➜ Simule l’envoi d’un masque de segmentation vers Slicer
        # ➜ En vrai : construire et envoyer un message IMAGE via pyigtl.OpenIGTLinkServer
        print(f"send_mask called — mask shape: {mask_array.shape}")
