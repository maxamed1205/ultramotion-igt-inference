"""Skeleton inference service for OpenIGTLink.

This script is a minimal starting point: it connects as a client to a PlusServer
OpenIGTLink endpoint, subscribes to IMAGE messages, performs a mocked
inference step, and republishes a binary mask IMAGE under the device name
`BoneMask` for Slicer to consume.

Replace the mock_infer() with the real D-FINE -> MobileSAM pipeline later.
"""

import time                 # ➜ Pour mesurer la latence et simuler le temps d’inférence
import csv                  # ➜ Pour écrire les mesures de latence dans un fichier CSV
from pathlib import Path     # ➜ Pour gérer les chemins de fichiers proprement

from igthelper import IGTGateway   # ➜ Classe de communication OpenIGTLink (mockée pour le dev)

LOG_PATH = Path("logs")      # ➜ Dossier de logs à la racine du projet
LOG_PATH.mkdir(exist_ok=True)  # ➜ Créé le dossier s’il n’existe pas déjà


def mock_infer(image):
    # ➜ Simule une étape d’inférence (temps de traitement artificiel)
    time.sleep(0.1)  # ➜ Pause de 100 ms pour reproduire le comportement d’un modèle IA réel
    import numpy as np  # ➜ Import local (pour éviter les dépendances inutiles si non utilisé)
    # ➜ Crée un masque vide (même forme que l’image d’entrée)
    return (np.zeros_like(image) > 0).astype('uint8')


def main():
    # ➜ Configuration réseau par défaut : PlusServer → port 18944, Slicer → port 18945
    plusserver_host = "127.0.0.1"   # ➜ Adresse IP de PlusServer (local par défaut)
    plusserver_port = 18944         # ➜ Port d’entrée des images échographiques
    slicer_listen_port = 18945      # ➜ Port de sortie pour renvoyer les masques à Slicer

    gw = IGTGateway(plusserver_host, plusserver_port, slicer_listen_port)  # ➜ Initialise la passerelle IGT
    gw.start()  # ➜ Démarre la communication (mockée ici)

    log_file = LOG_PATH / "latency_log.csv"  # ➜ Fichier CSV pour enregistrer les temps de traitement
    with open(log_file, "w", newline="") as f:  # ➜ Ouvre le fichier CSV en écriture
        writer = csv.writer(f)  # ➜ Crée un writer CSV
        # ➜ Écrit l’en-tête de colonnes (timestamps et durée d’inférence)
        writer.writerow(["frame_id", "t_in_igt", "t_recv", "t_after_infer", "t_sent_mask", "dt_infer_ms"])

        try:
            # ➜ Boucle principale : réception d’image → inférence → envoi du masque → log
            for frame_id, (img, meta) in gw.image_generator():  # ➜ Récupère les images (mockées)
                t_recv = time.time()              # ➜ Timestamp à la réception
                t_in_igt = meta.get("timestamp", None)  # ➜ Timestamp d’origine (envoyé par PlusServer)

                mask = mock_infer(img)            # ➜ Exécute le modèle IA simulé
                t_after = time.time()             # ➜ Timestamp après inférence

                gw.send_mask(mask, meta)          # ➜ Renvoie le masque à Slicer via OpenIGTLink
                t_sent = time.time()              # ➜ Timestamp après l’envoi du masque

                # ➜ Enregistre toutes les mesures temporelles dans le fichier CSV
                writer.writerow([frame_id, t_in_igt, t_recv, t_after, t_sent, (t_after - t_recv) * 1000.0])
                f.flush()                         # ➜ Force l’écriture immédiate (utile pour suivi en direct)

        except KeyboardInterrupt:
            print("Shutting down")                # ➜ Arrêt manuel via Ctrl+C
        finally:
            gw.stop()                             # ➜ Stoppe proprement la passerelle IGT (fermeture threads/sockets)


if __name__ == "__main__":  # ➜ Exécute main() uniquement si le script est lancé directement
    main()                  # ➜ Lance le service d’inférence mocké
