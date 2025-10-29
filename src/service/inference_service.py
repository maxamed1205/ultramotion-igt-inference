"""Service d'inférence minimal basé sur OpenIGTLink.

Ce script constitue un squelette de départ : il se connecte comme client à un
serveur PlusServer OpenIGTLink, s’abonne aux messages IMAGE, exécute une
inférence simulée (mockée) et republie un masque binaire IMAGE sous le nom
de périphérique `BoneMask` afin que 3D Slicer puisse le visualiser.

Remplacer la fonction mock_infer() par la vraie pipeline D-FINE → MobileSAM plus tard.
"""

import time                 # utilisé pour mesurer la latence et simuler le temps d’inférence
import csv                  # utilisé pour écrire les mesures de latence dans un fichier CSV
from pathlib import Path     # utilisé pour manipuler les chemins de fichiers de manière sûre

import logging               # module standard de gestion des logs

from igthelper import IGTGateway   # importe la classe IGTGateway (alias vers service.gateway.manager)

LOG = logging.getLogger("igt.service")  # crée un logger spécifique au service d’inférence

# Prefer the central logs directory created by the application bootstrap (main.py).
# WARNING: In production, start services via `main.py` which will create/initialize
# the `logs/` directory and (optionally) enable asynchronous logging. Scripts
# executed directly may create a local `logs/` folder as a fallback, but this
# can lead to divergent logging behaviour. Start via `main.py` to ensure a
# single canonical logging initialization.
# Fallback to creating a local logs/ directory if the package-level one is not present.
try:
    # Resolve project root relative to this file
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    LOG_PATH = PROJECT_ROOT / "logs"
    LOG_PATH.mkdir(parents=True, exist_ok=True)
except Exception:
    # Best-effort fallback: create logs in cwd
    LOG_PATH = Path("logs")
    LOG_PATH.mkdir(exist_ok=True)


def mock_infer(image):
    """Simule une étape d’inférence IA."""
    time.sleep(0.1)  # pause artificielle de 100 ms pour imiter le temps d’un modèle réel
    import numpy as np  # import local de NumPy (évite la dépendance si non utilisée ailleurs)
    return (np.zeros_like(image) > 0).astype('uint8')  # retourne un masque vide (même taille que l’image)


def main():
    """Point d’entrée principal du service d’inférence."""
    plusserver_host = "127.0.0.1"   # adresse IP du PlusServer (locale par défaut)
    plusserver_port = 18944         # port d’entrée des images échographiques
    slicer_listen_port = 18945      # port de sortie pour renvoyer les masques à 3D Slicer

    gw = IGTGateway(plusserver_host, plusserver_port, slicer_listen_port)  # initialise la passerelle IGTLink
    gw.start()  # démarre les threads RX/TX (connexion au PlusServer et à Slicer)

    log_file = LOG_PATH / "latency_log.csv"  # fichier CSV où seront enregistrées les latences
    with open(log_file, "w", newline="") as f:  # ouvre le fichier CSV en écriture
        writer = csv.writer(f)  # instancie un writer CSV
        writer.writerow(["frame_id", "t_in_igt", "t_recv", "t_after_infer", "t_sent_mask", "dt_infer_ms"])  # écrit les en-têtes

        try:
            # boucle principale : réception d'une image → inférence → envoi du masque → enregistrement latence
            for frame_id, (img, meta) in gw.image_generator():  # récupère les images reçues (mockées ou réelles)
                t_recv = time.time()              # horodatage local à la réception
                t_in_igt = meta.get("timestamp", None)  # timestamp original du message IGTLink
                
                # 🎯 MÉTRIQUES INTER-ÉTAPES : Marquer RX (début du workflow)
                try:
                    if hasattr(gw, 'stats') and frame_id is not None:
                        gw.stats.mark_interstage_rx(frame_id, t_recv)
                except Exception:
                    pass
                
                # 🎯 MÉTRIQUES INTER-ÉTAPES : Simuler CPU→GPU transfer (étape 1)
                t_cpu_gpu_start = time.time()
                time.sleep(0.001)  # Simuler 1ms de transfert CPU→GPU 
                t_cpu_gpu_end = time.time()
                try:
                    if hasattr(gw, 'stats') and frame_id is not None:
                        gw.stats.mark_interstage_cpu_to_gpu(frame_id, t_cpu_gpu_end)
                except Exception:
                    pass

                # 🎯 MÉTRIQUES INTER-ÉTAPES : Processing GPU (étape 2)
                t_proc_start = time.time()
                mask = mock_infer(img)            # exécute l'inférence simulée (IA fictive)
                t_proc_end = time.time()
                try:
                    if hasattr(gw, 'stats') and frame_id is not None:
                        gw.stats.mark_interstage_proc_done(frame_id, t_proc_end)
                except Exception:
                    pass
                
                # 🎯 MÉTRIQUES INTER-ÉTAPES : Simuler GPU→CPU transfer (étape 3)
                t_gpu_cpu_start = time.time()
                time.sleep(0.001)  # Simuler 1ms de transfert GPU→CPU
                t_gpu_cpu_end = time.time()
                try:
                    if hasattr(gw, 'stats') and frame_id is not None:
                        gw.stats.mark_interstage_gpu_to_cpu(frame_id, t_gpu_cpu_end)
                except Exception:
                    pass

                t_after = time.time()             # horodatage juste après l'inférence

                gw.send_mask(mask, meta)          # envoie le masque résultant à 3D Slicer (TX marqué dans slicer_server.py)
                t_sent = time.time()              # horodatage après envoi

                # enregistre la latence dans le CSV (pour chaque frame)
                writer.writerow([frame_id, t_in_igt, t_recv, t_after, t_sent, (t_after - t_recv) * 1000.0])
                f.flush()                         # force l’écriture immédiate pour suivre la performance en direct

        except KeyboardInterrupt:
            LOG.info("Arrêt manuel du service (Ctrl+C)")  # capture l’arrêt manuel de l’utilisateur
        finally:
            gw.stop()  # arrête proprement la passerelle et ferme les connexions (threads RX/TX, sockets)


if __name__ == "__main__":  # exécution directe du script
    main()  # lance le service d’inférence simulé
