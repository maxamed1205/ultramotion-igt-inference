"""Thread client Plus : reçoit les messages IGTLink de type IMAGE et les empile sous forme de RawFrame.

Cette implémentation privilégie l'usage de la bibliothèque `pyigtl` (client réel).
Si `pyigtl` n'est pas installée (cas typique en développement ou CI), 
une version simulée génère des tableaux numpy à environ 25 FPS.
L'API est volontairement simple pour que cette fonction puisse être directement 
utilisée dans `IGTGateway.start()` comme `run_plus_client`.
"""
from glob import glob
from typing import Optional, Callable
import time
import logging
import numpy as np
from glob import glob  # Explicitly import the glob module
from pathlib import Path
import cv2  # Ajoute cette ligne pour importer le module cv2

from core.types import RawFrame, FrameMeta  # structures de données utilisées pour encapsuler image + métadonnées

LOG = logging.getLogger("igt.plus_client")  # création d'un logger spécifique pour ce module

# Variables globales pour la simulation
_image_files = None
_image_index = 0

def run_plus_client(mailbox, stop_event, host, port, stats_cb: Optional[Callable] = None, event_cb: Optional[Callable] = None) -> None:
    """Thread RX : client IGTLink vers PlusServer.

    - Reçoit des messages IMAGE (et plus tard TRANSFORM),
    - Empile les RawFrame dans la mailbox (file d'entrée de la passerelle),
    - Appelle stats_cb(fps, ts) toutes les 2 secondes si fourni,
    - Appelle event_cb('rx_connect', {...}) / ('rx_disconnect', {...}) pour notifier les connexions.
    """
    global _image_files, _image_index
    
    try:
        import pyigtl  # tente d'importer la vraie bibliothèque OpenIGTLink (communication réseau)
    except Exception:
        pyigtl = None  # si pyigtl est absent, on passe en mode simulation
        LOG.debug("pyigtl non disponible ; utilisation du simulateur pour plus_client")  # message de debug

    try:
        if event_cb:  # si une fonction callback d'événement est fournie
            event_cb("rx_connect", {"host": host, "port": port})  # signale la connexion RX au gestionnaire d'événements
    except Exception:
        LOG.exception("event_cb a échoué lors de la notification de connexion")  # log l'erreur sans interrompre le flux

    fps_window = []  # liste des timestamps des frames reçues (sert à calculer le FPS moyen)
    last_stats = time.time()  # enregistre le moment du dernier envoi de statistiques
    frame_id = 0  # compteur d'identifiants de frames (incrémenté à chaque nouvelle image)

    try:
        if pyigtl:  # si la bibliothèque pyigtl est disponible (mode réel)
            try:
                client = pyigtl.OpenIGTLinkClient(host, port)  # crée un client OpenIGTLink vers PlusServer (API hypothétique)
                client.connect()  # tente de se connecter au serveur IGTLink distant
            except Exception:
                LOG.exception("Échec de connexion du client pyigtl ; bascule vers le mode simulateur")  # message d'erreur
                pyigtl = None  # désactive pyigtl pour passer en mode simulation

    except Exception:
        pyigtl = None  # en cas d'erreur imprévue, on force le mode simulateur
    
    # Initialisation pour le mode simulation
    if not pyigtl:
        DATASET_PATH = Path(r"C:\Users\maxam\Desktop\TM\ultramotion-igt-inference\Video_001")
        if _image_files is None:
            _image_files = sorted(glob(str(DATASET_PATH / "*.jpg")))
            _image_index = 0
            LOG.info(f"[RX - SIM] Chargement des images depuis le dossier pour simulation ({len(_image_files)} images trouvées)")

    # Boucle principale de réception (tourne en continu tant que le thread n'est pas arrêté)
    while not stop_event.is_set():  # continue tant que le signal d'arrêt global n'a pas été activé
        start = time.time()  # enregistre l'heure de début du cycle (utile pour le calcul de FPS ou latence)

        if stop_event.is_set(): # Si l'arrêt a été demandé, on quitte proprement la boucle
            LOG.info("[RX - SIM] Arrêt demandé, sortie de la boucle principale")
            break

        try:
            if pyigtl:  # si la bibliothèque pyigtl est disponible → mode réel (connexion à PlusServer)
                print("[REAL] Tentative de réception d'un message IGTLink depuis PlusServer")
                try:
                    msg = client.receive(timeout=0.1)  # tente de recevoir un message IGTLink (attente max 100 ms)
                except Exception:
                    msg = None  # en cas d'erreur, on ignore le message

                if msg is None:  # aucun message reçu pendant ce cycle
                    time.sleep(0.01)  # petite pause pour éviter de saturer le CPU
                    continue  # retourne au début de la boucle pour réessayer

                # Conversion du message IGTLink en tableau numpy (image brute)
                try:
                    arr = np.asarray(msg.image, dtype=np.uint8)  # tente de lire les données image du message
                except Exception:
                    raise RuntimeError("Erreur de lecture de l'image. L'exécution est arrêtée.")  # Si échec, lève une exception pour arrêter le code

                frame_id += 1  # incrémente le compteur global d'images reçues
                meta = FrameMeta(frame_id=frame_id, ts=time.time())  # crée les métadonnées associées (id + timestamp)
                rf = RawFrame(image=arr, meta=meta)  # encapsule l'image et ses métadonnées dans un objet RawFrame
                try:
                    mailbox.append(rf)  # place la frame dans la file d'entrée de la passerelle (AdaptiveDeque)
                except Exception:
                    try:
                        from core.monitoring.kpi import increment_drops
                        increment_drops(image_id=f"frame_{frame_id}", name="rx.drop_total", delta=1, emit=True)  # passer frame_id comme image_id
                    except Exception:
                        pass
                    LOG.exception("Échec d'ajout de la frame reçue dans la mailbox")  # log en cas d'erreur d'insertion

            else:  # Mode simulation - traiter UNE SEULE image par cycle
                if _image_files and _image_index < len(_image_files):
                    image_path = _image_files[_image_index]
                    
                    # Charger l'image avec OpenCV
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
                        target_size = (256, 256)
                        resized_img = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_LINEAR)

                        frame_id += 1
                        meta = FrameMeta(frame_id=frame_id, ts=time.time())
                        rf = RawFrame(image=resized_img, meta=meta)

                        # LOG.info(f"Image traitée : {image_path}, Frame ID : {frame_id}, Taille de l'image : {resized_img.shape}")
                        LOG.info(f"Image traitée : {image_path}, "
                                f"Frame ID : {rf.meta.frame_id}, "
                                f"Taille de l'image : {resized_img.shape}, "
                                f"Timestamp : {rf.meta.ts}, "
                                f"Pose valide : {rf.meta.pose.valid}, "
                                f"Spacing : {rf.meta.spacing}, "
                                f"Orientation : {rf.meta.orientation}, "
                                f"Device Name : {rf.meta.device_name}")
                        try:
                            mailbox.append(rf)
                        except Exception:
                            try:
                                from core.monitoring.kpi import increment_drops
                                increment_drops(image_id=f"frame_{frame_id}", name="rx.drop_total", delta=1, emit=True)
                            except Exception:
                                pass
                            LOG.exception("Échec d'ajout de la frame simulée dans la mailbox")
                    else:
                        print(f"[SIMULATION] Erreur de chargement de l'image {image_path}")

                    # Incrémenter l'index et arrêter si on a traité toutes les images
                    _image_index += 1
                    if _image_index >= len(_image_files):  # Si on a parcouru toutes les images
                        print("[SIMULATION] Toutes les images ont été traitées.")
                        break  # Arrêter la boucle une fois que toutes les images ont été traitées

                time.sleep(0.04)  # Simuler 25 FPS

        except Exception:
            LOG.exception("Exception dans la boucle principale de run_plus_client")  # log toute exception inattendue pendant la boucle
            time.sleep(0.1)  # pause courte avant de reprendre pour éviter une boucle d'erreur infinie

    # Nettoyage final après arrêt du thread RX
    try:
        if event_cb:  # si une fonction de callback d'événements est définie
            event_cb("rx_disconnect", {"host": host, "port": port})  # signale la déconnexion du client RX à la passerelle
    except Exception:
        LOG.exception("event_cb a échoué lors de la déconnexion")  # log l'erreur sans interrompre la fermeture

    LOG.info("[RX - SIM]run_plus_client arrêté")  # message d'information indiquant l'arrêt propre du thread RX