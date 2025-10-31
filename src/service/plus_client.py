"""Thread client Plus : reçoit les messages IGTLink de type IMAGE et les empile sous forme de RawFrame.

Cette implémentation privilégie l’usage de la bibliothèque `pyigtl` (client réel).
Si `pyigtl` n’est pas installée (cas typique en développement ou CI), 
une version simulée génère des tableaux numpy à environ 25 FPS.
L’API est volontairement simple pour que cette fonction puisse être directement 
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

LOG = logging.getLogger("igt.plus_client")  # création d’un logger spécifique pour ce module


def run_plus_client(mailbox, stop_event, host, port, stats_cb: Optional[Callable] = None, event_cb: Optional[Callable] = None) -> None:
    """Thread RX : client IGTLink vers PlusServer.

    - Reçoit des messages IMAGE (et plus tard TRANSFORM),
    - Empile les RawFrame dans la mailbox (file d’entrée de la passerelle),
    - Appelle stats_cb(fps, ts) toutes les 2 secondes si fourni,
    - Appelle event_cb('rx_connect', {...}) / ('rx_disconnect', {...}) pour notifier les connexions.
    """
    try:
        import pyigtl  # tente d’importer la vraie bibliothèque OpenIGTLink (communication réseau)
    except Exception:
        pyigtl = None  # si pyigtl est absent, on passe en mode simulation
        LOG.debug("pyigtl non disponible ; utilisation du simulateur pour plus_client")  # message de debug

    try:
        if event_cb:  # si une fonction callback d’événement est fournie
            event_cb("rx_connect", {"host": host, "port": port})  # signale la connexion RX au gestionnaire d’événements
    except Exception:
        LOG.exception("event_cb a échoué lors de la notification de connexion")  # log l’erreur sans interrompre le flux

    fps_window = []  # liste des timestamps des frames reçues (sert à calculer le FPS moyen)
    last_stats = time.time()  # enregistre le moment du dernier envoi de statistiques
    frame_id = 0  # compteur d’identifiants de frames (incrémenté à chaque nouvelle image)

    try:
        if pyigtl:  # si la bibliothèque pyigtl est disponible (mode réel)
            try:
                client = pyigtl.OpenIGTLinkClient(host, port)  # crée un client OpenIGTLink vers PlusServer (API hypothétique)
                client.connect()  # tente de se connecter au serveur IGTLink distant
            except Exception:
                LOG.exception("Échec de connexion du client pyigtl ; bascule vers le mode simulateur")  # message d’erreur
                pyigtl = None  # désactive pyigtl pour passer en mode simulation

    except Exception:
        pyigtl = None  # en cas d’erreur imprévue, on force le mode simulateur
    # Boucle principale de réception (tourne en continu tant que le thread n’est pas arrêté)
    while not stop_event.is_set():  # continue tant que le signal d’arrêt global n’a pas été activé
        start = time.time()  # enregistre l’heure de début du cycle (utile pour le calcul de FPS ou latence)
        try:
            print("[SIMULATION] Génération d'une frame simulée") 
            if pyigtl:  # si la bibliothèque pyigtl est disponible → mode réel (connexion à PlusServer)
                try:
                    msg = client.receive(timeout=0.1)  # tente de recevoir un message IGTLink (attente max 100 ms)
                except Exception:
                    msg = None  # en cas d’erreur, on ignore le message

                if msg is None:  # aucun message reçu pendant ce cycle
                    time.sleep(0.01)  # petite pause pour éviter de saturer le CPU
                    continue  # retourne au début de la boucle pour réessayer

                # Conversion du message IGTLink en tableau numpy (image brute)
                try:
                    arr = np.asarray(msg.image, dtype=np.uint8)  # tente de lire les données image du message
                except Exception:
                    raise RuntimeError("Erreur de lecture de l'image. L'exécution est arrêtée.")  # Si échec, lève une exception pour arrêter le code

                frame_id += 1  # incrémente le compteur global d’images reçues
                meta = FrameMeta(frame_id=frame_id, ts=time.time())  # crée les métadonnées associées (id + timestamp)
                rf = RawFrame(image=arr, meta=meta)  # encapsule l’image et ses métadonnées dans un objet RawFrame
                try:
                    mailbox.append(rf)  # place la frame dans la file d’entrée de la passerelle (AdaptiveDeque)
                except Exception:
                    try:
                        from core.monitoring.kpi import increment_drops
                        increment_drops(image_id=f"frame_{frame_id}", name="rx.drop_total", delta=1, emit=True)  # passer frame_id comme image_id
                    except Exception:
                        pass
                    LOG.exception("Échec d’ajout de la frame reçue dans la mailbox")  # log en cas d’erreur d’insertion

            else:  # sinon → mode simulation (aucune connexion réelle, on génère des images synthétiques)
                 # message de debug pour indiquer le mode simulation
                # time.sleep(0.04)  # attend environ 40 ms → simule une fréquence de 25 FPS
                DATASET_PATH = Path(r"C:\Users\maxam\Desktop\TM\ultramotion-igt-inference\Video_001")
                # Charger la liste des fichiers image
                image_files = sorted(glob(str(DATASET_PATH / "*.jpg")))  # Récupère tous les fichiers JPG dans le dossier

                # Simulation du traitement des images avec un délai de 40ms (pour simuler 25 FPS)
                for image_path in image_files:
                    # Charger l'image avec OpenCV
                    img = cv2.imread(str(image_path)) # l'image sera déjà en numpy array
                    # Vérifier si l'image a été correctement chargée
                    if img is None:
                        print(f"Erreur de chargement de l'image {image_path}")
                        continue  # passer à l'image suivante si erreur
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8') # Convertir l'image en niveaux de gris (grayscale) avec sécurité 8 bits si jamais il y a un problème de format
                    target_size = (256, 256) 
                    resized_img = cv2.resize(gray_img, target_size, interpolation=cv2.INTER_LINEAR) # Redimensionner l'image (cible de taille 256x256)

                    frame_id += 1  # incrémente le compteur d’images
                    meta = FrameMeta(frame_id=frame_id, ts=time.time())  # crée des métadonnées (id + timestamp)
                    rf = RawFrame(image=resized_img, meta=meta)  # encapsule les données dans un RawFrame simulé
                    try:
                        mailbox.append(rf)  # empile la frame simulée dans la file d’entrée
                    except Exception:
                        try:
                            from core.monitoring.kpi import increment_drops
                            increment_drops(image_id=f"frame_{frame_id}", name="rx.drop_total", delta=1, emit=True)  # passer frame_id comme image_id
                        except Exception:
                            pass
                        LOG.exception("Échec d’ajout de la frame simulée dans la mailbox")  # log en cas d’erreur

                    time.sleep(0.04)  # Simuler un délai pour atteindre environ 25 FPS attend 40 ms avant de traiter la prochaine image


            # # Enregistre le timestamp de réception de cette frame
            # now = time.time()  # capture l'instant exact où la frame a été reçue
            # fps_window.append(now)  # ajoute ce timestamp dans la liste des frames récentes (fenêtre temporelle glissante)

            # # Ne conserver que les 5 dernières secondes d'historique
            # cutoff = now - 5.0  # seuil : on veut garder uniquement les timestamps récents (< 5 s)
            # while fps_window and fps_window[0] < cutoff:  # tant qu'il existe des timestamps trop anciens
            #     fps_window.pop(0)  # on les retire de la liste pour maintenir une fenêtre de 5 secondes max

            # # Émettre les statistiques toutes les 2 secondes
            # if now - last_stats >= 2.0:  # si plus de 2 s se sont écoulées depuis le dernier envoi de stats
            #     # Calcul du FPS moyen sur les 5 dernières secondes
            #     fps = len(fps_window) / max(1.0, min(5.0, now - (fps_window[0] if fps_window else now)))  # nombre de frames / durée écoulée
            #     try:
            #         # Calcule le nombre d’octets reçus pour la dernière frame (si l’attribut "nbytes" existe)
            #         bytes_count = getattr(arr, "nbytes", 0)  # récupération sécurisée de la taille mémoire du tableau numpy
            #         if stats_cb:  # si une fonction de callback pour les statistiques est fournie
            #             stats_cb(fps, now, bytes_count)  # appelle la fonction pour reporter les statistiques (fps, timestamp, taille)
            #     except Exception:
            #         LOG.exception("stats_cb a échoué dans plus_client")  # journalise toute erreur dans la mise à jour des stats

            #     last_stats = now  # met à jour le moment du dernier envoi de statistiques

        except Exception:
            LOG.exception("Exception dans la boucle principale de run_plus_client")  # log toute exception inattendue pendant la boucle
            time.sleep(0.1)  # pause courte avant de reprendre pour éviter une boucle d’erreur infinie

        # Nettoyage final après arrêt du thread RX
        try:
            if event_cb:  # si une fonction de callback d’événements est définie
                event_cb("rx_disconnect", {"host": host, "port": port})  # signale la déconnexion du client RX à la passerelle
        except Exception:
            LOG.exception("event_cb a échoué lors de la déconnexion")  # log l’erreur sans interrompre la fermeture

        LOG.info("run_plus_client arrêté")  # message d’information indiquant l’arrêt propre du thread RX
