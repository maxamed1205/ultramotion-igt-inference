"""
core/acquisition/receiver.py
───────────────────────────────────────────────────────────────
📖 Contexte et rôle du module

Ce module définit la boucle d’acquisition locale (ReceiverThread)
et la fonction de démarrage associée (start_receiver_thread).

💡 Cependant, dans la version actuelle du projet (`main.py`), ce fichier
n’est **pas utilisé directement** : la réception des images est déjà
prise en charge par le `IGTGateway` (src/service/gateway/manager.py),
qui gère lui-même les threads RX/TX et la supervision.

🧩 Architecture actuelle (production / main.py)
───────────────────────────────────────────────
main.py
 ├── Configure le logging (async, KPI)
 ├── Charge la config (GatewayConfig)
 ├── Crée et démarre `IGTGateway`
 │     ↳ _rx_thread → reçoit depuis PlusServer
 │     ↳ _tx_thread → envoie vers 3D Slicer
 │     ↳ _supervisor_thread → surveille FPS, latence, drops
 ├── Lance `start_monitor_thread` (KPI globaux)
 └── Boucle infinie (service vivant)

➡️ Dans ce mode, `ReceiverThread` n’est pas nécessaire :
`IGTGateway.start()` inclut déjà la réception réseau complète.

🧩 Usage alternatif (mode “standalone” ou test local)
───────────────────────────────────────────────────────────────
`receiver.py` reste utile comme outil modulaire pour :
  - lancer un thread local d’acquisition,
  - consommer les frames depuis un `IGTGateway` réel ou simulé,
  - et alimenter une file temps réel (Queue_RT_dyn) pour la
    segmentation IA / traitement local.

Ce mode est privilégié pour :
  • les tests unitaires (hors réseau),
  • le développement de la pipeline IA indépendante,
  • ou les environnements de simulation (`StubGateway`).

🧱 Schéma simplifié
───────────────────────────────────────────────────────────────
A. Mode production (main.py)
   PlusServer → IGTGateway._rx_thread → _mailbox → TX → 3D Slicer

B. Mode standalone (receiver.py)
   start_receiver_thread()
       ↓
   IGTGateway.start()       ← threads réseau RX/TX
       ↓
   ReceiverThread.run()     ← lit gateway.receive_image()
       ↓
   Queue_RT_dyn → pipeline IA → gateway.send_mask()

🧠 TL;DR
───────────────────────────────────────────────────────────────
- `receiver.py` n’est pas utilisé dans `main.py` actuel (Gateway complet)
- Il reste essentiel pour les tests, le dev local et les pipelines IA isolées
- Il sert à piloter l’acquisition, mesurer les KPI et alimenter la queue RT
"""

# TODO [Phase 5 - Intégration du système de logs]
# - Ajouter le logger KPI pour les métriques d’acquisition (fps, latence, pertes)
# - Publier les KPI dans logs/kpi.log via igt.kpi
# - Conserver des logs contextuels (frame_id, timestamp, taille de la file)
# - Ajuster les niveaux INFO/WARNING/ERROR selon la gravité
# - Rendre compatible avec PerfFilter (désactivation propre en mode performance)

"""Module Receiver — Thread A

Rôle : recevoir les images et poses via IGTLink, puis les insérer dans
Queue_Raw et Queue_RT_dyn.

Contient les signatures et la boucle principale du thread d’acquisition.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import time
import logging
import threading

from typing import Tuple as _Tuple

from core.types import FrameMeta, RawFrame
from core.queues.buffers import get_queue_rt_dyn, enqueue_nowait_rt, apply_rt_backpressure
from core.monitoring.monitor import log_kpi_tick

LOG = logging.getLogger("igt.receiver")  # logger principal du module d’acquisition
KPI = logging.getLogger("igt.kpi")  # logger KPI (flux métriques vers kpi.log)

class ReceiverThread(threading.Thread):
    """Thread A — Acquisition continue (PlusServer → Queue_RT_dyn).

    Récupère les images via IGTGateway.receive_image() et les insère dans
    la file `Queue_RT_dyn` via `enqueue_to_rt()`.

    Note : la boucle utilise un court time.sleep pour rester économe en CPU
    lorsque la source n’envoie rien. Quand pyigtl sera intégré, on remplacera
    ce polling par un mécanisme bloquant (select/epoll) pour éviter les boucles actives.
    """

    def __init__(self, gateway: object, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)  # initialise le thread en mode démon (se termine avec le programme principal)
        self.gateway = gateway  # référence vers l’objet passerelle (IGTGateway ou StubGateway)
        self.stop_event = stop_event  # signal d’arrêt partagé entre threads

    def run(self) -> None:
        LOG.info("ReceiverThread démarré.")  # message d’information au démarrage
        # Tout le thread est protégé par deux couches de try/except , Analogie simple Imagine un opérateur dans une chaîne de production : S’il rate une pièce → il la jette, mais continue à bosser (try interne).
        # Les KPI agrégés sont gérés de façon centralisée par core.monitoring.monitor

        while not self.stop_event.is_set():  # boucle principale tant que l’arrêt n’est pas demandé
            try:
                frame = None  # initialisation du tampon de frame
                recv_start = time.time()  # horodatage du début de la réception (pour mesurer la latence réseau)

                try:
                    frame = self.gateway.receive_image()  # tente de récupérer une image depuis PlusServer
                except Exception as e:
                    LOG.error("Erreur IGTGateway.receive_image : %r", e)  # erreur non fatale lors de la réception
                    frame = None  # on ignore la frame et continue la boucle

                t0 = time.time()  # instant après réception et construction de la frame

                if frame:  # si une image a bien été reçue
                    try:
                        enqueue_latency_ms = 0.0  # initialisation de la latence d’enfilement
                        enqueue_start = time.time()  # mesure du temps d’insertion en file

                        q = get_queue_rt_dyn()  # récupère la file temps réel dynamique
                        if not enqueue_nowait_rt(q, frame):  # essaie d’insérer sans blocage
                            apply_rt_backpressure(q, now=time.time(), max_lag_ms=500)  # applique la politique de backpressure (supprime les frames trop anciennes)
                            enqueue_nowait_rt(q, frame)  # retente une fois après nettoyage

                        enqueue_latency_ms = (time.time() - enqueue_start) * 1000.0  # calcule la durée d’insertion (ms)

                        try: # récupère la taille actuelle de la file (pour log de contexte)
                            qsize = q.qsize() if not isinstance(q, list) else len(q)
                        except Exception:
                            qsize = -1  # valeur par défaut si la taille est inaccessible

                        if LOG.isEnabledFor(logging.DEBUG):  # en mode debug, affiche les infos détaillées
                            LOG.debug(
                                "Frame insérée (id=%s ts=%.3f queue=%d)",
                                getattr(frame.meta, "frame_id", -1),
                                getattr(frame.meta, "ts", -1.0),
                                qsize,
                            )

                        # Envoi d’un KPI pour la frame : latence totale mesurée
                        proc_latency_ms = (t0 - recv_start) * 1000.0  # latence de traitement (réception → frame construite)
                        total_latency_ms = (time.time() - recv_start) * 1000.0  # latence complète (réception → enfilée)
                        total_proc_ms = proc_latency_ms + enqueue_latency_ms  # latence combinée
                        try:
                            log_kpi_tick(0.0, 0.0, total_proc_ms, gpu_util=0.0)  # enregistre une mesure KPI unitaire (fps placeholders = 0)
                        except Exception:
                            if LOG.isEnabledFor(logging.DEBUG):
                                LOG.debug("Échec log_kpi_tick pour frame_id=%s", getattr(frame.meta, "frame_id", -1))
                    except Exception as e:
                        LOG.debug("Erreur ReceiverThread lors de l’enfilage : %r", e)  # échec lors de l’envoi dans la file (non critique)
            except Exception as e:
                LOG.debug("Erreur inattendue ReceiverThread : %r", e)  # capture toute autre erreur pour éviter d’interrompre la boucle
            # ⚠️ TODO 💡 L’alternative : modèle non-polling avec select.select()
            # L’idée : plutôt que de se réveiller sans raison, on dort jusqu’à ce que la socket dise “j’ai des données prêtes”.
            time.sleep(0.001)  # courte pause pour éviter de saturer le CPU (sera remplacée par select plus tard)

        LOG.info("ReceiverThread arrêté.")  # log propre à la sortie de boucle



def start_receiver_thread(config: Dict) -> None:
    """Démarre le thread d’acquisition.

    Args:
        config: dictionnaire de configuration (ports, hôtes, etc.)
    """
    stop_event = threading.Event()  # crée un signal d’arrêt partagé

    plus_host = config.get("plus_host", "127.0.0.1")  # hôte du serveur PlusServer
    plus_port = int(config.get("plus_port", 18944))  # port d’écoute PlusServer
    slicer_port = int(config.get("slicer_port", 18945))  # port d’envoi vers Slicer

    try:
        from service.igthelper import IGTGateway  # import standard
    except Exception:
        try:
            from igthelper import IGTGateway  # fallback local
        except Exception:
            IGTGateway = None  # si tout échoue, on utilisera le stub

    if IGTGateway is not None:  # si la vraie passerelle est dispo
        gateway = IGTGateway(plus_host, plus_port, slicer_port)
        try:
            gateway.start()  # démarre la connexion IGTLink
        except Exception:
            LOG.debug("IGTGateway.start() a échoué ou est un stub")  # tolère les environnements de test
    else:
        from simulation.mock_gateway import StubGateway  # passerelle simulée pour les tests
        gateway = StubGateway()

    receiver = ReceiverThread(gateway, stop_event)  # crée le thread de réception
    receiver.start()  # lance le thread
    LOG.info("Thread Receiver démarré (PlusServer=%s:%d)", plus_host, plus_port)
    return receiver, stop_event  # renvoie le thread et le signal pour contrôle ultérieur


# _igt_callback déplacé dans core.acquisition.decode.decode_igt_image


def stop_receiver_thread() -> None:
    """Arrête proprement le thread d’acquisition (ancienne interface non utilisée)."""
    raise NotImplementedError(
        "Utiliser stop_receiver_thread_ex(receiver, stop_event) avec les objets renvoyés par start_receiver_thread"
    )


def stop_receiver_thread_ex(receiver: threading.Thread, stop_event: threading.Event, timeout: float = 0.1) -> None:
    """Arrête proprement une instance existante de ReceiverThread.

    Args:
        receiver: objet thread renvoyé par start_receiver_thread
        stop_event: événement d’arrêt associé
        timeout: délai maximal (en secondes) pour l’attente du join
    """
    try:
        stop_event.set()  # envoie le signal d’arrêt
        receiver.join(timeout=timeout)  # attend la fin du thread
        LOG.info("Thread Receiver arrêté proprement.")
    except Exception as e:
        LOG.debug("Erreur à l’arrêt du thread Receiver : %r", e)


