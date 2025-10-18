"""
service.heartbeat
-----------------

Module léger pour mesurer la latence réseau via une connexion TCP courte.
Utilisé par le superviseur du gateway pour publier un KPI `latency_ms`.

Exemple :
>>> measure_latency("127.0.0.1", 18944)
2.3   # en millisecondes
"""

import socket  # fournit les primitives réseau bas niveau (création de connexions TCP)
import time    # permet de mesurer le temps écoulé entre deux instants
import logging # gestion centralisée des messages de log

LOG = logging.getLogger("igt.heartbeat")  # initialise un logger spécifique au module heartbeat


def measure_latency(host: str, port: int, timeout: float = 1.0) -> float:
    """Mesure la latence moyenne d'une connexion TCP courte.

    Args:
        host: Adresse du serveur (ex. PlusServer ou Slicer).
        port: Port du service cible.
        timeout: Délai maximum de tentative de connexion (secondes).

    Returns:
        Latence en millisecondes (float).
        Retourne -1.0 en cas d'échec ou de timeout.
    """
    start = time.time()  # enregistre le moment où la tentative de connexion commence
    try:
        # socket.create_connection crée et ouvre une connexion TCP vers (host, port)
        # la fonction bloque au maximum `timeout` secondes avant d'abandonner
        with socket.create_connection((host, port), timeout=timeout):
            pass  # la connexion réussit, on ne fait rien d'autre que la refermer aussitôt
        return (time.time() - start) * 1000.0  # calcule la durée écoulée (ms) entre début et fin de la connexion
    except Exception as e:  # capture toute exception (timeout, refus, erreur réseau, etc.)
        LOG.debug("Échec de la mesure de latence : %s", e)  # enregistre un message de debug
        return -1.0  # retourne -1.0 pour signaler que la mesure a échoué
