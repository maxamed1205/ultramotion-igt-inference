"""
service.registry
----------------

Registry statique des fonctions d’entrée pour les threads RX/TX du gateway.

Ce module centralise les références vers les points d’entrée concrets
(PlusServer client, Slicer server, etc.) afin de supprimer les imports
dynamiques dans `gateway.manager`.

Il est chargé une seule fois au démarrage de l’application et permet
de remplacer facilement les implémentations (mock, test, etc.).
"""

from service.plus_client import run_plus_client
from service.slicer_server import run_slicer_server

THREAD_REGISTRY = {
    "rx": run_plus_client,     # clé "rx" associée à la fonction run_plus_client — point d’entrée du thread de réception (client PlusServer)
    "tx": run_slicer_server,   # clé "tx" associée à la fonction run_slicer_server — point d’entrée du thread d’envoi (serveur Slicer)
}

__all__ = ["THREAD_REGISTRY"]  # définit explicitement les symboles exportés lors d’un import * ; ici seul THREAD_REGISTRY est rendu accessible publiquement

