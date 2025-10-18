"""Compatibilité rétroactive : expose IGTGateway depuis le nouveau module gateway.manager.

Ce module maintient le chemin d'importation historique `service.igthelper.IGTGateway`
fonctionnel, alors que l'implémentation réelle a été déplacée vers `service.gateway.manager`.
"""

from service.gateway.manager import IGTGateway  # réimporte la classe IGTGateway depuis son nouvel emplacement (gateway.manager)

__all__ = ["IGTGateway"]  # définit explicitement les symboles exportés lors d’un import global (ex: from service.igthelper import *)
