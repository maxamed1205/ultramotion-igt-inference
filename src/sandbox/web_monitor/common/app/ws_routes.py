"""
common/app/ws_routes.py
------------------------
Gestion des WebSockets du Web Monitor.
Diffusion périodique de métriques temps réel vers les clients connectés.
"""

import asyncio
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect, Query
from .utils import cached_snapshot
import time


def register_ws_routes(app):
    """Attache les routes WebSocket à l'application FastAPI."""
    log = logging.getLogger("igt.dashboard")
    connected_clients = set()

    async def broadcast_metrics(data):
        """Envoie les données JSON à tous les clients connectés."""
        for ws in list(connected_clients):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                connected_clients.remove(ws)

    @app.websocket("/ws/v1/test")
    async def ws_test(websocket: WebSocket):
        """WebSocket de test sans authentification."""
        log.info("[WS] WebSocket de test - connexion")
        await websocket.accept()
        await websocket.send_text(json.dumps({"status": "test ok"}))
        await websocket.close()

    @app.websocket("/ws/v1/metrics")
    async def ws_metrics(websocket: WebSocket):
        """
        Diffusion périodique des métriques GPU/CPU/FPS.
        Vérifie le token de sécurité (passé en paramètre d'URL).
        """
        log.info("[WS] Tentative de connexion WebSocket")

        # 1️⃣ Vérification du token (ex: ws://host/ws/v1/metrics?token=lab-igt-access)
        token = websocket.query_params.get("token")
        cfg = app.state.cfg
        expected = cfg["dashboard"]["security"]["token"]
        security_enabled = cfg["dashboard"]["security"].get("enabled", False)

        if security_enabled and token != expected:
            log.warning(f"[WS] Connexion refusée (token invalide)")
            await websocket.close(code=4003)
            return

        # 2️⃣ Connexion acceptée
        await websocket.accept()
        connected_clients.add(websocket)
        log.info(f"[WS] Client connecté ({len(connected_clients)} total)")

        # 3️⃣ Diffusion continue des métriques
        try:
            while connected_clients:
                try:
                    data = cached_snapshot(_mock_metrics, ttl=1.0)
                    await broadcast_metrics(data)
                    await asyncio.sleep(1.0)
                except Exception as e:
                    log.error(f"[WS] Erreur lors de la récupération des métriques : {e}")
                    raise

            log.info("[WS] Aucun client restant, arrêt de la diffusion.")
        except WebSocketDisconnect:
            connected_clients.discard(websocket)
            log.info(f"[WS] Client déconnecté ({len(connected_clients)} restants)")
        except Exception as e:
            log.error(f"[WS] Erreur: {e}")
            connected_clients.discard(websocket)



def _mock_metrics():
    """Mock de données système pour test local (sera remplacé par FusionCollector)."""
    return {
        "gpu": {"usage": 65.5, "vram_used": 1500, "temp": 56.1},
        "cpu": {"usage": 30.2, "threads": 22},
        "fps": {"rx": 25.0, "proc": 24.7, "tx": 24.9},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
