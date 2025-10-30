"""
common/app/ws_routes.py
------------------------
Gestion des WebSockets du Web Monitor.
Diffusion périodique de métriques temps réel vers les clients connectés.
"""

import asyncio
import json
import logging
import time
from fastapi import WebSocket, WebSocketDisconnect

# Pas besoin de cached_snapshot si non défini
# from .utils import cached_snapshot

log = logging.getLogger("igt.dashboard")


def register_ws_routes(app):
    """Attache les routes WebSocket à l'application FastAPI."""
    connected_clients = set()

    # ─────────────────────────────────────────────
    async def broadcast_metrics(data):
        """Envoie les données JSON à tous les clients connectés."""
        for ws in list(connected_clients):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                connected_clients.remove(ws)

    # ─────────────────────────────────────────────
    @app.websocket("/ws/v1/metrics")
    async def ws_metrics(websocket: WebSocket):
        """WebSocket principale du dashboard — envoie des métriques système simulées."""
        log.info("[WS] Tentative de connexion WebSocket /ws/v1/metrics")

        cfg = app.state.cfg
        security = cfg.get("dashboard", {}).get("security", {})
        enabled = security.get("enabled", False)
        expected = security.get("token")

        token = websocket.query_params.get("token")

        # Sécurité optionnelle
        if enabled and token != expected:
            log.warning("[WS] Token invalide, fermeture de la connexion.")
            await websocket.close(code=4003)
            return

        await websocket.accept()
        connected_clients.add(websocket)
        log.info(f"[WS] Client connecté ({len(connected_clients)} total)")

        try:
            while websocket in connected_clients:
                # Envoie périodique de données simulées
                data = _mock_metrics()
                await websocket.send_text(json.dumps({"type": "system_metrics", "data": data}))
                await asyncio.sleep(1.0)

        except WebSocketDisconnect:
            log.info("[WS] Client déconnecté")
        except Exception as e:
            log.error(f"[WS] Erreur WebSocket : {e}")
        finally:
            connected_clients.discard(websocket)
            log.info(f"[WS] Client supprimé ({len(connected_clients)} restants)")

    # ─────────────────────────────────────────────
    @app.websocket("/ws/v1/pipeline")
    async def ws_pipeline(websocket: WebSocket):
        """
        Diffusion en temps réel des frames agrégées du LogCollector.
        """
        log.info("[WS] WebSocket /ws/v1/pipeline : connexion ouverte")
        await websocket.accept()

        # On récupère l'instance du collector via app.state
        collector = getattr(app.state, "collector", None)
        if collector is None:
            log.error("[WS] Aucun collector attaché à app.state.collector")
            await websocket.send_json({"error": "collector_not_initialized"})
            await websocket.close()
            return

        try:
            while True:
                # On récupère un snapshot complet à envoyer au dashboard
                snap = collector.as_snapshot()
                await websocket.send_text(json.dumps(snap.to_dict()))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            log.info("[WS] Client pipeline déconnecté")
        finally:
            await websocket.close()


# ─────────────────────────────────────────────
#  Données simulées pour tests locaux
# ─────────────────────────────────────────────

def _mock_metrics():
    """Mock de données système pour test local (sera remplacé par FusionCollector)."""
    return {
        "gpu": {"usage": 65.5, "vram_used": 1500, "temp": 56.1},
        "cpu": {"usage": 30.2, "threads": 22},
        "fps": {"rx": 25.0, "proc": 24.7, "tx": 24.9},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
