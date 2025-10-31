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
import random
from fastapi import WebSocket, WebSocketDisconnect

# ✅ Import du vrai Collector
try:
    from sandbox.web_monitor.common.collector.log_collector.collector import Collector
    HAS_COLLECTOR = True
except Exception as e:
    HAS_COLLECTOR = False
    print(f"[WARN] Impossible d'importer Collector: {e}")

log = logging.getLogger("igt.dashboard")


# ─────────────────────────────────────────────
#  Enregistrement des routes WebSocket
# ─────────────────────────────────────────────
def register_ws_routes(app):
    """Attache les routes WebSocket à l'application FastAPI."""
    connected_clients = set()

    # ✅ Signal d’arrêt propre partagé (placé dans app.state)
    if not hasattr(app.state, "stop_event"):
        app.state.stop_event = asyncio.Event()

    # Instance globale du Collector (si disponible)
    if HAS_COLLECTOR:
        if not hasattr(app.state, "collector") or not isinstance(app.state.collector, Collector):
            try:
                app.state.collector = Collector()
                log.info("[Collector] Instance Collector initialisée depuis ws_routes.py")
            except Exception as e:
                log.error(f"[Collector] Erreur d'initialisation : {e}")
                app.state.collector = None
    else:
        app.state.collector = None

    # ─────────────────────────────────────────────
    async def broadcast_metrics(data):
        """Envoie les données JSON à tous les clients connectés."""
        for ws in list(connected_clients):
            try:
                await ws.send_text(json.dumps(data))
            except Exception as e:
                log.warning(f"[WS] Client retiré (erreur d'envoi): {e}")
                connected_clients.remove(ws)

    # ─────────────────────────────────────────────
    @app.websocket("/ws/v1/pipeline")
    async def ws_pipeline(websocket: WebSocket):
        """
        WebSocket principale du dashboard.
        Si Collector dispo → données réelles.
        Sinon → fallback mocké (_mock_metrics()).
        """
        log.info("[WS] Connexion WebSocket /ws/v1/pipeline ouverte")

        cfg = getattr(app.state, "cfg", {})
        security = cfg.get("dashboard", {}).get("security", {})
        enabled = security.get("enabled", False)
        expected = security.get("token")

        token = websocket.query_params.get("token")

        # 🔒 Sécurité optionnelle
        if enabled and token != expected:
            log.warning("[WS] Token invalide, fermeture de la connexion.")
            await websocket.close(code=4003)
            return

        await websocket.accept()
        connected_clients.add(websocket)
        log.info(f"[WS] Client connecté via /ws/v1/pipeline ({len(connected_clients)} total)")
        try:
            while websocket in connected_clients and not app.state.stop_event.is_set():
                # ✅ Stop immédiat si le websocket est déjà fermé
                if websocket.application_state.name.lower() != "connected":
                    log.info("[WS] Connexion fermée détectée — arrêt de la boucle d'envoi.")
                    break

                try:
                    collector = getattr(app.state, "collector", None)
                    data = None

                    if collector is not None:
                        try:
                            data = collector.get_latest_metrics()
                            if data:
                                log.debug(f"[Collector] Données réelles récupérées: {data}")
                            else:
                                log.debug("[Collector] Aucune donnée disponible pour le moment")
                        except Exception as e:
                            log.error(f"[Collector] Erreur lors de get_latest_metrics(): {e}")

                    if not data:
                        data = _mock_metrics()
                        log.debug("[WS] Fallback vers données mockées")

                    message = {"type": "system_metrics", "data": data}

                    # ✅ Envoi protégé (ne pas lever d'erreur si fermeture en cours)
                    try:
                        await websocket.send_text(json.dumps(message))
                    except (RuntimeError, ConnectionError):
                        log.info("[WS] Fermeture WS détectée pendant l'envoi — arrêt propre.")
                        break
                    except Exception as send_error:
                        log.warning(f"[WS] Échec envoi métriques: {send_error}")
                        break

                except Exception as inner_e:
                    log.error(f"[WS] Erreur boucle WS pipeline: {inner_e}")
                    break

                await asyncio.sleep(1.0)


        except WebSocketDisconnect:
            log.info("[WS] Client pipeline déconnecté")
        except Exception as e:
            log.error(f"[WS] Erreur WebSocket pipeline: {e}")
        finally:
            connected_clients.discard(websocket)
            await websocket.close()
            log.info(f"[WS] Client supprimé ({len(connected_clients)} restants)")

    # ─────────────────────────────────────────────
    # Route de compatibilité ancienne version
    @app.websocket("/ws")
    async def ws_legacy(websocket: WebSocket):
        """Ancien endpoint — redirige vers pipeline."""
        log.warning("[WS] Client connecté via /ws (legacy)")
        await websocket.accept()
        try:
            while websocket in connected_clients and not app.state.stop_event.is_set():
                data = _mock_metrics()
                await websocket.send_text(json.dumps({"type": "system_metrics", "data": data}))
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            log.info("[WS] Client déconnecté (legacy)")
        finally:
            connected_clients.discard(websocket)

    # ─────────────────────────────────────────────
    @app.websocket("/ws/v1/metrics")
    async def ws_metrics(websocket: WebSocket):
        """Compatibilité (non utilisée actuellement)."""
        log.info("[WS] Connexion WebSocket /ws/v1/metrics (mockées)")
        await websocket.accept()
        connected_clients.add(websocket)
        try:
            while websocket in connected_clients and not app.state.stop_event.is_set():
                data = _mock_metrics()
                await websocket.send_text(json.dumps({"type": "system_metrics", "data": data}))
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            log.info("[WS] Client metrics déconnecté")
        finally:
            connected_clients.discard(websocket)
            await websocket.close()


# ─────────────────────────────────────────────
#  Fallback mock pour tests
# ─────────────────────────────────────────────
def _mock_metrics():
    """Mock de données système pour test local (sera remplacé par Collector réel)."""
    rx_cpu = round(random.uniform(1.5, 2.5), 2)
    cpu_gpu = round(random.uniform(10.0, 12.0), 2)
    proc_gpu = round(random.uniform(1.0, 1.5), 2)
    gpu_cpu = round(random.uniform(0.8, 1.2), 2)
    cpu_tx = round(random.uniform(0.5, 1.0), 2)
    total = round(rx_cpu + cpu_gpu + proc_gpu + gpu_cpu + cpu_tx, 2)

    result = {
        "gpu": {"usage": 65.5, "vram_used": 1500, "temp": 56.1, "streams": 4},
        "cpu": {"usage": 30.2, "threads": 22},
        "fps": {"rx": 25.0, "proc": 24.7, "tx": 24.9},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interstage": {
            "frame_id": random.randint(1, 999),
            "rx_cpu": rx_cpu,
            "cpu_gpu": cpu_gpu,
            "proc_gpu": proc_gpu,
            "gpu_cpu": gpu_cpu,
            "cpu_tx": cpu_tx,
            "total": total
        },
    }

    log.debug(f"[WS] Données mockées générées: {result}")
    return result
