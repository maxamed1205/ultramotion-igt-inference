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
    from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector as Collector
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

    # ✅ Signal d’arrêt partagé (placé dans app.state)
    if not hasattr(app.state, "stop_event"):
        app.state.stop_event = asyncio.Event()

    # Vérification Collector disponible
    if not hasattr(app.state, "collector") or not isinstance(getattr(app.state, "collector", None), Collector):
        log.warning("[WS] Collector non disponible — fallback mock activé.")
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

        # ✅ Tâche parallèle qui attend le shutdown
        stop_event = app.state.stop_event
        shutdown_task = asyncio.create_task(stop_event.wait())

        try:
            while (
                websocket in connected_clients
                and not stop_event.is_set()
                and websocket.application_state.name.lower() == "connected"
            ):
                # Attente simultanée : données ou stop_event
                try:
                    collector = getattr(app.state, "collector", None)
                    data = None

                    if collector is None:
                        log.warning("[WS] ❌ Aucun collector — envoi de mock temporaire")
                        data = _mock_metrics()
                    else:
                        try:
                            data = collector.get_latest()
                        except Exception as e:
                            log.error(f"[WS] 💥 Erreur collector.get_latest(): {e}")
                            data = None

                    if not data:
                        # Pas de donnée → petit mock pour éviter silence total
                        log.debug("[WS] ⚠️ Aucune donnée collector, envoi mock")
                        data = _mock_metrics()

                    message = {"type": "system_metrics", "data": data}
                    await websocket.send_text(json.dumps(message))

                    # 🕒 Attente surveillée — stop instantané si shutdown détecté
                    done, pending = await asyncio.wait(
                        {shutdown_task},
                        timeout=1.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stop_event.is_set():
                        log.info("[WS] 🔔 Shutdown détecté — sortie immédiate de la boucle WS")
                        break

                except WebSocketDisconnect:
                    log.info("[WS] Client pipeline déconnecté")
                    break
                except asyncio.CancelledError:
                    log.info("[WS] ✅ Boucle WS annulée proprement (CancelledError)")
                    break
                except Exception as inner_e:
                    log.error(f"[WS] 💥 Erreur boucle WS pipeline: {inner_e}")
                    break

        finally:
            # Nettoyage propre
            if websocket in connected_clients:
                connected_clients.discard(websocket)
            try:
                await websocket.close()
            except Exception:
                pass
            shutdown_task.cancel()
            log.info(f"[WS] Client supprimé ({len(connected_clients)} restants)")

    # ─────────────────────────────────────────────
    @app.websocket("/ws")
    async def ws_legacy(websocket: WebSocket):
        """Ancien endpoint — redirige vers pipeline (mock permanent)."""
        log.warning("[WS] Client connecté via /ws (legacy)")
        await websocket.accept()
        connected_clients.add(websocket)
        stop_event = app.state.stop_event
        try:
            while websocket in connected_clients and not stop_event.is_set():
                data = _mock_metrics()
                await websocket.send_text(json.dumps({"type": "system_metrics", "data": data}))
                done, _ = await asyncio.wait(
                    {asyncio.create_task(stop_event.wait())},
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_event.is_set():
                    break
        except WebSocketDisconnect:
            log.info("[WS] Client legacy déconnecté")
        finally:
            connected_clients.discard(websocket)
            try:
                await websocket.close()
            except Exception:
                pass

    # ─────────────────────────────────────────────
    @app.websocket("/ws/v1/metrics")
    async def ws_metrics(websocket: WebSocket):
        """Compatibilité (non utilisée actuellement)."""
        log.info("[WS] Connexion WebSocket /ws/v1/metrics (mockées)")
        await websocket.accept()
        connected_clients.add(websocket)
        stop_event = app.state.stop_event
        try:
            while websocket in connected_clients and not stop_event.is_set():
                data = _mock_metrics()
                await websocket.send_text(json.dumps({"type": "system_metrics", "data": data}))
                done, _ = await asyncio.wait(
                    {asyncio.create_task(stop_event.wait())},
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_event.is_set():
                    break
        except WebSocketDisconnect:
            log.info("[WS] Client metrics déconnecté")
        finally:
            connected_clients.discard(websocket)
            try:
                await websocket.close()
            except Exception:
                pass


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
            "total": total,
        },
    }

    log.debug(f"[WS] Données mockées générées: {result}")
    return result
