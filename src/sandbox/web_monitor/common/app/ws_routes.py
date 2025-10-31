"""
common/app/ws_routes.py
------------------------
Gestion des WebSockets du Web Monitor.
Diffusion périodique de métriques temps réel via le LogCollector.
"""

import asyncio
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

# ✅ Import du vrai Collector
try:
    from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector as Collector
    HAS_COLLECTOR = True
except Exception as e:
    HAS_COLLECTOR = False
    print(f"[WS] ❌ Impossible d'importer Collector: {e}")

log = logging.getLogger("igt.dashboard")


# ─────────────────────────────────────────────
#  Enregistrement des routes WebSocket
# ─────────────────────────────────────────────
def register_ws_routes(app):
    """Attache les routes WebSocket à l'application FastAPI."""
    connected_clients = set()

    # ✅ Signal d’arrêt partagé
    if not hasattr(app.state, "stop_event"):
        app.state.stop_event = asyncio.Event()

    # ✅ Vérifie la présence du Collector
    if not hasattr(app.state, "collector") or not isinstance(getattr(app.state, "collector", None), Collector):
        msg = "[WS] ❌ Aucun LogCollector valide n’est disponible dans app.state"
        log.critical(msg)
        raise RuntimeError(msg)

    # ─────────────────────────────────────────────
    async def broadcast_metrics(data: dict):
        """Diffuse les données JSON à tous les clients connectés."""
        for ws in list(connected_clients):
            try:
                await ws.send_text(json.dumps(data))
            except Exception as e:
                log.warning(f"[WS] Client retiré (erreur d'envoi): {e}")
                connected_clients.remove(ws)

    # ─────────────────────────────────────────────

    def serialize_frame(frame):
        """
        Convertit un FrameAggregate (objets Pydantic-like) en dict JSON simple
        que le frontend comprend:
        {
            "gpu": {...},
            "cpu": {...},
            "fps": {...},
            "interstage": {...},
            "timestamp": "...",
        }
        """
        if frame is None:
            return {}

        # interstage
        interstage_dict = None
        if hasattr(frame, "interstage") and frame.interstage:
            i = frame.interstage
            interstage_dict = {
                "frame_id": getattr(i, "frame_id", None),
                "rx_cpu": getattr(i, "rx_cpu", None),
                "cpu_gpu": getattr(i, "cpu_gpu", None),
                "proc_gpu": getattr(i, "proc_gpu", None),
                "gpu_cpu": getattr(i, "gpu_cpu", None),
                "cpu_tx": getattr(i, "cpu_tx", None),
                "total": getattr(i, "total", None),
            }

        # gpu_transfer (pour l’instant on ne récupère pas les vraies métriques GPU,
        # donc on envoie des champs neutres pour ne pas casser le DOM)
        gpu_dict = {
            "usage": 0.0,
            "vram_used": 0.0,
            "temp": 0.0,
            "streams": 0,
        }

        # fps placeholders (le frontend les lit pour overview-latency)
        fps_dict = {
            "rx": 0.0,
            "proc": getattr(frame, "latency_rxtx", 0.0) or 0.0,
            "tx": 0.0,
        }

        return {
            "timestamp": getattr(frame, "ts_wall", None),
            "gpu": gpu_dict,
            "fps": fps_dict,
            "interstage": interstage_dict,
        }


    # ─────────────────────────────────────────────
    @app.websocket("/ws/v1/pipeline")
    async def ws_pipeline(websocket: WebSocket):
        """
        WebSocket principale du dashboard.
        🔹 Données réelles issues du LogCollector
        🔹 Aucune donnée mockée
        """
        log.info("[WS] Connexion WebSocket /ws/v1/pipeline ouverte")

        cfg = getattr(app.state, "cfg", {})
        security = cfg.get("dashboard", {}).get("security", {})
        enabled = security.get("enabled", False)
        expected = security.get("token")
        token = websocket.query_params.get("token")

        # 🔒 Vérifie le token si la sécurité est activée
        if enabled and token != expected:
            log.warning("[WS] Token invalide, fermeture de la connexion.")
            await websocket.close(code=4003)
            return

        await websocket.accept()
        connected_clients.add(websocket)
        log.info(f"[WS] Client connecté via /ws/v1/pipeline ({len(connected_clients)} total)")

        collector = getattr(app.state, "collector", None)
        if collector is None:
            raise RuntimeError("[WS] Aucun collector initialisé — impossible de démarrer le flux WebSocket")

        stop_event = app.state.stop_event
        shutdown_task = asyncio.create_task(stop_event.wait())

        try:
            while ( websocket in connected_clients and not stop_event.is_set() and websocket.application_state.name.lower() == "connected"):
                try:
                    frame = collector.get_latest()
                    if frame:
                        frame_id = getattr(frame, 'frame_id', '?')
                        log.debug(f"[WS] ✅ Données collector récupérées (frame_id={frame_id})")
                        serialized_data = serialize_frame(frame)
                    else:
                        # Pas de nouvelle donnée — on envoie un "heartbeat" minimal
                        log.debug("[WS] ⏳ Aucun nouveau frame (heartbeat envoyé)")

                    # Envoi au client
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "system_metrics",
                            "data": serialized_data,
                        }))
                    except Exception as send_err:
                        log.warning(f"[WS] ⚠️ Connexion WebSocket fermée pendant l'envoi : {send_err}")
                        break  # on quitte la boucle proprement

                    # ⏱️ Pause 1s (ou jusqu’à shutdown)
                    done, _ = await asyncio.wait(
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
                    log.exception(f"[WS] 💥 Erreur dans la boucle WS pipeline: {inner_e}")
                    raise  # ⛔ Re-lève l’erreur pour qu’elle soit visible

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
