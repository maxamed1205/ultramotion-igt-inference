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

    @app.websocket("/ws")
    async def ws_main(websocket: WebSocket):
        """WebSocket principal pour le dashboard - compatible avec le JavaScript."""
        log.info("[WS] Connexion WebSocket principale")
        
        # Vérification de sécurité
        cfg = app.state.cfg
        security_enabled = cfg.get("dashboard", {}).get("security", {}).get("enabled", False)
        
        if security_enabled:
            token = websocket.query_params.get("token")
            expected = cfg["dashboard"]["security"]["token"]
            if token != expected:
                log.warning(f"[WS] Connexion refusée (token invalide)")
                await websocket.close(code=4003)
                return
        
        # Connexion acceptée
        await websocket.accept()
        connected_clients.add(websocket)
        log.info(f"[WS] Client connecté ({len(connected_clients)} total)")
        
        # Diffusion continue des métriques
        try:
            while websocket in connected_clients:
                try:
                    # Données simulées compatibles avec notre dashboard
                    data = {
                        "type": "gpu_metrics",
                        "data": {
                            "utilization": 73.2,
                            "memory_used": 1456.5,
                            "memory_reserved": 2048.0,
                            "device": "NVIDIA RTX 4080",
                            "driver": "535.98 / CUDA 12.2",
                            "streams": 4,
                            "frames": 1547,
                            "avg_latency": 15.3,
                            "throughput": 24.8,
                            "breakdown": {"norm": 156, "pin": 89, "copy": 203}
                        }
                    }
                    await websocket.send_text(json.dumps(data))
                    
                    # Données pipeline
                    pipeline_data = {
                        "type": "pipeline_metrics",
                        "data": {
                            "rx_to_gpu": 3.2,
                            "gpu_to_proc": 12.1,
                            "proc_to_cpu": 8.7,
                            "cpu_to_tx": 2.8,
                            "total_latency": 26.8,
                            "samples": 1547
                        }
                    }
                    await websocket.send_text(json.dumps(pipeline_data))
                    
                    # Données queues
                    queue_data = {
                        "type": "queue_metrics",
                        "data": {
                            "queue_rt": 2,
                            "queue_gpu": 1,
                            "drops": 0
                        }
                    }
                    await websocket.send_text(json.dumps(queue_data))
                    
                    # Données système
                    system_data = {
                        "type": "system_health",
                        "data": {
                            "status": "operational"
                        }
                    }
                    await websocket.send_text(json.dumps(system_data))
                    
                    await asyncio.sleep(1.0)
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    log.error(f"[WS] Erreur lors de l'envoi des métriques : {e}")
                    break
                    
        except WebSocketDisconnect:
            log.info(f"[WS] Client déconnecté")
        except Exception as e:
            log.error(f"[WS] Erreur WebSocket: {e}")
        finally:
            connected_clients.discard(websocket)
            log.info(f"[WS] Client supprimé ({len(connected_clients)} restants)")

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
