"""
Dashboard Web Temps Réel pour Ultramotion IGT Inference
=========================================================

Dashboard interactif moderne avec FastAPI + Plotly pour monitoring complet :
- Vue temps réel de toute la pipeline
- Graphiques de performances (FPS, latence, GPU)
- Métriques détaillées par composant (RX, PROC, TX)
- Alertes automatiques et seuils configurables
- Historique glissant configurable
- API REST pour intégration externe

Architecture :
--------------
1. Backend FastAPI expose :
   - Endpoint /metrics (JSON live)
   - Endpoint /history (données historiques)
   - WebSocket pour push temps réel
   
2. Frontend Dash affiche :
   - Graphiques interactifs Plotly
   - Tables de métriques
   - Indicateurs de santé

Usage :
-------
    python -m service.dashboard_service --port 8050

    Puis ouvrir : http://localhost:8050
"""

import logging
import time
import json
import asyncio
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Thread, Lock

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Imports internes
from core.monitoring.monitor import get_aggregated_metrics, get_gpu_utilization
from core.queues.buffers import collect_queue_metrics

LOG = logging.getLogger("igt.dashboard")

# ═════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    """Configuration du dashboard"""
    port: int = 8050
    host: str = "0.0.0.0"
    history_size: int = 300  # Nombre de points d'historique (5 min @ 1Hz)
    update_interval: float = 1.0  # Secondes entre chaque mise à jour
    
    # Seuils d'alerte
    fps_warning: float = 70.0
    fps_critical: float = 50.0
    latency_warning: float = 30.0  # ms
    latency_critical: float = 50.0  # ms
    gpu_warning: float = 90.0  # %
    gpu_critical: float = 95.0  # %


# ═════════════════════════════════════════════════════════════
#  COLLECTEUR DE MÉTRIQUES
# ═════════════════════════════════════════════════════════════

class MetricsCollector:
    """Collecte et stocke les métriques de la pipeline"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.lock = Lock()
        
        # Historique des métriques (deque circulaire)
        self.history: deque = deque(maxlen=config.history_size)
        
        # Dernières métriques
        self.latest: Optional[Dict[str, Any]] = None
        
        # Compteurs
        self.total_frames_rx = 0
        self.total_frames_tx = 0
        self.total_drops = 0
        
    def collect(self) -> Dict[str, Any]:
        """Collecte toutes les métriques disponibles"""
        timestamp = time.time()
        
        # Métriques agrégées (monitor.py) - peut être None si pas de pipeline active
        aggregated = get_aggregated_metrics() or {}
        
        # Si pas de métriques temps réel, essayer de lire depuis les logs
        if not aggregated or aggregated.get("fps_in", 0.0) == 0.0:
            aggregated = self._read_metrics_from_logs()
        
        # Utilisation GPU
        gpu_util = get_gpu_utilization()
        
        # Métriques des queues
        queue_metrics = {}
        try:
            queue_metrics = collect_queue_metrics()
        except Exception:
            # Si les queues ne sont pas initialisées, continuer avec valeurs vides
            pass
        
        # Construction du snapshot complet
        snapshot = {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            
            # Pipeline globale
            "fps_in": aggregated.get("fps_in", 0.0),
            "fps_out": aggregated.get("fps_out", 0.0),
            "latency_ms": aggregated.get("latency_ms", 0.0),
            
            # RX/PROC/TX détaillés (depuis pipeline.log)
            "rx_count": aggregated.get("rx_count", 0),
            "proc_count": aggregated.get("proc_count", 0),
            "tx_count": aggregated.get("tx_count", 0),
            "last_frame_rx": aggregated.get("last_frame_rx", 0),
            "last_frame_proc": aggregated.get("last_frame_proc", 0),
            "last_frame_tx": aggregated.get("last_frame_tx", 0),
            "fps_rx": aggregated.get("fps_rx", 0.0),
            "fps_proc": aggregated.get("fps_proc", 0.0),
            "fps_tx": aggregated.get("fps_tx", 0.0),
            "sync_txproc": aggregated.get("sync_txproc", 0.0),
            
            # Latences inter-étapes
            "latency_rxproc_avg": aggregated.get("latency_rxproc_avg", 0.0),
            "latency_rxproc_last": aggregated.get("latency_rxproc_last", 0.0),
            "latency_proctx_avg": aggregated.get("latency_proctx_avg", 0.0),
            "latency_proctx_last": aggregated.get("latency_proctx_last", 0.0),
            "latency_rxtx_avg": aggregated.get("latency_rxtx_avg", 0.0),
            "latency_rxtx_last": aggregated.get("latency_rxtx_last", 0.0),
            
            # KPI globaux (depuis kpi.log)
            "fps_rx_kpi": aggregated.get("fps_rx_kpi", 0.0),
            "latency_kpi": aggregated.get("latency_kpi", 0.0),
            "drops_tx": aggregated.get("drops_tx", 0),
            
            # GPU
            "gpu_util": gpu_util,
            "gpu_memory_mb": self._get_gpu_memory(),
            
            # Queues
            "queue_rt_size": queue_metrics.get("rt.size", 0),
            "queue_rt_drops": queue_metrics.get("rt.drops_total", 0),
            "queue_gpu_size": queue_metrics.get("gpu.size", 0),
            "queue_out_size": queue_metrics.get("out.size", 0),
            
            # État de santé
            "health": self._compute_health(aggregated, gpu_util, queue_metrics),
        }
        
        with self.lock:
            self.history.append(snapshot)
            self.latest = snapshot
            
        return snapshot
    
    def _read_metrics_from_logs(self) -> Dict[str, float]:
        """Lit les métriques depuis kpi.log et pipeline.log"""
        import re
        from pathlib import Path
        
        # Parser pipeline.log pour RX/PROC/TX détaillés
        pipeline_metrics = self._parse_pipeline_log()
        
        # Parser kpi.log pour métriques globales
        kpi_log = Path("logs/kpi.log")
        kpi_metrics = {"fps_rx_kpi": 0.0, "latency_kpi": 0.0, "drops_tx": 0}
        
        if kpi_log.exists():
            try:
                with open(kpi_log, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[-50:]
                
                fps_values, latency_values, drops = [], [], []
                for line in lines:
                    if "fps_rx=" in line:
                        m = re.search(r"fps_rx=(\d+\.?\d*)", line)
                        if m:
                            fps_values.append(float(m.group(1)))
                    if "latency_ms" in line:
                        m = re.search(r"latency_ms[_a-z]*[:=](\d+\.?\d*)", line)
                        if m:
                            latency_values.append(float(m.group(1)))
                    if "tx.drop_total" in line:
                        m = re.search(r"tx\.drop_total=(\d+)", line)
                        if m:
                            drops.append(int(m.group(1)))
                
                kpi_metrics["fps_rx_kpi"] = round(sum(fps_values) / len(fps_values), 1) if fps_values else 0.0
                kpi_metrics["latency_kpi"] = round(sum(latency_values) / len(latency_values), 1) if latency_values else 0.0
                kpi_metrics["drops_tx"] = drops[-1] if drops else 0
            except Exception as e:
                LOG.debug(f"Could not read KPI log: {e}")
        
        # Fusionner les métriques
        return {
            "fps_in": pipeline_metrics.get("fps_rx", kpi_metrics["fps_rx_kpi"]),
            "fps_out": pipeline_metrics.get("fps_tx", 0.0),
            "latency_ms": pipeline_metrics.get("latency_rxtx_avg", kpi_metrics["latency_kpi"]),
            **pipeline_metrics,
            **kpi_metrics
        }
    
    def _parse_pipeline_log(self) -> Dict[str, Any]:
        """Parse pipeline.log pour extraire RX/PROC/TX avec timestamps"""
        import re
        from pathlib import Path
        
        pipeline_log = Path("logs/pipeline.log")
        if not pipeline_log.exists():
            return {}
        
        try:
            with open(pipeline_log, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-800:]  # Fenêtre de 800 lignes
            
            rx, proc, tx = [], [], []
            rx_t, proc_t, tx_t = {}, {}, {}
            ts_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]")
            
            for line in lines:
                t_match = ts_pattern.search(line)
                ts = None
                if t_match:
                    ts = (
                        time.mktime(time.strptime(f"{t_match.group(1)} {t_match.group(2)}", "%Y-%m-%d %H:%M:%S"))
                        + int(t_match.group(3)) / 1000
                    )
                
                # RX - chercher "Generated frame #XXX" ou "Generated frame —XXX"
                if "[RX-SIM]" in line and "Generated frame" in line:
                    m = re.search(r"frame [#—](\d+)", line)
                    if m:
                        fid = int(m.group(1))
                        rx.append(fid)
                        if ts:
                            rx_t[fid] = ts
                
                # PROC - chercher "Processed frame #XXX" ou "Processed frame —XXX"
                elif "[PROC-SIM]" in line and "Processed frame" in line:
                    m = re.search(r"frame [#—](\d+)", line)
                    if m:
                        fid = int(m.group(1))
                        proc.append(fid)
                        if ts:
                            proc_t[fid] = ts
                
                # TX - chercher "Sent frame #XXX" ou "Sent frame —XXX"
                elif "[TX-SIM]" in line and "Sent frame" in line:
                    m = re.search(r"frame [#—](\d+)", line)
                    if m:
                        fid = int(m.group(1))
                        tx.append(fid)
                        if ts:
                            tx_t[fid] = ts
            
            # Calcul des FPS
            fps_rx = self._fps_from_timestamps(rx_t)
            fps_proc = self._fps_from_timestamps(proc_t)
            fps_tx = self._fps_from_timestamps(tx_t)
            
            # Calcul des latences RX→PROC→TX
            latencies = self._compute_latencies_full(rx_t, proc_t, tx_t)
            
            # Synchro TX/PROC
            sync_txproc = round(100 * len(tx) / len(proc), 1) if proc else 0.0
            
            return {
                "rx_count": len(rx),
                "proc_count": len(proc),
                "tx_count": len(tx),
                "last_frame_rx": rx[-1] if rx else 0,
                "last_frame_proc": proc[-1] if proc else 0,
                "last_frame_tx": tx[-1] if tx else 0,
                "fps_rx": fps_rx,
                "fps_proc": fps_proc,
                "fps_tx": fps_tx,
                "sync_txproc": sync_txproc,
                "latency_rxproc_avg": latencies["rxproc_avg"],
                "latency_rxproc_last": latencies["rxproc_last"],
                "latency_proctx_avg": latencies["proctx_avg"],
                "latency_proctx_last": latencies["proctx_last"],
                "latency_rxtx_avg": latencies["rxtx_avg"],
                "latency_rxtx_last": latencies["rxtx_last"],
            }
        except Exception as e:
            LOG.debug(f"Could not parse pipeline log: {e}")
            return {}
    
    def _fps_from_timestamps(self, t_dict: Dict[int, float]) -> float:
        """Calcule le FPS à partir d'un dict {frame_id: timestamp}"""
        if len(t_dict) < 2:
            return 0.0
        times = sorted(t_dict.values())
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0
        return round((len(times) - 1) / duration, 1)
    
    def _compute_latencies_full(self, rx_t: Dict, proc_t: Dict, tx_t: Dict) -> Dict[str, float]:
        """Calcule les latencies RX→PROC, PROC→TX, RX→TX"""
        rxproc_vals = []
        proctx_vals = []
        rxtx_vals = []
        
        # RX→PROC
        for fid in (set(rx_t.keys()) & set(proc_t.keys())):
            dt_ms = (proc_t[fid] - rx_t[fid]) * 1000.0
            if dt_ms >= 0:
                rxproc_vals.append(dt_ms)
        
        # PROC→TX
        for fid in (set(proc_t.keys()) & set(tx_t.keys())):
            dt_ms = (tx_t[fid] - proc_t[fid]) * 1000.0
            if dt_ms >= 0:
                proctx_vals.append(dt_ms)
        
        # RX→TX (direct)
        for fid in (set(rx_t.keys()) & set(tx_t.keys())):
            dt_ms = (tx_t[fid] - rx_t[fid]) * 1000.0
            if dt_ms >= 0:
                rxtx_vals.append(dt_ms)
        
        return {
            "rxproc_avg": round(sum(rxproc_vals) / len(rxproc_vals), 1) if rxproc_vals else 0.0,
            "rxproc_last": round(rxproc_vals[-1], 1) if rxproc_vals else 0.0,
            "proctx_avg": round(sum(proctx_vals) / len(proctx_vals), 1) if proctx_vals else 0.0,
            "proctx_last": round(proctx_vals[-1], 1) if proctx_vals else 0.0,
            "rxtx_avg": round(sum(rxtx_vals) / len(rxtx_vals), 1) if rxtx_vals else 0.0,
            "rxtx_last": round(rxtx_vals[-1], 1) if rxtx_vals else 0.0,
        }
    
    def _compute_latencies_simple(self, proc_t: Dict, tx_t: Dict) -> Dict[str, float]:
        """Calcule les latences PROC→TX simplement"""
        vals = []
        for fid in (set(proc_t.keys()) & set(tx_t.keys())):
            dt_ms = (tx_t[fid] - proc_t[fid]) * 1000.0
            if dt_ms >= 0:
                vals.append(dt_ms)
        
        if not vals:
            return {
                "proctx_avg": 0.0,
                "proctx_last": 0.0,
                "rxtx_avg": 0.0,
                "rxtx_last": 0.0,
            }
        
        avg = round(sum(vals) / len(vals), 1)
        last = round(vals[-1], 1)
        
        return {
            "proctx_avg": avg,
            "proctx_last": last,
            "rxtx_avg": avg,  # RX≈PROC donc RX→TX ≈ PROC→TX
            "rxtx_last": last,
        }
    
    
    def _compute_latencies(self, rx_t: Dict, proc_t: Dict, tx_t: Dict) -> Dict[str, float]:
        """Calcule les latences RX→PROC, PROC→TX, RX→TX"""
        def pair_lat(d_a: Dict, d_b: Dict):
            vals = []
            for fid in (set(d_a.keys()) & set(d_b.keys())):
                dt_ms = (d_b[fid] - d_a[fid]) * 1000.0
                if dt_ms >= 0:
                    vals.append(dt_ms)
            if not vals:
                return 0.0, 0.0
            return round(sum(vals) / len(vals), 1), round(vals[-1], 1)
        
        avg_rxp, last_rxp = pair_lat(rx_t, proc_t)
        avg_pxt, last_pxt = pair_lat(proc_t, tx_t)
        avg_rxt, last_rxt = pair_lat(rx_t, tx_t)
        
        # Fallback si pas de RX→TX direct
        if avg_rxt == 0.0 and avg_pxt > 0.0:
            avg_rxt = round(avg_rxp + avg_pxt, 1) if avg_rxp > 0.0 else avg_pxt
            last_rxt = round(last_rxp + last_pxt, 1) if last_rxp > 0.0 else last_pxt
        
        return {
            "rxproc_avg": avg_rxp,
            "rxproc_last": last_rxp,
            "proctx_avg": avg_pxt,
            "proctx_last": last_pxt,
            "rxtx_avg": avg_rxt,
            "rxtx_last": last_rxt,
        }
    
    def _get_gpu_memory(self) -> float:
        """Récupère la mémoire GPU utilisée (MB)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return info.used / (1024 ** 2)  # Bytes → MB
        except Exception:
            return 0.0
    
    def _compute_health(self, agg: Dict, gpu: float, queues: Dict) -> str:
        """Calcule l'état de santé global : OK, WARNING, CRITICAL"""
        fps_in = agg.get("fps_in", 0.0)
        latency = agg.get("latency_ms", 0.0)
        
        # CRITICAL
        if fps_in < self.config.fps_critical:
            return "CRITICAL"
        if latency > self.config.latency_critical:
            return "CRITICAL"
        if gpu > self.config.gpu_critical:
            return "CRITICAL"
        
        # WARNING
        if fps_in < self.config.fps_warning:
            return "WARNING"
        if latency > self.config.latency_warning:
            return "WARNING"
        if gpu > self.config.gpu_warning:
            return "WARNING"
        
        return "OK"
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Retourne l'historique (derniers N points ou tout)"""
        with self.lock:
            if last_n:
                return list(self.history)[-last_n:]
            return list(self.history)
    
    def get_latest(self) -> Optional[Dict]:
        """Retourne le dernier snapshot"""
        with self.lock:
            return self.latest


# ═════════════════════════════════════════════════════════════
#  API FASTAPI
# ═════════════════════════════════════════════════════════════

def create_app(collector: MetricsCollector, config: DashboardConfig) -> FastAPI:
    """Crée l'application FastAPI"""
    
    app = FastAPI(
        title="Ultramotion IGT Dashboard",
        description="Real-time monitoring dashboard for IGT inference pipeline",
        version="1.0.0"
    )
    
    # CORS pour accès depuis le frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ─────────────────────────────────────────────────────────
    # Endpoints REST
    # ─────────────────────────────────────────────────────────
    
    @app.get("/")
    async def root():
        """Page d'accueil avec dashboard HTML"""
        return HTMLResponse(content=generate_dashboard_html(config))
    
    @app.get("/api/metrics/latest")
    async def get_latest_metrics():
        """Retourne les dernières métriques"""
        latest = collector.get_latest()
        if not latest:
            return JSONResponse({"status": "no_data"}, status_code=503)
        return JSONResponse(latest)
    
    @app.get("/api/metrics/history")
    async def get_history(last: int = 60):
        """Retourne l'historique des métriques"""
        history = collector.get_history(last_n=last)
        return JSONResponse({"data": history, "count": len(history)})
    
    @app.get("/api/health")
    async def health_check():
        """Endpoint de santé"""
        latest = collector.get_latest()
        if not latest:
            return JSONResponse({"status": "initializing"}, status_code=503)
        
        return JSONResponse({
            "status": latest["health"],
            "timestamp": latest["timestamp"],
            "uptime": time.time() - (collector.history[0]["timestamp"] if collector.history else time.time())
        })
    
    # ─────────────────────────────────────────────────────────
    # WebSocket pour streaming temps réel
    # ─────────────────────────────────────────────────────────
    
    @app.websocket("/ws/metrics")
    async def websocket_endpoint(websocket: WebSocket):
        """Stream de métriques en temps réel via WebSocket"""
        await websocket.accept()
        LOG.info("WebSocket client connected")
        
        try:
            while True:
                latest = collector.get_latest()
                if latest:
                    try:
                        await websocket.send_json(latest)
                    except Exception as e:
                        LOG.debug(f"Error sending to WebSocket: {e}")
                        break
                await asyncio.sleep(config.update_interval)
        except WebSocketDisconnect:
            LOG.info("WebSocket client disconnected normally")
        except Exception as e:
            LOG.debug(f"WebSocket connection closed: {e}")
        finally:
            LOG.info("WebSocket connection terminated")
    
    return app


# ═════════════════════════════════════════════════════════════
#  GÉNÉRATEUR HTML DASHBOARD
# ═════════════════════════════════════════════════════════════

def generate_dashboard_html(config: DashboardConfig) -> str:
    """Génère le HTML du dashboard interactif"""
    return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultramotion IGT Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }}
        .card h2 {{
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ font-weight: 500; opacity: 0.9; }}
        .metric-value {{
            font-weight: bold;
            font-size: 1.2em;
        }}
        .status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }}
        .status-ok {{ background: #10b981; }}
        .status-warning {{ background: #f59e0b; }}
        .status-critical {{ background: #ef4444; }}
        #chart-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            opacity: 0.8;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ultramotion IGT - Dashboard Temps Réel</h1>
        
        <div class="grid">
            <!-- Status Card -->
            <div class="card">
                <h2>📊 État Général</h2>
                <div class="metric">
                    <span class="metric-label">Statut</span>
                    <span id="health-status" class="status status-ok">OK</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Dernière mise à jour</span>
                    <span id="last-update" class="metric-value">--:--:--</span>
                </div>
            </div>
            
            <!-- RX/PROC/TX Card -->
            <div class="card">
                <h2>📡 Flux Pipeline (RX → PROC → TX)</h2>
                <div class="metric">
                    <span class="metric-label">RX: Frame #</span>
                    <span id="rx-last" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">RX: Count / FPS</span>
                    <span id="rx-count-fps" class="metric-value">0 / 0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">PROC: Frame #</span>
                    <span id="proc-last" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">PROC: Count / FPS</span>
                    <span id="proc-count-fps" class="metric-value">0 / 0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">TX: Frame #</span>
                    <span id="tx-last" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">TX: Count / FPS</span>
                    <span id="tx-count-fps" class="metric-value">0 / 0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Synchro TX/PROC (%)</span>
                    <span id="sync-txproc" class="metric-value">0.0</span>
                </div>
            </div>
            
            <!-- Latencies Card -->
            <div class="card">
                <h2>⏱️ Latences Inter-Étapes</h2>
                <div class="metric">
                    <span class="metric-label">RX → PROC (avg/last ms)</span>
                    <span id="lat-rxproc" class="metric-value">0.0 / 0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">PROC → TX (avg/last ms)</span>
                    <span id="lat-proctx" class="metric-value">0.0 / 0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">RX → TX total (avg/last ms)</span>
                    <span id="lat-rxtx" class="metric-value">0.0 / 0.0</span>
                </div>
            </div>
            
            <!-- KPI Card -->
            <div class="card">
                <h2>📈 KPI Globaux</h2>
                <div class="metric">
                    <span class="metric-label">FPS RX (kpi.log)</span>
                    <span id="fps-rx-kpi" class="metric-value">0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Latence KPI (ms)</span>
                    <span id="latency-kpi" class="metric-value">0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Drops TX</span>
                    <span id="drops-tx" class="metric-value">0</span>
                </div>
            </div>
            
            <!-- GPU Card -->
            <div class="card">
                <h2>🎮 GPU</h2>
                <div class="metric">
                    <span class="metric-label">Utilisation (%)</span>
                    <span id="gpu-util" class="metric-value">0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mémoire (MB)</span>
                    <span id="gpu-memory" class="metric-value">0</span>
                </div>
            </div>
            
            <!-- Queues Card -->
            <div class="card">
                <h2>📦 Files d'attente</h2>
                <div class="metric">
                    <span class="metric-label">Queue RT</span>
                    <span id="queue-rt" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Queue GPU</span>
                    <span id="queue-gpu" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Drops RT</span>
                    <span id="queue-drops" class="metric-value">0</span>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div id="chart-container">
            <div id="fps-chart" style="height: 300px;"></div>
            <div id="latency-chart" style="height: 300px; margin-top: 20px;"></div>
            <div id="gpu-chart" style="height: 300px; margin-top: 20px;"></div>
        </div>
        
        <div class="footer">
            Ultramotion IGT Inference Pipeline · Mis à jour toutes les {config.update_interval}s
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${{window.location.hostname}}:{config.port}/ws/metrics`);
        
        // Data buffers
        const maxPoints = 100;
        const timestamps = [];
        const fpsInData = [];
        const fpsOutData = [];
        const latencyData = [];
        const gpuData = [];
        
        // Initialize charts
        const fpsLayout = {{
            title: 'FPS Temps Réel',
            xaxis: {{ title: 'Temps' }},
            yaxis: {{ title: 'FPS' }},
            showlegend: true,
            margin: {{ t: 40, r: 20, b: 40, l: 50 }}
        }};
        
        const latencyLayout = {{
            title: 'Latence Pipeline (ms)',
            xaxis: {{ title: 'Temps' }},
            yaxis: {{ title: 'Latence (ms)' }},
            margin: {{ t: 40, r: 20, b: 40, l: 50 }}
        }};
        
        const gpuLayout = {{
            title: 'Utilisation GPU (%)',
            xaxis: {{ title: 'Temps' }},
            yaxis: {{ title: 'Utilisation (%)', range: [0, 100] }},
            margin: {{ t: 40, r: 20, b: 40, l: 50 }}
        }};
        
        Plotly.newPlot('fps-chart', [], fpsLayout);
        Plotly.newPlot('latency-chart', [], latencyLayout);
        Plotly.newPlot('gpu-chart', [], gpuLayout);
        
        // Handle WebSocket messages
        ws.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            updateDashboard(data);
        }};
        
        ws.onerror = function(error) {{
            console.error('WebSocket error:', error);
            document.getElementById('health-status').textContent = 'ERREUR';
            document.getElementById('health-status').className = 'status status-critical';
        }};
        
        function updateDashboard(data) {{
            // Update RX/PROC/TX metrics
            document.getElementById('rx-last').textContent = data.last_frame_rx || 0;
            document.getElementById('rx-count-fps').textContent = `${{data.rx_count || 0}} / ${{(data.fps_rx || 0).toFixed(1)}}`;
            document.getElementById('proc-last').textContent = data.last_frame_proc || 0;
            document.getElementById('proc-count-fps').textContent = `${{data.proc_count || 0}} / ${{(data.fps_proc || 0).toFixed(1)}}`;
            document.getElementById('tx-last').textContent = data.last_frame_tx || 0;
            document.getElementById('tx-count-fps').textContent = `${{data.tx_count || 0}} / ${{(data.fps_tx || 0).toFixed(1)}}`;
            document.getElementById('sync-txproc').textContent = (data.sync_txproc || 0).toFixed(1);
            
            // Update latencies
            document.getElementById('lat-rxproc').textContent = 
                `${{(data.latency_rxproc_avg || 0).toFixed(1)}} / ${{(data.latency_rxproc_last || 0).toFixed(1)}}`;
            document.getElementById('lat-proctx').textContent = 
                `${{(data.latency_proctx_avg || 0).toFixed(1)}} / ${{(data.latency_proctx_last || 0).toFixed(1)}}`;
            document.getElementById('lat-rxtx').textContent = 
                `${{(data.latency_rxtx_avg || 0).toFixed(1)}} / ${{(data.latency_rxtx_last || 0).toFixed(1)}}`;
            
            // Update KPI metrics
            document.getElementById('fps-rx-kpi').textContent = (data.fps_rx_kpi || 0).toFixed(1);
            document.getElementById('latency-kpi').textContent = (data.latency_kpi || 0).toFixed(1);
            document.getElementById('drops-tx').textContent = data.drops_tx || 0;
            
            // Update GPU
            document.getElementById('gpu-util').textContent = data.gpu_util.toFixed(1);
            document.getElementById('gpu-memory').textContent = Math.round(data.gpu_memory_mb);
            
            // Update queues
            document.getElementById('queue-rt').textContent = data.queue_rt_size;
            document.getElementById('queue-gpu').textContent = data.queue_gpu_size;
            document.getElementById('queue-drops').textContent = data.queue_rt_drops;
            
            // Update status
            const statusEl = document.getElementById('health-status');
            statusEl.textContent = data.health;
            statusEl.className = 'status status-' + data.health.toLowerCase();
            
            // Update timestamp
            const now = new Date(data.datetime);
            document.getElementById('last-update').textContent = now.toLocaleTimeString();
            
            // Update charts data
            const timeLabel = now.toLocaleTimeString();
            timestamps.push(timeLabel);
            fpsInData.push(data.fps_rx || data.fps_in);
            fpsOutData.push(data.fps_tx || data.fps_out);
            latencyData.push(data.latency_rxtx_avg || data.latency_ms);
            gpuData.push(data.gpu_util);
            
            // Keep only last N points
            if (timestamps.length > maxPoints) {{
                timestamps.shift();
                fpsInData.shift();
                fpsOutData.shift();
                latencyData.shift();
                gpuData.shift();
            }}
            
            // Update FPS chart
            Plotly.react('fps-chart', [
                {{ x: timestamps, y: fpsInData, name: 'FPS RX', type: 'scatter', mode: 'lines+markers', line: {{ color: '#10b981' }} }},
                {{ x: timestamps, y: fpsOutData, name: 'FPS TX', type: 'scatter', mode: 'lines+markers', line: {{ color: '#3b82f6' }} }}
            ], fpsLayout);
            
            // Update latency chart
            Plotly.react('latency-chart', [
                {{ x: timestamps, y: latencyData, name: 'Latence RX→TX', type: 'scatter', mode: 'lines+markers', line: {{ color: '#f59e0b' }}, fill: 'tozeroy' }}
            ], latencyLayout);
            
            // Update GPU chart
            Plotly.react('gpu-chart', [
                {{ x: timestamps, y: gpuData, name: 'GPU', type: 'scatter', mode: 'lines+markers', line: {{ color: '#8b5cf6' }}, fill: 'tozeroy' }}
            ], gpuLayout);
        }}
    </script>
</body>
</html>
"""


# ═════════════════════════════════════════════════════════════
#  THREAD DE COLLECTE
# ═════════════════════════════════════════════════════════════

class CollectorThread(Thread):
    """Thread qui collecte périodiquement les métriques"""
    
    def __init__(self, collector: MetricsCollector, interval: float):
        super().__init__(daemon=True, name="MetricsCollector")
        self.collector = collector
        self.interval = interval
        self.running = True
    
    def run(self):
        """Boucle de collecte"""
        LOG.info("Metrics collector thread started")
        while self.running:
            try:
                self.collector.collect()
            except Exception as e:
                LOG.error(f"Error collecting metrics: {e}", exc_info=True)
            time.sleep(self.interval)
    
    def stop(self):
        """Arrête le thread"""
        self.running = False


# ═════════════════════════════════════════════════════════════
#  SERVICE PRINCIPAL
# ═════════════════════════════════════════════════════════════

class DashboardService:
    """Service de dashboard principal"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.collector = MetricsCollector(self.config)
        self.collector_thread: Optional[CollectorThread] = None
        self.app = create_app(self.collector, self.config)
    
    def start(self):
        """Démarre le service"""
        LOG.info(f"Starting dashboard service on {self.config.host}:{self.config.port}")
        
        # Démarre le thread de collecte
        self.collector_thread = CollectorThread(
            self.collector, 
            self.config.update_interval
        )
        self.collector_thread.start()
        
        # Démarre le serveur FastAPI
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
    
    def stop(self):
        """Arrête le service"""
        if self.collector_thread:
            self.collector_thread.stop()
        LOG.info("Dashboard service stopped")


# ═════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════

def main():
    """Point d'entrée CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultramotion IGT Dashboard Service")
    parser.add_argument("--port", type=int, default=8050, help="Port du serveur")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host du serveur")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalle de collecte (s)")
    
    args = parser.parse_args()
    
    config = DashboardConfig(
        port=args.port,
        host=args.host,
        update_interval=args.interval
    )
    
    service = DashboardService(config)
    
    try:
        service.start()
    except KeyboardInterrupt:
        LOG.info("Shutting down...")
        service.stop()


if __name__ == "__main__":
    main()
