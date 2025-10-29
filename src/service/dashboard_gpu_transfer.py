"""
Dashboard de Visualisation - GPU Transfer (Étapes 1 & 2)
========================================================

Dashboard temps réel pour monitorer les performances du transfert CPU→GPU :

Métriques affichées :
--------------------
1. Latence CPU→GPU par frame (total_ms)
2. Décomposition des étapes :
   - norm_ms : Normalisation [0,255] → [0,1]
   - pin_ms  : Allocation/copie vers pinned memory
   - copy_ms : Transfert asynchrone CPU→GPU
3. Throughput (frames/sec)
4. Statistiques temps réel (moyenne, min, max)

Source des données :
-------------------
Parse les logs KPI générés par cpu_to_gpu.py :
  event=copy_async device=cuda:0 H=512 W=512 norm_ms=1.0 pin_ms=2.0 copy_ms=1.0 total_ms=4.0 frame=0

Interface :
----------
- FastAPI backend sur http://localhost:8051
- Plotly graphs avec auto-refresh toutes les 2 secondes
- Graphiques interactifs avec zoom/pan

Inspiré de dashboard_service.py mais simplifié pour Étapes 1+2.
"""

import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, deque

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

LOG = logging.getLogger("dashboard.gpu")

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
KPI_LOG_PATH = LOGS_DIR / "kpi.log"

# Taille maximale du buffer (garder les N dernières frames)
MAX_BUFFER_SIZE = 500

# ──────────────────────────────────────────────
#  Collecteur de Métriques GPU
# ──────────────────────────────────────────────
class GPUMetricsCollector:
    """Parse kpi.log et extrait les métriques de transfert GPU."""
    
    def __init__(self, kpi_log_path: Path):
        self.kpi_log_path = kpi_log_path
        self.last_position = 0
        
        # Buffers circulaires pour les métriques
        self.timestamps = deque(maxlen=MAX_BUFFER_SIZE)
        self.frame_ids = deque(maxlen=MAX_BUFFER_SIZE)
        self.total_latencies = deque(maxlen=MAX_BUFFER_SIZE)
        self.norm_latencies = deque(maxlen=MAX_BUFFER_SIZE)
        self.pin_latencies = deque(maxlen=MAX_BUFFER_SIZE)
        self.copy_latencies = deque(maxlen=MAX_BUFFER_SIZE)
        
        # Regex pour parser les logs KPI copy_async
        # Format: event=copy_async device=cuda:0 H=512 W=512 norm_ms=1.0 pin_ms=2.0 copy_ms=1.0 total_ms=4.0 frame=0
        self.copy_async_pattern = re.compile(
            r"event=copy_async.*?"
            r"norm_ms=([\d.]+).*?"
            r"pin_ms=([\d.]+).*?"
            r"copy_ms=([\d.]+).*?"
            r"total_ms=([\d.]+).*?"
            r"frame=(\d+)"
        )
    
    def parse_kpi_log(self) -> None:
        """Parse les nouvelles lignes du fichier kpi.log."""
        if not self.kpi_log_path.exists():
            return
        
        try:
            with open(self.kpi_log_path, "r", encoding="utf-8", errors="ignore") as f:
                # Se positionner à la dernière position lue
                f.seek(self.last_position)
                
                for line in f:
                    # Parser les événements copy_async
                    match = self.copy_async_pattern.search(line)
                    if match:
                        norm_ms = float(match.group(1))
                        pin_ms = float(match.group(2))
                        copy_ms = float(match.group(3))
                        total_ms = float(match.group(4))
                        frame_id = int(match.group(5))
                        
                        # Ajouter aux buffers
                        self.timestamps.append(time.time())
                        self.frame_ids.append(frame_id)
                        self.total_latencies.append(total_ms)
                        self.norm_latencies.append(norm_ms)
                        self.pin_latencies.append(pin_ms)
                        self.copy_latencies.append(copy_ms)
                
                # Mettre à jour la position
                self.last_position = f.tell()
        
        except Exception as e:
            LOG.debug(f"Error parsing kpi.log: {e}")
    
    def get_metrics(self) -> Dict:
        """Retourne les métriques sous forme de dict pour l'API."""
        # Parser les nouvelles données
        self.parse_kpi_log()
        
        # Convertir deques en listes pour JSON
        frames = list(self.frame_ids)
        total = list(self.total_latencies)
        norm = list(self.norm_latencies)
        pin = list(self.pin_latencies)
        copy = list(self.copy_latencies)
        
        # Calculer statistiques
        stats = {}
        if total:
            stats = {
                "total_frames": len(frames),
                "avg_latency": round(sum(total) / len(total), 2),
                "min_latency": round(min(total), 2),
                "max_latency": round(max(total), 2),
                "avg_norm": round(sum(norm) / len(norm), 2),
                "avg_pin": round(sum(pin) / len(pin), 2),
                "avg_copy": round(sum(copy) / len(copy), 2),
            }
            
            # Calculer throughput (frames/sec) sur les 10 dernières frames
            if len(self.timestamps) >= 2:
                recent_timestamps = list(self.timestamps)[-10:]
                if len(recent_timestamps) >= 2:
                    time_span = recent_timestamps[-1] - recent_timestamps[0]
                    if time_span > 0:
                        fps = (len(recent_timestamps) - 1) / time_span
                        stats["throughput_fps"] = round(fps, 1)
                    else:
                        stats["throughput_fps"] = 0.0
                else:
                    stats["throughput_fps"] = 0.0
        
        return {
            "frames": frames,
            "total_ms": total,
            "norm_ms": norm,
            "pin_ms": pin,
            "copy_ms": copy,
            "stats": stats,
        }


# ──────────────────────────────────────────────
#  Application FastAPI
# ──────────────────────────────────────────────
app = FastAPI(title="GPU Transfer Dashboard")

# Collecteur global
collector = GPUMetricsCollector(KPI_LOG_PATH)

# Templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Page principale du dashboard."""
    return templates.TemplateResponse("gpu_dashboard.html", {"request": request})


@app.get("/api/metrics")
async def get_metrics() -> JSONResponse:
    """Endpoint API pour récupérer les métriques."""
    metrics = collector.get_metrics()
    return JSONResponse(content=metrics)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "gpu-transfer-dashboard"}


# ──────────────────────────────────────────────
#  Point d'entrée
# ──────────────────────────────────────────────
def run_dashboard(host: str = "127.0.0.1", port: int = 8051):
    """Démarre le serveur dashboard."""
    LOG.info(f"Starting GPU Transfer Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard()
