import sys
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# Ajout du chemin vers src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from service.gateway.stats import GatewayStats


def format_ts(ts: float) -> str:
    """Convertit un timestamp epoch en heure locale lisible (Europe/Zurich)."""
    try:
        dt = datetime.fromtimestamp(ts, tz=ZoneInfo("Europe/Zurich"))
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecondes
    except Exception:
        return f"{ts:.3f}"  # fallback si erreur


# Création du collecteur de statistiques
stats = GatewayStats(rolling_window_size=5, latency_window_size=20)

# Simulation : réception et envoi de 5 frames
for frame_id in range(5):
    ts_rx = time.time()
    stats.mark_rx(frame_id, ts_rx)
    stats.update_rx(fps=25.0, ts=ts_rx, bytes_count=512 * 512)
    time.sleep(0.02)  # délai entre RX et TX (~20 ms)
    ts_tx = time.time()
    stats.mark_tx(frame_id, ts_tx)
    stats.update_tx(fps=25.0, bytes_count=256 * 256)

# Snapshot final
snapshot = stats.snapshot()

print("=== Exemple de snapshot (horodatages lisibles) ===")
for k, v in snapshot.items():
    if "ts" in k or "update" in k or "started" in k:  # tous les timestamps
        print(f"{k:20s}: {format_ts(v)}")
    else:
        print(f"{k:20s}: {v}")
