"""
Dashboard complet pour la pipeline Ultramotion (Gateway temps réel)
===================================================================

Affiche :
 - RX / PROC / TX avec fps et synchro colorés
 - Latences RX→PROC / PROC→TX / RX→TX [ms]
 - KPI globaux : fps_rx, latence moyenne
 - Drops TX
 - Charge CPU / mémoire

Usage :
    python tools/dashboard_gateway.py
"""

import time
import re
import psutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
LOG_DIR = Path("logs")
PIPELINE_LOG = LOG_DIR / "pipeline.log"
KPI_LOG = LOG_DIR / "kpi.log"
WINDOW_LINES = 800

console = Console()

# ──────────────────────────────────────────────
# UTILITAIRES DE LECTURE
# ──────────────────────────────────────────────
def tail_file(path: Path, n=WINDOW_LINES):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []

# ──────────────────────────────────────────────
# PARSING PIPELINE.LOG
# ──────────────────────────────────────────────
def parse_pipeline():
    rx, proc, tx = [], [], []
    rx_t, proc_t, tx_t = {}, {}, {}
    ts_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]")

    for l in tail_file(PIPELINE_LOG):
        t = ts_pattern.search(l)
        ts = None
        if t:
            ts = (
                time.mktime(time.strptime(f"{t.group(1)} {t.group(2)}", "%Y-%m-%d %H:%M:%S"))
                + int(t.group(3)) / 1000
            )
        # RX
        if "[RX-SIM]" in l and "Frame generator" not in l and "frame #" in l:
            m = re.search(r"frame #(\d+)", l)
            if m:
                fid = int(m.group(1))
                rx.append(fid)
                if ts:
                    rx_t[fid] = ts
        # PROC
        elif "[PROC-SIM]" in l and "Processed frame" in l:
            m = re.search(r"frame #(\d+)", l)
            if m:
                fid = int(m.group(1))
                proc.append(fid)
                if ts:
                    proc_t[fid] = ts
        # TX
        elif "[TX-SIM]" in l and "Sent frame" in l:
            m = re.search(r"#(\d+)", l)
            if m:
                fid = int(m.group(1))
                tx.append(fid)
                if ts:
                    tx_t[fid] = ts
    return rx, proc, tx, rx_t, proc_t, tx_t

# ──────────────────────────────────────────────
# KPI : FPS RX / latence / drops
# ──────────────────────────────────────────────
def parse_kpi():
    lines = tail_file(KPI_LOG, 500)
    fps_rx_values, lats, drops = [], [], []
    for l in lines:
        if "fps_rx=" in l:
            m = re.search(r"fps_rx=(\d+\.\d+)", l)
            if m:
                fps_rx_values.append(float(m.group(1)))
        if "latency_ms" in l:
            m = re.search(r"latency_ms[_a-z]*[:=](\d+\.\d+)", l)
            if m:
                lats.append(float(m.group(1)))
        if "tx.drop_total" in l:
            m = re.search(r"tx\.drop_total=(\d+)", l)
            if m:
                drops.append(int(m.group(1)))
    avg_fps_rx = round(sum(fps_rx_values) / len(fps_rx_values), 1) if fps_rx_values else 0
    avg_lat = round(sum(lats) / len(lats), 1) if lats else 0
    last_drop = drops[-1] if drops else 0
    return avg_fps_rx, avg_lat, last_drop

# ──────────────────────────────────────────────
# LATENCES RX→PROC / PROC→TX / RX→TX
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# LATENCES RX→PROC / PROC→TX / RX→TX
# ──────────────────────────────────────────────
def compute_latencies(rx_t, proc_t, tx_t):
    """Return ((avg_rxp,last_rxp), (avg_pxt,last_pxt), (avg_rxt,last_rxt)).
    Each leg is computed on the intersection of its two dicts.
    """
    def pair_lat(d_a, d_b):
        vals = []
        for fid in (set(d_a.keys()) & set(d_b.keys())):
            dt_ms = (d_b[fid] - d_a[fid]) * 1000.0
            if dt_ms >= 0:
                vals.append(dt_ms)
        if not vals:
            return 0.0, 0.0
        return round(sum(vals) / len(vals), 1), round(vals[-1], 1)

    avg_rxp, last_rxp = pair_lat(rx_t, proc_t)   # RX→PROC
    avg_pxt, last_pxt = pair_lat(proc_t, tx_t)   # PROC→TX
    avg_rxt, last_rxt = pair_lat(rx_t, tx_t)     # RX→TX direct

    # Fallback when there is no per-frame RX in logs (common in your current run):
    if avg_rxt == 0.0 and avg_pxt > 0.0:
        avg_rxt = round(avg_rxp + avg_pxt, 1) if avg_rxp > 0.0 else avg_pxt
        last_rxt = round(last_rxp + last_pxt, 1) if last_rxp > 0.0 else last_pxt

    return (avg_rxp, last_rxp), (avg_pxt, last_pxt), (avg_rxt, last_rxt)


# ──────────────────────────────────────────────
# FPS calculé sur timestamps
# ──────────────────────────────────────────────
def fps_from_timestamps(t_dict):
    if len(t_dict) < 2:
        return 0.0
    times = sorted(t_dict.values())
    duration = times[-1] - times[0]
    if duration <= 0:
        return 0.0
    return round((len(times) - 1) / duration, 1)

# ──────────────────────────────────────────────
# TABLE PRINCIPALE
# ──────────────────────────────────────────────
def build_table():
    rx, proc, tx, rx_t, proc_t, tx_t = parse_pipeline()
    (avg_rxp, last_rxp), (avg_pxt, last_pxt), (avg_rxt, last_rxt) = compute_latencies(rx_t, proc_t, tx_t)
    fps_rx, avg_kpi_lat, drop_tx = parse_kpi()
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent

    fps_proc = fps_from_timestamps(proc_t)
    fps_tx = fps_from_timestamps(tx_t)
    sync_txproc = round(100 * len(tx) / len(proc), 1) if proc else 0

    def color(val, lo, hi):
        if val >= hi:
            return "green"
        elif val >= lo:
            return "yellow"
        return "red"

    table = Table(title="📡 Ultramotion Gateway Live Dashboard", show_lines=False)
    table.add_column("Flux", justify="center", style="cyan bold")
    table.add_column("Dernière frame", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Synchro TX/PROC", justify="right")
    table.add_column("RX→PROC [ms]", justify="right")
    table.add_column("PROC→TX [ms]", justify="right")
    table.add_column("RX→TX total [ms]", justify="right")
    table.add_column("fps_RX [kpi]", justify="right")
    table.add_column("KPI Lat [ms]", justify="right")
    table.add_column("Drops TX", justify="right")
    table.add_column("CPU / MEM", justify="center")

    table.add_row(
        "PROC",
        str(proc[-1] if proc else "-"),
        str(len(proc)),
        Text(f"{fps_proc:.1f}", style=color(fps_proc, 60, 90)),
        "-",
        Text(f"{avg_rxp:.1f}", style=color(avg_rxp, 1, 20)),
        "-",
        "-",
        Text(f"{fps_rx:.1f}", style=color(fps_rx, 60, 90)),
        Text(f"{avg_kpi_lat:.1f}", style=color(avg_kpi_lat, 10, 40)),
        Text(str(drop_tx), style=color(drop_tx, 1, 10)),
        f"{cpu:.1f}% / {mem:.1f}%",
    )

    table.add_row(
        "TX",
        str(tx[-1] if tx else "-"),
        str(len(tx)),
        Text(f"{fps_tx:.1f}", style=color(fps_tx, 60, 90)),
        Text(f"{sync_txproc:.1f}%", style=color(sync_txproc, 80, 95)),
        "-",
        Text(f"{avg_pxt:.1f}", style=color(avg_pxt, 1, 20)),
        Text(f"{avg_rxt:.1f}", style=color(avg_rxt, 2, 30)),
        "-",
        "-",
        "-",
        "-",
    )

    return table

# ──────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────
def live_dashboard():
    with Live(console=console, refresh_per_second=2):
        while True:
            console.clear()
            console.print(build_table())
            console.print("-" * 110)
            time.sleep(1)

if __name__ == "__main__":
    console.print("[green bold]🚀 Dashboard Gateway lancé[/green bold]")
    console.print(f"Lecture en continu depuis : {PIPELINE_LOG} et {KPI_LOG}")
    live_dashboard()
