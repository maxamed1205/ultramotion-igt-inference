"""
Dashboard console pour visualiser en temps réel les logs de la pipeline Ultramotion.

Affiche :
 - RX / PROC / TX avec fréquences colorées (fps)
 - Synchronisation TX/RX
 - Délais RX→PROC / PROC→TX / RX→TX / TOTAL [ms] (moyenne + dernière frame)
 - Score et latence moyenne extraits de kpi.log

Usage :
    python tools/dashboard_logs.py

Dépendances :
    pip install rich
"""

import time
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.live import Live

# ──────────────────────────────────────────────
#  CONFIGURATION DES FICHIERS DE LOG
# ──────────────────────────────────────────────
LOG_DIR = Path("logs")
PIPELINE_LOG = LOG_DIR / "pipeline.log"
KPI_LOG = LOG_DIR / "kpi.log"

console = Console()

# Taille de la fenêtre d'analyse (nb de lignes récentes)
WINDOW_LINES = 500


# ──────────────────────────────────────────────
#  UTILITAIRES DE LECTURE DES LOGS
# ──────────────────────────────────────────────
def tail_file(path: Path, n=WINDOW_LINES):
    """Retourne les n dernières lignes d'un fichier texte."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


def parse_logs():
    """Analyse pipeline.log pour extraire les infos RX / PROC / TX."""
    rx, proc, tx = [], [], []
    for line in tail_file(PIPELINE_LOG):
        if "[RX]" in line and "Produced frame" in line:
            m = re.search(r"frame #(\d+)", line)
            if m:
                rx.append(int(m.group(1)))
        elif "[PROC]" in line and "Processed frame" in line:
            m = re.search(r"frame #(\d+)", line)
            if m:
                proc.append(int(m.group(1)))
        elif "[TX]" in line and "Sent" in line:
            m = re.search(r"#(\d+)", line)
            if m:
                tx.append(int(m.group(1)))
    return rx, proc, tx


def parse_kpi():
    """Analyse kpi.log pour extraire score moyen et latence moyenne."""
    lines = tail_file(KPI_LOG, 300)
    scores, lats = [], []
    for l in lines:
        if "score" in l:
            m = re.search(r"score[:=](\d+\.\d+)", l)
            if m:
                scores.append(float(m.group(1)))
        if "latency_ms" in l:
            m = re.search(r"latency_ms[:=](\d+\.\d+)", l)
            if m:
                lats.append(float(m.group(1)))
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0
    avg_lat = round(sum(lats) / len(lats), 1) if lats else 0
    return avg_score, avg_lat


# ──────────────────────────────────────────────
#  CALCUL DES DÉLAIS ENTRE RX / PROC / TX
# ──────────────────────────────────────────────
def parse_timestamps():
    """Analyse pipeline.log pour estimer les délais RX→PROC→TX."""
    lines = tail_file(PIPELINE_LOG)
    rx_times, proc_times, tx_times = {}, {}, {}

    for l in lines:
        # extraction timestamp [YYYY-MM-DD HH:MM:SS,mmm]
        t = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]", l)
        if not t:
            continue
        ts = time.mktime(time.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")) + float(t.group(2)) / 1000

        if "[RX]" in l and "Produced frame" in l:
            m = re.search(r"frame #(\d+)", l)
            if m:
                rx_times[int(m.group(1))] = ts
        elif "[PROC]" in l and "Processed frame" in l:
            m = re.search(r"frame #(\d+)", l)
            if m:
                proc_times[int(m.group(1))] = ts
        elif "[TX]" in l and "Sent" in l:
            m = re.search(r"#(\d+)", l)
            if m:
                tx_times[int(m.group(1))] = ts

    delays = []
    for fid in rx_times:
        if fid in proc_times and fid in tx_times:
            d_rx_proc = (proc_times[fid] - rx_times[fid]) * 1000
            d_proc_tx = (tx_times[fid] - proc_times[fid]) * 1000
            d_rx_tx = (tx_times[fid] - rx_times[fid]) * 1000
            d_total = d_rx_proc + d_proc_tx
            delays.append((fid, d_rx_proc, d_proc_tx, d_rx_tx, d_total))

    if not delays:
        return (0, 0, 0, 0), (0, 0, 0, 0)

    # Moyennes
    avg = tuple(round(sum(x[i] for x in delays) / len(delays), 1) for i in (1, 2, 3, 4))
    # Dernières valeurs
    last = tuple(round(delays[-1][i], 1) for i in (1, 2, 3, 4))
    return avg, last


# ──────────────────────────────────────────────
#  TABLE PRINCIPALE RX / PROC / TX
# ──────────────────────────────────────────────
def build_table():
    """Construit le tableau de performance complet de la pipeline."""
    rx, proc, tx = parse_logs()
    avg, last = parse_timestamps()
    avg_rxp, avg_pxt, avg_rxt, avg_tot = avg
    last_rxp, last_pxt, last_rxt, last_tot = last

    table = Table(title="📊 Ultramotion Gateway Monitor", show_lines=False)
    table.add_column("Flux", style="bold cyan", justify="center")
    table.add_column("Dernière frame", justify="right")
    table.add_column("Δ frames [count]", justify="right")
    table.add_column("Fréquence [fps]", justify="right")
    table.add_column("Synchro TX/RX [%]", justify="right")
    table.add_column("⏱ RX→PROC [ms]", justify="right")
    table.add_column("⏱ PROC→TX [ms]", justify="right")
    table.add_column("⏱ RX→TX [ms]", justify="right")
    table.add_column("⏱ TOTAL [ms]", justify="right")

    def fps(seq):
        """Calcule un FPS approximatif sur 5 secondes."""
        return round(len(seq) / 5.0, 1) if len(seq) >= 2 else 0

    def synchro(rx, tx):
        """Pourcentage de frames TX par rapport à RX."""
        return round(100 * len(tx) / max(len(rx), 1), 1) if rx and tx else 0

    fps_rx, fps_proc, fps_tx = fps(rx), fps(proc), fps(tx)
    sync = synchro(rx, tx)

    # Couleurs dynamiques
    def colorize(val, low, mid, high):
        if val >= high:
            return "green"
        elif val >= mid:
            return "yellow"
        return "red"

    def color_latency(val):
        if val <= 20:
            return "green"
        elif val <= 40:
            return "yellow"
        return "red"

    # RX
    table.add_row(
        "RX",
        str(rx[-1] if rx else "-"),
        f"{len(rx)}",
        Text(f"{fps_rx} fps", style=colorize(fps_rx, 5, 8, 9)),
        "-",
        "-", "-", "-", "-"
    )

    # PROC (moyennes)
    table.add_row(
        "PROC",
        str(proc[-1] if proc else "-"),
        f"{len(proc)}",
        Text(f"{fps_proc} fps", style=colorize(fps_proc, 5, 8, 9)),
        "-",
        Text(f"{avg_rxp:.1f}", style=color_latency(avg_rxp)),
        Text(f"{avg_pxt:.1f}", style=color_latency(avg_pxt)),
        Text(f"{avg_rxt:.1f}", style=color_latency(avg_rxt)),
        Text(f"{avg_tot:.1f}", style=color_latency(avg_tot)),
    )

    # TX (dernières valeurs)
    table.add_row(
        "TX",
        str(tx[-1] if tx else "-"),
        f"{len(tx)}",
        Text(f"{fps_tx} fps", style=colorize(fps_tx, 5, 8, 9)),
        Text(f"{sync} %", style=colorize(sync, 70, 90, 99)),
        Text(f"{last_rxp:.1f}", style=color_latency(last_rxp)),
        Text(f"{last_pxt:.1f}", style=color_latency(last_pxt)),
        Text(f"{last_rxt:.1f}", style=color_latency(last_rxt)),
        Text(f"{last_tot:.1f}", style=color_latency(last_tot)),
    )

    return table


# ──────────────────────────────────────────────
#  BOUCLE DE RAFRAÎCHISSEMENT
# ──────────────────────────────────────────────
def live_dashboard():
    """Affiche le tableau + KPIs en direct avec rafraîchissement."""
    with Live(console=console, refresh_per_second=2):
        while True:
            console.clear()
            table = build_table()
            avg_score, avg_lat = parse_kpi()
            console.print(table)
            console.print(
                f"[bold white]Score moyen :[/bold white] {avg_score:.3f}    "
                f"[bold white]Latence moyenne :[/bold white] {avg_lat:.1f} ms"
            )
            console.print("-" * 100)
            time.sleep(1)


# ──────────────────────────────────────────────
#  POINT D’ENTRÉE
# ──────────────────────────────────────────────
if __name__ == "__main__":
    console.print("[bold green]Dashboard des logs lancé...[/bold green]")
    console.print(f"Lecture en continu dans : [cyan]{PIPELINE_LOG}[/cyan]")
    live_dashboard()
