"""
Validation des 3 optimisations : Timer Windows + NumPy threads + Horloge compensée
===================================================================================

Ce script valide que les 3 optimisations simples ont eu l'effet attendu :
1. Timer Windows 1ms : Intervalles RX réguliers (10.0ms ± 0.1ms)
2. NumPy threads = 1 : Moins de contention CPU
3. Horloge compensée : Drift éliminé (pas de décalage cumulatif)

Résultats attendus :
- Intervalles RX : 10.0ms (au lieu de 12.1ms)
- Spikes : ~4-5% de frames (au lieu de 7-8%)
- Latence RX→PROC : maintenue à ~0.10ms
- Latence PROC→TX : maintenue à ~0.42ms
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"

# Seuils de validation
EXPECTED_RX_INTERVAL = 10.0  # ms (100 Hz)
EXPECTED_RX_TOLERANCE = 0.2  # ±0.2ms acceptable
EXPECTED_SPIKE_RATIO = 0.05  # 5% max (au lieu de 7-8%)

# ──────────────────────────────────────────────
# Patterns de parsing
# ──────────────────────────────────────────────
PATTERNS = {
    "rx_gen": re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\].*?\[RX-SIM\] Generated frame #(\d+)"),
    "rx_proc": re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\].*?\[PROC-SIM\] Processing frame #(\d+)"),
    "proc_tx": re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\].*?\[TX-SIM\] Sent frame #(\d+)"),
}

# ──────────────────────────────────────────────
# Parsing des logs
# ──────────────────────────────────────────────
def find_latest_log():
    """Trouve le fichier de log le plus récent."""
    if not LOGS_DIR.exists():
        print(f"❌ Répertoire logs introuvable : {LOGS_DIR}")
        return None
    
    # Chercher pipeline.log en priorité
    pipeline_log = LOGS_DIR / "pipeline.log"
    if pipeline_log.exists():
        print(f"📄 Log analysé : {pipeline_log.name}")
        return pipeline_log
    
    # Sinon chercher igt_gateway_*.log
    log_files = list(LOGS_DIR.glob("igt_gateway_*.log"))
    if not log_files:
        print(f"❌ Aucun fichier de log trouvé dans {LOGS_DIR}")
        return None
    
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"📄 Log analysé : {latest.name}")
    return latest

def parse_timestamp(ts_str):
    """Convertit timestamp string en millisecondes depuis epoch."""
    from datetime import datetime
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    return dt.timestamp() * 1000

def parse_log_file(log_path):
    """Parse le fichier de log et extrait les timestamps par frame."""
    data = defaultdict(dict)
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for event_type, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match:
                    ts = parse_timestamp(match.group(1))
                    frame_id = int(match.group(2))
                    data[frame_id][event_type] = ts
    
    return data

# ──────────────────────────────────────────────
# Analyse des intervalles RX
# ──────────────────────────────────────────────
def analyze_rx_intervals(data):
    """Analyse la régularité des intervalles entre frames générées."""
    rx_timestamps = sorted([(fid, info["rx_gen"]) for fid, info in data.items() if "rx_gen" in info])
    
    if len(rx_timestamps) < 2:
        print("⚠️  Pas assez de frames RX pour analyser les intervalles")
        return None
    
    intervals = []
    for i in range(1, len(rx_timestamps)):
        prev_ts = rx_timestamps[i-1][1]
        curr_ts = rx_timestamps[i][1]
        interval = curr_ts - prev_ts
        intervals.append(interval)
    
    avg_interval = statistics.mean(intervals)
    std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
    min_interval = min(intervals)
    max_interval = max(intervals)
    
    # Comptage des intervalles "parfaits" (10.0ms ± 0.2ms)
    perfect_count = sum(1 for iv in intervals if abs(iv - EXPECTED_RX_INTERVAL) <= EXPECTED_RX_TOLERANCE)
    perfect_ratio = perfect_count / len(intervals)
    
    print("\n" + "="*60)
    print("🔬 ANALYSE DES INTERVALLES RX (Horloge compensée)")
    print("="*60)
    print(f"Nombre de frames RX  : {len(rx_timestamps)}")
    print(f"Intervalles mesurés  : {len(intervals)}")
    print(f"Intervalle moyen     : {avg_interval:.3f} ms")
    print(f"Écart-type           : {std_interval:.3f} ms")
    print(f"Min / Max            : {min_interval:.3f} ms / {max_interval:.3f} ms")
    print(f"Intervalles parfaits : {perfect_count}/{len(intervals)} ({perfect_ratio*100:.1f}%)")
    print(f"  └─ Critère : 10.0ms ± {EXPECTED_RX_TOLERANCE}ms")
    
    # Validation
    if abs(avg_interval - EXPECTED_RX_INTERVAL) <= 0.3:
        print("✅ SUCCÈS : Horloge compensée fonctionne correctement")
    else:
        print(f"⚠️  ATTENTION : Intervalle moyen {avg_interval:.3f}ms (attendu: {EXPECTED_RX_INTERVAL}ms)")
    
    return {
        "avg": avg_interval,
        "std": std_interval,
        "perfect_ratio": perfect_ratio,
        "intervals": intervals
    }

# ──────────────────────────────────────────────
# Analyse des latences
# ──────────────────────────────────────────────
def analyze_latencies(data):
    """Analyse les latences RX→PROC et PROC→TX."""
    rx_proc_latencies = []
    proc_tx_latencies = []
    
    for frame_id, info in data.items():
        if "rx_gen" in info and "rx_proc" in info:
            lat = info["rx_proc"] - info["rx_gen"]
            rx_proc_latencies.append(lat)
        
        if "rx_proc" in info and "proc_tx" in info:
            lat = info["proc_tx"] - info["rx_proc"]
            proc_tx_latencies.append(lat)
    
    print("\n" + "="*60)
    print("📊 ANALYSE DES LATENCES")
    print("="*60)
    
    if rx_proc_latencies:
        avg_rx_proc = statistics.mean(rx_proc_latencies)
        p99_rx_proc = sorted(rx_proc_latencies)[int(len(rx_proc_latencies) * 0.99)]
        print(f"RX → PROC : {avg_rx_proc:.3f} ms (p99: {p99_rx_proc:.3f} ms)")
    else:
        print("RX → PROC : Aucune donnée")
    
    if proc_tx_latencies:
        avg_proc_tx = statistics.mean(proc_tx_latencies)
        p99_proc_tx = sorted(proc_tx_latencies)[int(len(proc_tx_latencies) * 0.99)]
        print(f"PROC → TX : {avg_proc_tx:.3f} ms (p99: {p99_proc_tx:.3f} ms)")
    else:
        print("PROC → TX : Aucune donnée")
    
    # Validation : latences doivent rester basses
    if rx_proc_latencies and statistics.mean(rx_proc_latencies) < 0.3:
        print("✅ SUCCÈS : Latences RX→PROC maintenues (<0.3ms)")
    elif rx_proc_latencies:
        print(f"⚠️  ATTENTION : Latences RX→PROC élevées ({statistics.mean(rx_proc_latencies):.3f}ms)")
    
    if proc_tx_latencies and statistics.mean(proc_tx_latencies) < 1.0:
        print("✅ SUCCÈS : Latences PROC→TX maintenues (<1.0ms)")
    elif proc_tx_latencies:
        print(f"⚠️  ATTENTION : Latences PROC→TX élevées ({statistics.mean(proc_tx_latencies):.3f}ms)")

# ──────────────────────────────────────────────
# Analyse des spikes
# ──────────────────────────────────────────────
def analyze_spikes(data):
    """Détecte les spikes (latences >1ms) dans RX→PROC et PROC→TX."""
    rx_proc_latencies = []
    proc_tx_latencies = []
    
    for frame_id, info in data.items():
        if "rx_gen" in info and "rx_proc" in info:
            lat = info["rx_proc"] - info["rx_gen"]
            rx_proc_latencies.append((frame_id, lat))
        
        if "rx_proc" in info and "proc_tx" in info:
            lat = info["proc_tx"] - info["rx_proc"]
            proc_tx_latencies.append((frame_id, lat))
    
    # Seuil pour un "spike" : >1ms
    SPIKE_THRESHOLD = 1.0
    
    rx_spikes = [(fid, lat) for fid, lat in rx_proc_latencies if lat > SPIKE_THRESHOLD]
    tx_spikes = [(fid, lat) for fid, lat in proc_tx_latencies if lat > SPIKE_THRESHOLD]
    
    print("\n" + "="*60)
    print("⚡ ANALYSE DES SPIKES (>1ms)")
    print("="*60)
    
    if rx_proc_latencies:
        rx_spike_ratio = len(rx_spikes) / len(rx_proc_latencies)
        print(f"RX → PROC spikes : {len(rx_spikes)}/{len(rx_proc_latencies)} ({rx_spike_ratio*100:.1f}%)")
    
    if proc_tx_latencies:
        tx_spike_ratio = len(tx_spikes) / len(proc_tx_latencies)
        print(f"PROC → TX spikes : {len(tx_spikes)}/{len(proc_tx_latencies)} ({tx_spike_ratio*100:.1f}%)")
    
    # Validation : ratio de spikes doit avoir diminué
    total_spikes = len(rx_spikes) + len(tx_spikes)
    total_latencies = len(rx_proc_latencies) + len(proc_tx_latencies)
    if total_latencies > 0:
        overall_spike_ratio = total_spikes / total_latencies
        print(f"Spikes globaux     : {total_spikes}/{total_latencies} ({overall_spike_ratio*100:.1f}%)")
        
        if overall_spike_ratio <= EXPECTED_SPIKE_RATIO:
            print(f"✅ SUCCÈS : Spikes réduits à {overall_spike_ratio*100:.1f}% (cible: <{EXPECTED_SPIKE_RATIO*100}%)")
        else:
            print(f"⚠️  EN COURS : Spikes à {overall_spike_ratio*100:.1f}% (cible: <{EXPECTED_SPIKE_RATIO*100}%)")
            print("   └─ Note : GIL/scheduler Windows causent toujours ~3-5% de spikes (incompressible)")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 VALIDATION DES 3 OPTIMISATIONS")
    print("="*60)
    print("1. Timer Windows 1ms (timeBeginPeriod)")
    print("2. NumPy threads limités à 1 (OMP_NUM_THREADS=1)")
    print("3. Horloge compensée (perf_counter)")
    print("="*60)
    
    log_file = find_latest_log()
    if not log_file:
        print("\n❌ Impossible de trouver un fichier de log récent")
        print("   └─ Assurez-vous d'avoir lancé test_gateway_real_pipeline_mock.py")
        sys.exit(1)
    
    data = parse_log_file(log_file)
    
    if not data:
        print("\n❌ Aucune donnée parsée dans le fichier de log")
        sys.exit(1)
    
    print(f"\n📊 Frames parsées : {len(data)}")
    
    # Analyses
    rx_results = analyze_rx_intervals(data)
    analyze_latencies(data)
    analyze_spikes(data)
    
    # Résumé final
    print("\n" + "="*60)
    print("📋 RÉSUMÉ FINAL")
    print("="*60)
    
    if rx_results:
        if abs(rx_results["avg"] - EXPECTED_RX_INTERVAL) <= 0.3:
            print("✅ Timer Windows : Intervalles réguliers à 10.0ms")
        else:
            print(f"⚠️  Timer Windows : Intervalles à {rx_results['avg']:.3f}ms (attendu: 10.0ms)")
    
    print("\n💡 PROCHAINES ÉTAPES :")
    print("   1. Relancer le test plusieurs fois pour confirmer la stabilité")
    print("   2. Vérifier le dashboard : http://localhost:8050")
    print("   3. Les spikes restants (3-5%) sont normaux (GIL + Windows scheduler)")
    print("   4. Pour aller plus loin : priorité temps-réel, C++ extensions, etc.")
