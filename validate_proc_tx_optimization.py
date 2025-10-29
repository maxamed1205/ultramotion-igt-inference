#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation finale de l'optimisation Option A pour PROC→TX
Compare les latences AVANT vs APRÈS Event-based signaling
"""

import re
from collections import defaultdict

# Pattern pour extraire timestamp et frame_id
PROC_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[PROC-SIM\] Processing frame #(?P<frame_id>\d+)'
)
TX_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[TX-SIM\] Sent frame #(?P<frame_id>\d+)'
)

def parse_timestamp(ts_str):
    """Convertit '[2025-10-29 09:11:47,944]' en millisecondes"""
    parts = ts_str.replace('[', '').replace(']', '').strip()
    time_part = parts.split(' ')[-1]
    h, m, s = time_part.split(':')
    sec, ms = s.split(',')
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms)
    return total_ms

def analyze_latencies(log_path="logs/pipeline.log"):
    """Analyse les latences PROC→TX depuis le fichier log"""
    proc_frames = {}
    tx_frames = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            proc_match = PROC_PATTERN.search(line)
            if proc_match:
                frame_id = int(proc_match.group('frame_id'))
                ts_ms = parse_timestamp(proc_match.group('ts'))
                proc_frames[frame_id] = ts_ms
            
            tx_match = TX_PATTERN.search(line)
            if tx_match:
                frame_id = int(tx_match.group('frame_id'))
                ts_ms = parse_timestamp(tx_match.group('ts'))
                tx_frames[frame_id] = ts_ms
    
    # Calculer les latences PROC→TX
    latencies = []
    for frame_id in sorted(proc_frames.keys()):
        if frame_id in tx_frames:
            latency_ms = tx_frames[frame_id] - proc_frames[frame_id]
            latencies.append(latency_ms)
    
    if not latencies:
        print("❌ Aucune donnée trouvée dans les logs")
        return
    
    # Statistiques
    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_lat = sum(latencies) / len(latencies)
    median_lat = sorted(latencies)[len(latencies) // 2]
    
    # Distribution
    distribution = defaultdict(int)
    for lat in latencies:
        distribution[lat] += 1
    
    print("=" * 80)
    print("✅ VALIDATION DE L'OPTIMISATION - Option A (Event PROC→TX)")
    print("=" * 80)
    print()
    print(f"📊 Frames analysées : {len(latencies)}")
    print()
    print("🔍 LATENCES PROC → TX (après optimisation Event) :")
    print(f"   Min     : {min_lat:6.2f} ms")
    print(f"   Max     : {max_lat:6.2f} ms")
    print(f"   Moyenne : {avg_lat:6.2f} ms")
    print(f"   Médiane : {median_lat:6.2f} ms")
    print()
    print("📈 DISTRIBUTION :")
    for latency in sorted(distribution.keys()):
        count = distribution[latency]
        percentage = (count / len(latencies)) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {latency:4.0f} ms : {count:3d} frames ({percentage:5.1f}%) {bar}")
    print()
    
    # Comparaison avec objectifs
    print("=" * 80)
    print("🎯 COMPARAISON AVEC OBJECTIFS")
    print("=" * 80)
    print()
    print("AVANT (baseline avec sleep(0.005)) :")
    print("   - Moyenne : 4.01 ms")
    print("   - 0 ms    : ~1% des frames")
    print("   - 1-2 ms  : ~30% des frames")
    print("   - 3-5 ms  : ~60% des frames")
    print("   - ≥6 ms   : ~10% des frames (pics)")
    print()
    print("OBJECTIF (après Event) :")
    print("   - Moyenne : 0.20-0.50 ms")
    print("   - 0 ms    : 60-80% des frames")
    print("   - 1 ms    : 15-30% des frames")
    print()
    print("RÉSULTAT ACTUEL :")
    zero_ms_pct = (distribution[0] / len(latencies)) * 100 if 0 in distribution else 0
    one_ms_pct = (distribution[1] / len(latencies)) * 100 if 1 in distribution else 0
    print(f"   - Moyenne : {avg_lat:.2f} ms")
    print(f"   - 0 ms    : {zero_ms_pct:.1f}% des frames")
    print(f"   - 1 ms    : {one_ms_pct:.1f}% des frames")
    print()
    
    # Verdict
    print("=" * 80)
    if avg_lat <= 0.50 and zero_ms_pct >= 60:
        print("✅ OPTIMISATION RÉUSSIE !")
        print(f"   → Gain de latence : {4.01 - avg_lat:.2f} ms (-{((4.01-avg_lat)/4.01)*100:.0f}%)")
        print(f"   → Frames instantanées : +{zero_ms_pct - 1:.0f}% (de ~1% → {zero_ms_pct:.0f}%)")
    elif avg_lat <= 0.50:
        print("✅ OPTIMISATION PARTIELLEMENT RÉUSSIE")
        print(f"   → Moyenne atteinte ({avg_lat:.2f} ms ≤ 0.50 ms)")
        print(f"   → Mais frames instantanées sous objectif ({zero_ms_pct:.0f}% < 60%)")
    else:
        print("⚠️  OBJECTIF NON ATTEINT")
        print(f"   → Moyenne encore élevée ({avg_lat:.2f} ms > 0.50 ms)")
    print("=" * 80)

if __name__ == "__main__":
    analyze_latencies()
