#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation de l'optimisation Option A (Event + Logs déplacés)
Compare les latences AVANT vs APRÈS l'implémentation de threading.Event
"""

import re
from collections import defaultdict

# Pattern pour extraire timestamp et frame_id des logs
RX_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[RX-SIM\] Generated frame #(?P<frame_id>\d+)'
)
PROC_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[PROC-SIM\] Processing frame #(?P<frame_id>\d+)'
)

def parse_timestamp(ts_str):
    """Convertit '[2025-10-29 09:11:47,944]' en millisecondes"""
    # Format: [2025-10-29 09:11:47,944]
    parts = ts_str.replace('[', '').replace(']', '').strip()
    # '2025-10-29 09:11:47,944' → prendre les secondes et millisecondes
    time_part = parts.split(' ')[-1]  # '09:11:47,944'
    h, m, s = time_part.split(':')
    sec, ms = s.split(',')
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms)
    return total_ms

def analyze_latencies(log_path="logs/pipeline.log"):
    """Analyse les latences RX→PROC depuis le fichier log"""
    rx_frames = {}
    proc_frames = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            rx_match = RX_PATTERN.search(line)
            if rx_match:
                frame_id = int(rx_match.group('frame_id'))
                ts_ms = parse_timestamp(rx_match.group('ts'))
                rx_frames[frame_id] = ts_ms
            
            proc_match = PROC_PATTERN.search(line)
            if proc_match:
                frame_id = int(proc_match.group('frame_id'))
                ts_ms = parse_timestamp(proc_match.group('ts'))
                proc_frames[frame_id] = ts_ms
    
    # Calculer les latences RX→PROC
    latencies = []
    for frame_id in sorted(rx_frames.keys()):
        if frame_id in proc_frames:
            latency_ms = proc_frames[frame_id] - rx_frames[frame_id]
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
    print("✅ VALIDATION DE L'OPTIMISATION - Option A (Event + Logs déplacés)")
    print("=" * 80)
    print()
    print(f"📊 Frames analysées : {len(latencies)}")
    print()
    print("🔍 LATENCES RX → PROC (après optimisation) :")
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
    print("AVANT (baseline) :")
    print("   - Moyenne : 0.87 ms")
    print("   - 0 ms    : 18% des frames")
    print("   - 1 ms    : 77% des frames")
    print("   - 2 ms    : 5% des frames")
    print()
    print("OBJECTIF (après Event) :")
    print("   - Moyenne : 0.20-0.50 ms")
    print("   - 0 ms    : 60-70% des frames")
    print("   - 1 ms    : 30-40% des frames")
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
        print(f"   → Gain de latence : {0.87 - avg_lat:.2f} ms (-{((0.87-avg_lat)/0.87)*100:.0f}%)")
        print(f"   → Frames instantanées : +{zero_ms_pct - 18:.0f}% (de 18% → {zero_ms_pct:.0f}%)")
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
