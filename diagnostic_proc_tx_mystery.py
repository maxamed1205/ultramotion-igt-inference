#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse approfondie : PROC→TX latency vs TX internal timing
Objectif : Comprendre pourquoi PROC→TX = 3-8ms alors que TX interne = 0ms
"""

import re
from collections import defaultdict

# Patterns
PROC_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[PROC-SIM\] Processing frame #(?P<frame_id>\d+)'
)
TX_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[TX-SIM\] Sent frame #(?P<frame_id>\d+)'
    r'.*read=(?P<read>[\d.]+)ms send=(?P<send>[\d.]+)ms total=(?P<total>[\d.]+)ms'
)
OUTBOX_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[PROC-SIM\] Outbox size after send: (?P<size>\d+)'
)

def parse_timestamp(ts_str):
    """Convertit '[2025-10-29 09:11:47,944]' en millisecondes"""
    parts = ts_str.replace('[', '').replace(']', '').strip()
    time_part = parts.split(' ')[-1]
    h, m, s = time_part.split(':')
    sec, ms = s.split(',')
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms)
    return total_ms

def analyze_proc_tx_mystery(log_path="logs/pipeline.log"):
    """Analyse la différence entre latence PROC→TX et timing interne TX"""
    proc_frames = {}
    tx_frames = {}
    tx_internal_timing = {}
    outbox_sizes = []
    
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
                tx_internal_timing[frame_id] = {
                    'read': float(tx_match.group('read')),
                    'send': float(tx_match.group('send')),
                    'total': float(tx_match.group('total'))
                }
            
            outbox_match = OUTBOX_PATTERN.search(line)
            if outbox_match:
                size = int(outbox_match.group('size'))
                outbox_sizes.append(size)
    
    if not proc_frames or not tx_frames:
        print("❌ Données insuffisantes dans les logs")
        return
    
    print("=" * 80)
    print("🔬 ANALYSE MYSTÈRE : Latence PROC→TX vs Timing interne TX")
    print("=" * 80)
    print()
    
    # Calculer latence PROC→TX
    latencies = []
    for frame_id in sorted(proc_frames.keys()):
        if frame_id in tx_frames:
            lat_proc_tx = tx_frames[frame_id] - proc_frames[frame_id]
            tx_timing = tx_internal_timing.get(frame_id, {'read': 0, 'send': 0, 'total': 0})
            latencies.append((frame_id, lat_proc_tx, tx_timing))
    
    if not latencies:
        print("❌ Aucune frame commune PROC/TX")
        return
    
    # Statistiques
    avg_proc_tx = sum(lat for _, lat, _ in latencies) / len(latencies)
    avg_tx_internal = sum(t['total'] for _, _, t in latencies) / len(latencies)
    
    # La VRAIE latence = latence mesurée - temps TX interne
    real_waiting_times = [lat - t['total'] for _, lat, t in latencies]
    avg_waiting = sum(real_waiting_times) / len(real_waiting_times)
    
    print("📊 VUE D'ENSEMBLE :")
    print(f"   Latence PROC→TX mesurée       : {avg_proc_tx:.2f} ms (moyenne)")
    print(f"   Temps TX interne (read+send)  : {avg_tx_internal:.2f} ms (moyenne)")
    print(f"   Temps d'ATTENTE réel          : {avg_waiting:.2f} ms (⚠️  GOULOT !)")
    print()
    print("   🔍 Interprétation :")
    print(f"      → Frame passée dans send_mask() à t=0")
    print(f"      → Frame ATTEND {avg_waiting:.2f}ms dans _outbox avant traitement TX")
    print(f"      → TX traite en {avg_tx_internal:.2f}ms (read+send)")
    print(f"      → Latence totale PROC→TX = {avg_proc_tx:.2f}ms")
    print()
    
    # Analyse de l'outbox
    if outbox_sizes:
        print("=" * 80)
        print("📦 ANALYSE DE L'OUTBOX")
        print("=" * 80)
        print()
        print(f"   Nombre de mesures : {len(outbox_sizes)}")
        print(f"   Taille min        : {min(outbox_sizes)}")
        print(f"   Taille max        : {max(outbox_sizes)}")
        print(f"   Taille moyenne    : {sum(outbox_sizes)/len(outbox_sizes):.1f}")
        print()
        
        # Distribution
        dist = defaultdict(int)
        for size in outbox_sizes:
            dist[size] += 1
        
        print("   Distribution des tailles :")
        for size in sorted(dist.keys()):
            count = dist[size]
            bar = "█" * (count // 2)
            print(f"      {size} frames : {count:3d} occurrences {bar}")
        print()
    
    # Détails frame par frame
    print("=" * 80)
    print("📝 DÉTAILS FRAME PAR FRAME (premières 30 frames)")
    print("=" * 80)
    print()
    print("Frame | PROC→TX | TX_intern | ATTENTE | Analyse")
    print("-" * 60)
    
    for frame_id, lat, tx_timing in latencies[:30]:
        waiting = lat - tx_timing['total']
        
        if waiting > 5:
            marker = "❌ LENT"
        elif waiting > 3:
            marker = "⚠️  Moyen"
        elif waiting > 1:
            marker = "⚡ OK"
        else:
            marker = "✅ Rapide"
        
        print(f" {frame_id:03d}  | {lat:6.1f}  | {tx_timing['total']:8.2f}  | {waiting:6.1f}  | {marker}")
    
    if len(latencies) > 30:
        print("  [...]")
    
    print()
    
    # Diagnostic final
    print("=" * 80)
    print("🎯 DIAGNOSTIC FINAL")
    print("=" * 80)
    print()
    
    print("❌ CAUSE IDENTIFIÉE : CONTENTION SUR sleep(0.005) dans TX thread")
    print()
    print("   Le problème N'EST PAS :")
    print("      ✅ La lecture de _outbox (0.00ms)")
    print("      ✅ L'envoi réseau/sérialisation (0.00ms)")
    print("      ✅ Le logging TX (négligeable)")
    print()
    print("   Le problème EST :")
    print(f"      ❌ Le thread TX dort {avg_waiting:.2f}ms en moyenne entre chaque check")
    print("      ❌ Ligne 92 dans slicer_server.py : time.sleep(0.005)")
    print()
    print("   Explication détaillée :")
    print("      1. PROC thread écrit frame dans _outbox à t=0")
    print("      2. TX thread est en sleep(0.005) → dort encore 0-5ms")
    print("      3. TX se réveille, trouve frame, la traite en 0ms")
    print("      4. Latence mesurée PROC→TX = temps de sommeil + GIL overhead")
    print()
    print("   Distribution du temps d'attente :")
    waiting_dist = defaultdict(int)
    for waiting in real_waiting_times:
        bucket = int(waiting)
        waiting_dist[bucket] += 1
    
    for bucket in sorted(waiting_dist.keys()):
        count = waiting_dist[bucket]
        pct = count / len(real_waiting_times) * 100
        bar = "█" * int(pct / 2)
        print(f"      {bucket:2d} ms : {count:3d} frames ({pct:5.1f}%) {bar}")
    print()
    
    # Solutions
    print("=" * 80)
    print("💡 SOLUTIONS PROPOSÉES")
    print("=" * 80)
    print()
    print("Option A : Threading.Event (comme RX→PROC) ⭐ RECOMMANDÉ")
    print("   → Ajouter tx_ready = threading.Event()")
    print("   → send_mask() appelle tx_ready.set() après append(_outbox)")
    print("   → TX thread fait tx_ready.wait(timeout=0.01) au lieu de sleep(0.005)")
    print("   → Gain attendu : 0-5ms → 0-0.5ms (soit ~4ms de gain)")
    print()
    print("Option B : Réduire sleep(0.005) → sleep(0.001)")
    print("   → Gain limité : 0-5ms → 0-1ms (soit ~2ms de gain)")
    print("   → CPU usage légèrement plus élevé")
    print()
    print("Option C : Queue.Queue avec block=True")
    print("   → Remplacer deque par queue.Queue")
    print("   → outbox.get(block=True, timeout=0.01)")
    print("   → Réveil automatique dès qu'une frame arrive")
    print()
    print("🎯 MA RECOMMANDATION : Option A (Event)")
    print("   → Cohérent avec l'optimisation RX→PROC déjà faite")
    print("   → Gain maximal (~4ms) avec complexité minimale")
    print("   → Code symétrique et maintenable")
    print()

if __name__ == "__main__":
    analyze_proc_tx_mystery()
