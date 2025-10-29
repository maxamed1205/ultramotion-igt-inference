#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic des pics RX→PROC à 2ms malgré l'optimisation Event
Objectif : Identifier POURQUOI certaines frames ont encore 1-2ms de latence
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
    parts = ts_str.replace('[', '').replace(']', '').strip()
    time_part = parts.split(' ')[-1]  # '09:11:47,944'
    h, m, s = time_part.split(':')
    sec, ms = s.split(',')
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms)
    return total_ms

def analyze_spikes(log_path="logs/pipeline.log"):
    """Analyse détaillée des pics de latence RX→PROC"""
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
    
    # Calculer les latences et identifier les patterns
    latencies = []
    spike_frames = []  # Frames avec latence >= 1ms
    perfect_frames = []  # Frames avec latence = 0ms
    
    for frame_id in sorted(rx_frames.keys()):
        if frame_id in proc_frames:
            latency_ms = proc_frames[frame_id] - rx_frames[frame_id]
            latencies.append((frame_id, latency_ms))
            
            if latency_ms >= 1:
                spike_frames.append(frame_id)
            else:
                perfect_frames.append(frame_id)
    
    if not latencies:
        print("❌ Aucune donnée trouvée dans les logs")
        return
    
    print("=" * 80)
    print("🔬 DIAGNOSTIC DES PICS RX→PROC (malgré threading.Event)")
    print("=" * 80)
    print()
    
    # 1. Vue d'ensemble
    total = len(latencies)
    spikes = len(spike_frames)
    perfect = len(perfect_frames)
    
    print(f"📊 VUE D'ENSEMBLE ({total} frames analysées) :")
    print(f"   ✅ Frames instantanées (0ms)    : {perfect:3d} ({perfect/total*100:5.1f}%)")
    print(f"   ⚠️  Frames avec latence (≥1ms)  : {spikes:3d} ({spikes/total*100:5.1f}%)")
    print()
    
    # 2. Distribution détaillée
    distribution = defaultdict(int)
    for _, lat in latencies:
        distribution[lat] += 1
    
    print("📈 DISTRIBUTION DES LATENCES :")
    for latency in sorted(distribution.keys()):
        count = distribution[latency]
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        marker = "✅" if latency == 0 else "⚠️ " if latency >= 2 else "⚡"
        print(f"   {marker} {latency:2.0f} ms : {count:3d} frames ({percentage:5.1f}%) {bar}")
    print()
    
    # 3. Analyse des patterns temporels
    print("=" * 80)
    print("🔍 ANALYSE DES PATTERNS (pics >= 1ms)")
    print("=" * 80)
    print()
    
    if not spike_frames:
        print("✅ Aucun pic détecté - performance parfaite !")
        return
    
    # Espacements entre pics
    spike_intervals = []
    for i in range(1, len(spike_frames)):
        interval = spike_frames[i] - spike_frames[i-1]
        spike_intervals.append(interval)
    
    if spike_intervals:
        print(f"📏 ESPACEMENTS ENTRE PICS :")
        print(f"   Min     : {min(spike_intervals):3d} frames")
        print(f"   Max     : {max(spike_intervals):3d} frames")
        print(f"   Moyenne : {sum(spike_intervals)/len(spike_intervals):5.1f} frames")
        print()
        
        # Distribution des intervalles
        interval_dist = defaultdict(int)
        for interval in spike_intervals:
            # Grouper par buckets
            if interval <= 5:
                bucket = interval
            elif interval <= 10:
                bucket = 10
            elif interval <= 20:
                bucket = 20
            elif interval <= 50:
                bucket = 50
            else:
                bucket = 100
            interval_dist[bucket] += 1
        
        print("   Distribution des intervalles :")
        for bucket in sorted(interval_dist.keys()):
            count = interval_dist[bucket]
            label = f"{bucket}" if bucket <= 5 else f"~{bucket}"
            print(f"      {label:>3} frames : {count:2d} occurrences")
        print()
    
    # 4. Liste des frames avec pics (premières et dernières)
    print("🎯 FRAMES AVEC PICS (détails) :")
    print()
    print("   Premières frames avec latence :")
    for frame_id in spike_frames[:10]:
        lat = proc_frames[frame_id] - rx_frames[frame_id]
        print(f"      Frame #{frame_id:03d} : {lat:.0f} ms")
    
    if len(spike_frames) > 20:
        print("      [...]")
        print("   Dernières frames avec latence :")
        for frame_id in spike_frames[-10:]:
            lat = proc_frames[frame_id] - rx_frames[frame_id]
            print(f"      Frame #{frame_id:03d} : {lat:.0f} ms")
    elif len(spike_frames) > 10:
        print("   Autres frames avec latence :")
        for frame_id in spike_frames[10:]:
            lat = proc_frames[frame_id] - rx_frames[frame_id]
            print(f"      Frame #{frame_id:03d} : {lat:.0f} ms")
    print()
    
    # 5. Hypothèses sur les causes
    print("=" * 80)
    print("💡 HYPOTHÈSES SUR LES CAUSES DES PICS")
    print("=" * 80)
    print()
    
    avg_interval = sum(spike_intervals) / len(spike_intervals) if spike_intervals else 0
    
    print("🔬 ANALYSE :")
    print()
    
    # Hypothèse 1 : GIL contention
    if spikes / total < 0.10:  # Moins de 10% de pics
        print("✅ Hypothèse #1 : GIL (Global Interpreter Lock)")
        print("   → Les pics sont RARES (<10%) → Normal pour Python threading")
        print("   → Le GIL peut bloquer PROC pendant 0-2ms quand :")
        print("      • RX thread détient le GIL (création numpy array)")
        print("      • Async logging thread écrit sur disque")
        print("      • Garbage collector s'exécute")
        print("   → Solution : ACCEPTER ces pics (overhead Python incompressible)")
        print()
    
    # Hypothèse 2 : Batch processing
    if avg_interval > 10:
        print("⚠️  Hypothèse #2 : Windows Scheduler (time slicing)")
        print(f"   → Pics espacés en moyenne de {avg_interval:.1f} frames")
        print("   → Windows alloue du CPU aux autres threads tous les ~10-15ms")
        print("   → Quand PROC perd son time slice, wait() peut durer 1-2ms")
        print("   → Solution :")
        print("      • Option A : Augmenter priorité thread PROC (THREAD_PRIORITY_ABOVE_NORMAL)")
        print("      • Option B : Accepter ces pics (scheduler OS incompressible)")
        print()
    
    # Hypothèse 3 : I/O async logging
    io_frames = [fid for fid in spike_frames if fid % 10 == 0]
    if len(io_frames) / len(spike_frames) > 0.3:
        print("⚠️  Hypothèse #3 : Async Logging I/O")
        print(f"   → {len(io_frames)} pics ({len(io_frames)/spikes*100:.0f}%) proches de frames multiples de 10")
        print("   → Le QueueListener flush sur disque peut bloquer le GIL")
        print("   → Solution : Augmenter buffer size du QueueHandler")
        print()
    
    # Hypothèse 4 : Clear tardif
    consecutive_spikes = 0
    for i in range(1, len(spike_frames)):
        if spike_frames[i] - spike_frames[i-1] == 1:
            consecutive_spikes += 1
    
    if consecutive_spikes > 0:
        print("❌ Hypothèse #4 : frame_ready.clear() manquant ou tardif")
        print(f"   → {consecutive_spikes} paires de frames consécutives avec pics")
        print("   → Possible race condition : wait() retourne mais clear() pas encore appelé")
        print("   → Solution : Vérifier que clear() est IMMÉDIATEMENT après wait()")
        print()
    
    # 6. Conclusion
    print("=" * 80)
    print("📋 CONCLUSION")
    print("=" * 80)
    print()
    
    if spikes / total <= 0.08 and consecutive_spikes == 0:
        print("✅ DIAGNOSTIC : Performance EXCELLENTE")
        print()
        print("   Les pics observés (~5-8%) sont NORMAUX pour Python threading :")
        print("   • GIL switching : 0.1-0.5ms overhead incompressible")
        print("   • OS scheduler   : Windows time slicing ~15ms")
        print("   • Async I/O      : QueueListener flush périodique")
        print()
        print("   🎯 RECOMMANDATION : AUCUNE action nécessaire")
        print("      → L'optimisation Event a atteint son maximum (~92% à 0ms)")
        print("      → Les 8% restants sont l'overhead Python/OS incompressible")
        print()
    elif consecutive_spikes > spikes * 0.2:
        print("⚠️  DIAGNOSTIC : Possible problème de synchronisation")
        print()
        print("   🎯 RECOMMANDATION : Vérifier le code :")
        print("      1. clear() est-il appelé IMMÉDIATEMENT après wait() ?")
        print("      2. Y a-t-il du code entre wait() et receive_image() ?")
        print("      3. Le timeout de wait() est-il trop long (>0.01s) ?")
        print()
    else:
        print("⚠️  DIAGNOSTIC : Pics dus au scheduler OS + GIL")
        print()
        print("   🎯 RECOMMANDATION : Augmenter priorité thread PROC")
        print("      → Ajouter dans simulate_processing() :")
        print("         import win32process, win32api")
        print("         handle = win32api.GetCurrentThread()")
        print("         win32process.SetThreadPriority(handle, win32process.THREAD_PRIORITY_ABOVE_NORMAL)")
        print()

if __name__ == "__main__":
    analyze_spikes()
