#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic approfondi des latences PROC→TX
Objectif : Comprendre pourquoi 3-8ms de latence entre Processing et Transmission
"""

import re
from collections import defaultdict

# Patterns pour extraire timestamp et frame_id
PROC_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[PROC-SIM\] Processing frame #(?P<frame_id>\d+)'
)
TX_PATTERN = re.compile(
    r'\[(?P<ts>[\d\-: ,]+)\].*\[TX-SIM\] Sent frame #(?P<frame_id>\d+)'
)

def parse_timestamp(ts_str):
    """Convertit '[2025-10-29 09:11:47,944]' en millisecondes"""
    parts = ts_str.replace('[', '').replace(']', '').strip()
    time_part = parts.split(' ')[-1]  # '09:11:47,944'
    h, m, s = time_part.split(':')
    sec, ms = s.split(',')
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(sec) * 1000 + int(ms)
    return total_ms

def analyze_proc_tx_latency(log_path="logs/pipeline.log"):
    """Analyse détaillée des latences PROC→TX"""
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
    
    # Calculer les latences
    latencies = []
    for frame_id in sorted(proc_frames.keys()):
        if frame_id in tx_frames:
            latency_ms = tx_frames[frame_id] - proc_frames[frame_id]
            latencies.append((frame_id, latency_ms))
    
    if not latencies:
        print("❌ Aucune donnée trouvée dans les logs")
        return
    
    print("=" * 80)
    print("🔬 DIAGNOSTIC APPROFONDI : LATENCES PROC → TX")
    print("=" * 80)
    print()
    
    # 1. Statistiques globales
    lat_values = [lat for _, lat in latencies]
    min_lat = min(lat_values)
    max_lat = max(lat_values)
    avg_lat = sum(lat_values) / len(lat_values)
    median_lat = sorted(lat_values)[len(lat_values) // 2]
    
    print(f"📊 STATISTIQUES GLOBALES ({len(latencies)} frames analysées) :")
    print(f"   Min     : {min_lat:6.2f} ms")
    print(f"   Max     : {max_lat:6.2f} ms  {'⚠️  PIC ÉLEVÉ !' if max_lat >= 7 else ''}")
    print(f"   Moyenne : {avg_lat:6.2f} ms")
    print(f"   Médiane : {median_lat:6.2f} ms")
    print()
    
    # 2. Distribution détaillée
    distribution = defaultdict(int)
    for _, lat in latencies:
        # Grouper par buckets de 1ms
        bucket = int(lat)
        distribution[bucket] += 1
    
    print("📈 DISTRIBUTION PAR BUCKET (1ms) :")
    for bucket in sorted(distribution.keys()):
        count = distribution[bucket]
        percentage = (count / len(latencies)) * 100
        bar = "█" * int(percentage / 2)
        
        # Colorisation
        if bucket <= 2:
            marker = "✅"
            label = "EXCELLENT"
        elif bucket <= 4:
            marker = "⚡"
            label = "BON"
        elif bucket <= 6:
            marker = "⚠️ "
            label = "MOYEN"
        else:
            marker = "❌"
            label = "LENT"
        
        print(f"   {marker} {bucket:2d} ms : {count:3d} frames ({percentage:5.1f}%) {bar} {label}")
    print()
    
    # 3. Analyse des pics (≥6ms)
    print("=" * 80)
    print("🎯 ANALYSE DES PICS (≥ 6ms)")
    print("=" * 80)
    print()
    
    spike_frames = [(fid, lat) for fid, lat in latencies if lat >= 6]
    
    if not spike_frames:
        print("✅ Aucun pic détecté (toutes les frames < 6ms)")
    else:
        print(f"⚠️  {len(spike_frames)} frames avec latence ≥ 6ms :")
        print()
        for fid, lat in spike_frames:
            print(f"      Frame #{fid:03d} : {lat:.0f} ms")
        print()
        
        # Espacements entre pics
        spike_ids = [fid for fid, _ in spike_frames]
        if len(spike_ids) > 1:
            intervals = [spike_ids[i] - spike_ids[i-1] for i in range(1, len(spike_ids))]
            print(f"   📏 Espacements entre pics :")
            print(f"      Min     : {min(intervals):3d} frames")
            print(f"      Max     : {max(intervals):3d} frames")
            print(f"      Moyenne : {sum(intervals)/len(intervals):5.1f} frames")
            print()
    
    # 4. Analyse des oscillations (variabilité)
    print("=" * 80)
    print("📊 ANALYSE DE LA VARIABILITÉ")
    print("=" * 80)
    print()
    
    # Calculer les transitions (frame N → frame N+1)
    transitions = []
    for i in range(1, len(latencies)):
        _, lat_prev = latencies[i-1]
        _, lat_curr = latencies[i]
        delta = abs(lat_curr - lat_prev)
        transitions.append(delta)
    
    if transitions:
        avg_variation = sum(transitions) / len(transitions)
        max_variation = max(transitions)
        
        print(f"   Variation moyenne entre frames successives : {avg_variation:.2f} ms")
        print(f"   Variation maximale                         : {max_variation:.2f} ms")
        print()
        
        # Distribution des variations
        stable_count = sum(1 for d in transitions if d <= 1)
        moderate_count = sum(1 for d in transitions if 1 < d <= 3)
        high_count = sum(1 for d in transitions if d > 3)
        
        print("   Classification des transitions :")
        print(f"      Stables (≤1ms)     : {stable_count:3d} ({stable_count/len(transitions)*100:5.1f}%)")
        print(f"      Modérées (1-3ms)   : {moderate_count:3d} ({moderate_count/len(transitions)*100:5.1f}%)")
        print(f"      Élevées (>3ms)     : {high_count:3d} ({high_count/len(transitions)*100:5.1f}%)")
        print()
    
    # 5. Pattern de périodicité (oscillations)
    print("=" * 80)
    print("🔍 DÉTECTION DE PATTERNS PÉRIODIQUES")
    print("=" * 80)
    print()
    
    # Chercher des patterns répétitifs (ex: haut-bas-haut-bas)
    low_frames = [(fid, lat) for fid, lat in latencies if lat <= 3]
    high_frames = [(fid, lat) for fid, lat in latencies if lat >= 5]
    
    print(f"   Frames RAPIDES (≤3ms) : {len(low_frames):3d} ({len(low_frames)/len(latencies)*100:5.1f}%)")
    print(f"   Frames LENTES (≥5ms)  : {len(high_frames):3d} ({len(high_frames)/len(latencies)*100:5.1f}%)")
    print()
    
    # Alternance rapide/lent ?
    if len(low_frames) > 0 and len(high_frames) > 0:
        # Vérifier si les frames alternent
        sequence = ['L' if lat <= 3 else 'H' for _, lat in latencies]
        sequence_str = ''.join(sequence[:50])  # Premiers 50 frames
        
        print(f"   Pattern des 50 premières frames (L=≤3ms, H=≥5ms) :")
        print(f"      {sequence_str}")
        print()
        
        # Compter les alternances
        alternations = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
        alternation_rate = alternations / (len(sequence) - 1) * 100
        
        print(f"   Taux d'alternance : {alternation_rate:.1f}%")
        if alternation_rate > 60:
            print("      → ⚠️  OSCILLATIONS FRÉQUENTES détectées !")
        elif alternation_rate > 40:
            print("      → ⚡ Oscillations modérées")
        else:
            print("      → ✅ Comportement stable")
        print()
    
    # 6. Hypothèses sur les causes
    print("=" * 80)
    print("💡 HYPOTHÈSES SUR LES CAUSES DES LATENCES PROC→TX")
    print("=" * 80)
    print()
    
    print("🔬 ANALYSE DES COMPOSANTS IMPLIQUÉS :")
    print()
    print("   PROC thread (après wait()) :")
    print("      1. receive_image()       : Lecture depuis _mailbox (deque)")
    print("      2. mask = img > 128      : Seuillage numpy (~0.1-0.5ms)")
    print("      3. send_mask()           : Écriture dans _outbox (deque)")
    print("      4. LOG.info()            : Async logging (queued, ~0.1ms)")
    print()
    print("   TX thread (run_slicer_server) :")
    print("      1. Lecture depuis _outbox")
    print("      2. Sérialisation + encodage")
    print("      3. Envoi réseau via OpenIGTLink")
    print("      4. LOG.info() si envoyé")
    print()
    
    # Diagnostic basé sur les observations
    print("=" * 80)
    print("🎯 DIAGNOSTIC PROBABLE")
    print("=" * 80)
    print()
    
    if avg_lat >= 4 and max_lat >= 7:
        print("❌ Hypothèse #1 : GOULOT D'ÉTRANGLEMENT dans _outbox")
        print()
        print("   Observations :")
        print(f"      • Latence moyenne élevée : {avg_lat:.1f}ms")
        print(f"      • Pics fréquents : {len(spike_frames)} frames ≥ 6ms")
        print(f"      • Variabilité : {avg_variation:.2f}ms entre frames")
        print()
        print("   Cause probable :")
        print("      → Le thread TX (run_slicer_server) est LENT")
        print("      → send_mask() écrit dans _outbox mais TX ne consomme pas assez vite")
        print("      → _outbox s'accumule → contention → latence variable")
        print()
        print("   Preuves à chercher :")
        print("      1. Taille de _outbox au fil du temps (devrait osciller)")
        print("      2. Temps réel d'envoi réseau dans run_slicer_server()")
        print("      3. Blocking I/O dans l'envoi OpenIGTLink")
        print()
        print("   Solutions possibles :")
        print("      A. Instrumenter run_slicer_server() pour mesurer :")
        print("         - Temps de lecture _outbox")
        print("         - Temps de sérialisation")
        print("         - Temps d'envoi réseau")
        print()
        print("      B. Vérifier si _outbox.maxlen=10 est trop petit")
        print("         → Si TX est lent, frames s'accumulent et sont droppées")
        print()
        print("      C. Optimiser l'envoi TX :")
        print("         - Batching (envoyer plusieurs frames à la fois)")
        print("         - Compression (réduire payload réseau)")
        print("         - Async I/O (non-blocking socket)")
        print()
    
    # 7. Recommandations
    print("=" * 80)
    print("📋 PROCHAINES ÉTAPES - PLAN D'INVESTIGATION")
    print("=" * 80)
    print()
    print("ÉTAPE 1 : Instrumenter run_slicer_server() pour mesurer :")
    print("   → Ajouter des timestamps détaillés dans TX thread")
    print("   → Mesurer : t_read_outbox, t_serialize, t_network_send")
    print()
    print("ÉTAPE 2 : Logger la taille de _outbox en temps réel :")
    print("   → Ajouter LOG.debug(f'Outbox size: {len(gateway._outbox)}') dans PROC")
    print("   → Vérifier si accumulation → si oui, TX est le bottleneck")
    print()
    print("ÉTAPE 3 : Analyser les logs TX pour identifier le composant lent :")
    print("   → Si t_network_send > 5ms : problème réseau/socket")
    print("   → Si t_serialize > 2ms : sérialisation trop lourde")
    print("   → Si t_read_outbox > 1ms : contention sur deque")
    print()
    print("ÉTAPE 4 : Optimisations ciblées selon résultats :")
    print("   → Réseau lent : Passer à asyncio ou threads non-blocking")
    print("   → Sérialisation : Optimiser encodage ou compresser")
    print("   → Contention : Augmenter _outbox.maxlen ou utiliser queue.Queue")
    print()
    
    # 8. Output détaillé pour debugging
    print("=" * 80)
    print("📝 DONNÉES DÉTAILLÉES (pour analyse manuelle)")
    print("=" * 80)
    print()
    print("Frame ID | PROC→TX (ms)")
    print("-" * 25)
    for fid, lat in latencies[:30]:  # Premières 30 frames
        marker = "⚠️ " if lat >= 6 else "  "
        print(f"{marker} {fid:03d}     | {lat:6.1f}")
    
    if len(latencies) > 30:
        print("   [...]")
        print()
        print("Dernières frames :")
        for fid, lat in latencies[-10:]:
            marker = "⚠️ " if lat >= 6 else "  "
            print(f"{marker} {fid:03d}     | {lat:6.1f}")
    print()

if __name__ == "__main__":
    analyze_proc_tx_latency()
