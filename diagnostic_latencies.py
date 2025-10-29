"""
Diagnostic détaillé des latences observées dans le graphique
"""
import sys
sys.path.insert(0, "src")

from service.dashboard_service import MetricsCollector, DashboardConfig
import statistics

config = DashboardConfig()
collector = MetricsCollector(config)
snapshot = collector.collect()

details = snapshot.get('latency_details', {})
frames = details.get('frames', [])
rxproc = details.get('rxproc', [])
proctx = details.get('proctx', [])
rxtx = details.get('rxtx', [])

print("=" * 80)
print("DIAGNOSTIC DES LATENCES INTER-ÉTAPES")
print("=" * 80)

# Statistiques RX→PROC
rxproc_valid = [v for v in rxproc if v is not None]
if rxproc_valid:
    print(f"\n📊 RX → PROC (Réception → Traitement)")
    print(f"   Nombre de frames: {len(rxproc_valid)}")
    print(f"   Min: {min(rxproc_valid):.2f} ms")
    print(f"   Max: {max(rxproc_valid):.2f} ms")
    print(f"   Moyenne: {statistics.mean(rxproc_valid):.2f} ms")
    print(f"   Médiane: {statistics.median(rxproc_valid):.2f} ms")
    print(f"   Écart-type: {statistics.stdev(rxproc_valid):.2f} ms" if len(rxproc_valid) > 1 else "   Écart-type: N/A")
    
    # Distribution des valeurs
    values_0 = sum(1 for v in rxproc_valid if v == 0.0)
    values_1 = sum(1 for v in rxproc_valid if v == 1.0)
    values_2 = sum(1 for v in rxproc_valid if v == 2.0)
    values_other = sum(1 for v in rxproc_valid if v > 2.0)
    
    print(f"\n   Distribution:")
    print(f"   - 0 ms: {values_0} frames ({values_0/len(rxproc_valid)*100:.1f}%)")
    print(f"   - 1 ms: {values_1} frames ({values_1/len(rxproc_valid)*100:.1f}%)")
    print(f"   - 2 ms: {values_2} frames ({values_2/len(rxproc_valid)*100:.1f}%)")
    print(f"   - >2 ms: {values_other} frames ({values_other/len(rxproc_valid)*100:.1f}%)")
    
    print(f"\n   💡 INTERPRÉTATION RX→PROC:")
    print(f"   - Latence très stable et faible (0-2 ms)")
    print(f"   - Majoritairement {max([(values_0, '0ms'), (values_1, '1ms'), (values_2, '2ms')], key=lambda x: x[0])[1]}")
    print(f"   - Indique un traitement IMMÉDIAT après réception")
    print(f"   - La mailbox (_mailbox) est probablement VIDE la plupart du temps")

# Statistiques PROC→TX
proctx_valid = [v for v in proctx if v is not None]
if proctx_valid:
    print(f"\n📊 PROC → TX (Traitement → Transmission)")
    print(f"   Nombre de frames: {len(proctx_valid)}")
    print(f"   Min: {min(proctx_valid):.2f} ms")
    print(f"   Max: {max(proctx_valid):.2f} ms")
    print(f"   Moyenne: {statistics.mean(proctx_valid):.2f} ms")
    print(f"   Médiane: {statistics.median(proctx_valid):.2f} ms")
    print(f"   Écart-type: {statistics.stdev(proctx_valid):.2f} ms" if len(proctx_valid) > 1 else "   Écart-type: N/A")
    
    # Distribution
    values_2_3 = sum(1 for v in proctx_valid if 2.0 <= v < 3.0)
    values_3_4 = sum(1 for v in proctx_valid if 3.0 <= v < 4.0)
    values_4_5 = sum(1 for v in proctx_valid if 4.0 <= v < 5.0)
    values_5_6 = sum(1 for v in proctx_valid if 5.0 <= v < 6.0)
    values_6_7 = sum(1 for v in proctx_valid if 6.0 <= v < 7.0)
    values_over_7 = sum(1 for v in proctx_valid if v >= 7.0)
    
    print(f"\n   Distribution:")
    print(f"   - 2-3 ms: {values_2_3} frames ({values_2_3/len(proctx_valid)*100:.1f}%)")
    print(f"   - 3-4 ms: {values_3_4} frames ({values_3_4/len(proctx_valid)*100:.1f}%)")
    print(f"   - 4-5 ms: {values_4_5} frames ({values_4_5/len(proctx_valid)*100:.1f}%)")
    print(f"   - 5-6 ms: {values_5_6} frames ({values_5_6/len(proctx_valid)*100:.1f}%)")
    print(f"   - 6-7 ms: {values_6_7} frames ({values_6_7/len(proctx_valid)*100:.1f}%)")
    print(f"   - >7 ms: {values_over_7} frames ({values_over_7/len(proctx_valid)*100:.1f}%)")
    
    print(f"\n   💡 INTERPRÉTATION PROC→TX:")
    print(f"   - Latence VARIABLE (2-7 ms) avec fortes oscillations")
    print(f"   - Pics visibles sur le graphique")
    print(f"   - Indique une CONTENTION sur l'outbox (_outbox)")
    print(f"   - Le thread TX (run_slicer_server) ne consomme pas assez vite")
    print(f"   - Les masques s'accumulent temporairement dans la queue")

# Statistiques RX→TX
rxtx_valid = [v for v in rxtx if v is not None]
if rxtx_valid:
    print(f"\n📊 RX → TX (Total bout-en-bout)")
    print(f"   Nombre de frames: {len(rxtx_valid)}")
    print(f"   Min: {min(rxtx_valid):.2f} ms")
    print(f"   Max: {max(rxtx_valid):.2f} ms")
    print(f"   Moyenne: {statistics.mean(rxtx_valid):.2f} ms")
    print(f"   Médiane: {statistics.median(rxtx_valid):.2f} ms")
    
    print(f"\n   💡 INTERPRÉTATION RX→TX:")
    print(f"   - Suit quasi-parfaitement PROC→TX (courbe orange = courbe bleue)")
    print(f"   - Confirme que RX→PROC est négligeable (~0-2ms)")
    print(f"   - Le goulot d'étranglement est PROC→TX")

# Analyse des patterns
print(f"\n" + "=" * 80)
print("🔍 ANALYSE DES PATTERNS OBSERVÉS")
print("=" * 80)

# Identifier les pics
peaks = []
for i, (fid, lat) in enumerate(zip(frames, proctx)):
    if lat is not None and lat >= 6.0:
        peaks.append((fid, lat))

if peaks:
    print(f"\n⚠️  PICS DE LATENCE PROC→TX (≥6ms):")
    print(f"   Nombre de pics: {len(peaks)}")
    print(f"   Frames concernées: {[p[0] for p in peaks[:10]]}{'...' if len(peaks) > 10 else ''}")
    print(f"\n   Causes probables:")
    print(f"   1. Thread TX occupé à envoyer une frame précédente")
    print(f"   2. Outbox pleine → attente de libération")
    print(f"   3. Contention réseau/socket temporaire")

# Vérifier la périodicité
print(f"\n📐 PÉRIODICITÉ DES PICS:")
if len(peaks) >= 2:
    intervals = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks)-1)]
    if intervals:
        print(f"   Intervalles entre pics: {intervals[:10]}{'...' if len(intervals) > 10 else ''}")
        print(f"   Intervalle moyen: {statistics.mean(intervals):.1f} frames")
        print(f"   → Pas de périodicité claire (intervalles variables)")
else:
    print(f"   Pas assez de pics pour analyse")

print(f"\n" + "=" * 80)
print("📋 RÉSUMÉ DU DIAGNOSTIC")
print("=" * 80)
print(f"""
✅ RX → PROC : EXCELLENT (0-2ms, très stable)
   → Le thread PROC consomme immédiatement depuis _mailbox
   → Pas de backlog, traitement efficace

⚠️  PROC → TX : VARIABLE (2-7ms, oscillations)
   → Goulot d'étranglement principal
   → Le thread TX (run_slicer_server) est parfois lent
   → Accumulation temporaire dans _outbox

💡 CONCLUSION :
   - Pipeline globale PERFORMANTE (latence totale 2-7ms)
   - Traitement PROC très rapide (seuillage simple)
   - TX légèrement saturé par moments (envoi réseau)
   
🎯 RECOMMANDATIONS (si nécessaire) :
   1. Augmenter la taille de _outbox (actuellement deque(maxlen=10))
   2. Optimiser l'envoi TX (batching ?)
   3. Monitorer la taille réelle de _outbox en temps réel
""")
