#!/usr/bin/env python3
"""
Test complet du workflow inter-étapes avec GatewayStats.
Simule le traitement d'une frame complète pour valider les métriques.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Configuration du chemin
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from service.gateway.stats import GatewayStats


def test_full_interstage_workflow():
    """Test du workflow complet avec métriques inter-étapes."""
    print("🎯 Test du workflow complet RX → CPU-to-GPU → PROC(GPU) → GPU-to-CPU → TX")
    
    stats = GatewayStats()
    
    # Simuler le traitement de 3 frames pour obtenir des moyennes
    for frame_id in range(1, 4):
        print(f"\n📋 Frame #{frame_id}:")
        
        # Timestamps simulés avec des latences réalistes
        t_rx = time.perf_counter()
        print(f"  ⏱️  RX: {t_rx:.6f}")
        
        # 1. Enregistrer RX
        stats.mark_interstage_rx(frame_id, t_rx)
        
        # 2. CPU→GPU transfer (simulé 1-2ms)
        time.sleep(0.001 + np.random.uniform(0, 0.001))
        t1 = time.perf_counter()
        stats.mark_interstage_cpu_to_gpu(frame_id, t1)
        print(f"  ⏱️  CPU→GPU: {t1:.6f} (+{(t1-t_rx)*1000:.2f}ms)")
        
        # 3. GPU processing (simulé 3-7ms)
        time.sleep(0.003 + np.random.uniform(0, 0.004))
        t2 = time.perf_counter()
        stats.mark_interstage_proc_done(frame_id, t2)
        print(f"  ⏱️  PROC: {t2:.6f} (+{(t2-t1)*1000:.2f}ms)")
        
        # 4. GPU→CPU transfer (simulé 1-3ms)
        time.sleep(0.001 + np.random.uniform(0, 0.002))
        t3 = time.perf_counter()
        stats.mark_interstage_gpu_to_cpu(frame_id, t3)
        print(f"  ⏱️  GPU→CPU: {t3:.6f} (+{(t3-t2)*1000:.2f}ms)")
        
        # 5. TX (simulé 0.5-1ms)
        time.sleep(0.0005 + np.random.uniform(0, 0.0005))
        t_tx = time.perf_counter()
        stats.mark_interstage_tx(frame_id, t_tx)
        print(f"  ⏱️  TX: {t_tx:.6f} (+{(t_tx-t3)*1000:.2f}ms)")
        
        # Total pour cette frame
        total_ms = (t_tx - t_rx) * 1000
        print(f"  📊 Total: {total_ms:.2f}ms")
    
    # Récupérer le snapshot final avec toutes les métriques
    snapshot = stats.snapshot()
    
    print("\n" + "="*70)
    print("🎯 MÉTRIQUES INTER-ÉTAPES FINALES")
    print("="*70)
    
    print(f"📊 Latences moyennes:")
    print(f"   RX → CPU-to-GPU:    {snapshot['interstage_rx_to_cpu_gpu_ms']:.2f}ms")
    print(f"   CPU-to-GPU → PROC:  {snapshot['interstage_cpu_gpu_to_proc_ms']:.2f}ms") 
    print(f"   PROC → GPU-to-CPU:  {snapshot['interstage_proc_to_gpu_cpu_ms']:.2f}ms")
    print(f"   GPU-to-CPU → TX:    {snapshot['interstage_gpu_cpu_to_tx_ms']:.2f}ms")
    
    print(f"\n📊 Percentiles P95:")
    print(f"   RX → CPU-to-GPU:    {snapshot['interstage_rx_to_cpu_gpu_p95_ms']:.2f}ms")
    print(f"   CPU-to-GPU → PROC:  {snapshot['interstage_cpu_gpu_to_proc_p95_ms']:.2f}ms") 
    print(f"   PROC → GPU-to-CPU:  {snapshot['interstage_proc_to_gpu_cpu_p95_ms']:.2f}ms")
    print(f"   GPU-to-CPU → TX:    {snapshot['interstage_gpu_cpu_to_tx_p95_ms']:.2f}ms")
    
    print(f"\n📊 Échantillons: {snapshot['interstage_samples']}")
    
    # Calcul total pour vérification
    total_avg = (snapshot['interstage_rx_to_cpu_gpu_ms'] + 
                 snapshot['interstage_cpu_gpu_to_proc_ms'] + 
                 snapshot['interstage_proc_to_gpu_cpu_ms'] + 
                 snapshot['interstage_gpu_cpu_to_tx_ms'])
    
    print(f"\n🎯 Total moyen des étapes: {total_avg:.2f}ms")
    print(f"🎯 Latence RX→TX globale: {snapshot['latency_ms_avg']:.2f}ms")
    
    # Vérifications
    assert snapshot['interstage_samples'] == 3, f"Doit avoir 3 échantillons, trouvé {snapshot['interstage_samples']}"
    assert all(snapshot[k] > 0 for k in snapshot if k.startswith('interstage_') and k.endswith('_ms')), "Toutes les latences doivent être > 0"
    
    print("\n✅ Test complet réussi ! Les métriques inter-étapes fonctionnent parfaitement.")
    
    return snapshot


if __name__ == "__main__":
    import numpy as np  # Pour les variations aléatoires
    test_full_interstage_workflow()