#!/usr/bin/env python3
"""
Test rapide des nouvelles métriques inter-étapes pour valider le fonctionnement.
"""

import sys
import time
from pathlib import Path

# Configuration du chemin
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from service.gateway.stats import GatewayStats


def test_interstage_metrics():
    """Test des nouvelles métriques inter-étapes."""
    print("🎯 Test des métriques inter-étapes détaillées...")
    
    stats = GatewayStats()
    
    # Simuler un workflow complet pour frame_id=1
    frame_id = 1
    
    # Timestamps simulés (en microsecondes pour plus de précision)
    t_rx = time.perf_counter()
    time.sleep(0.001)  # 1ms
    
    t1 = time.perf_counter()  # Fin CPU→GPU
    time.sleep(0.005)  # 5ms pour processing GPU
    
    t2 = time.perf_counter()  # Fin PROC(GPU)
    time.sleep(0.002)  # 2ms pour GPU→CPU
    
    t3 = time.perf_counter()  # Fin GPU→CPU
    time.sleep(0.001)  # 1ms pour TX
    
    t_tx = time.perf_counter()  # Fin TX
    
    # Enregistrer les métriques
    stats.mark_interstage_rx(frame_id, t_rx)
    stats.mark_interstage_cpu_to_gpu(frame_id, t1)
    stats.mark_interstage_proc_done(frame_id, t2)
    stats.mark_interstage_gpu_to_cpu(frame_id, t3)
    stats.mark_interstage_tx(frame_id, t_tx)
    
    # Récupérer le snapshot
    snapshot = stats.snapshot()
    
    print(f"✅ Latences inter-étapes mesurées:")
    print(f"   RX → CPU-to-GPU:    {snapshot['interstage_rx_to_cpu_gpu_ms']:.2f}ms")
    print(f"   CPU-to-GPU → PROC:  {snapshot['interstage_cpu_gpu_to_proc_ms']:.2f}ms") 
    print(f"   PROC → GPU-to-CPU:  {snapshot['interstage_proc_to_gpu_cpu_ms']:.2f}ms")
    print(f"   GPU-to-CPU → TX:    {snapshot['interstage_gpu_cpu_to_tx_ms']:.2f}ms")
    print(f"   Échantillons:       {snapshot['interstage_samples']}")
    
    # Vérifier que les valeurs sont cohérentes
    assert snapshot['interstage_samples'] == 1, "Doit avoir 1 échantillon"
    assert snapshot['interstage_rx_to_cpu_gpu_ms'] > 0, "RX→GPU doit être > 0"
    assert snapshot['interstage_cpu_gpu_to_proc_ms'] > 0, "GPU→PROC doit être > 0"
    assert snapshot['interstage_proc_to_gpu_cpu_ms'] > 0, "PROC→CPU doit être > 0"
    assert snapshot['interstage_gpu_cpu_to_tx_ms'] > 0, "CPU→TX doit être > 0"
    
    print("🎯 Test réussi ! Les métriques inter-étapes fonctionnent correctement.")


if __name__ == "__main__":
    test_interstage_metrics()