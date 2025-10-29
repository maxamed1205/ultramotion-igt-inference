#!/usr/bin/env python3
"""
🔍 Diagnostic de la connexion entre système interstage et dashboard
==================================================================

Ce script teste la chaîne complète :
1. Créer un gateway avec métriques interstage
2. Marquer quelques frames avec des métriques
3. Vérifier que get_aggregated_metrics() retourne les bonnes valeurs
4. Comparer avec ce que le dashboard devrait voir
"""

import sys
import time
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Imports
from service.gateway.manager import IGTGateway
from core.monitoring.monitor import set_active_gateway, get_aggregated_metrics, collect_gateway_metrics

def test_interstage_to_dashboard():
    """Test la connexion complète interstage → monitoring → dashboard"""
    
    print("🔍 DIAGNOSTIC INTERSTAGE → DASHBOARD")
    print("=" * 60)
    
    # 1. Créer un gateway
    print("📡 Création du gateway...")
    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    
    # 2. Enregistrer le gateway dans le monitoring
    print("🔗 Enregistrement dans le monitoring...")
    set_active_gateway(gateway)
    
    # 3. Simuler quelques frames avec métriques interstage
    print("⚡ Simulation de 3 frames avec métriques interstage...")
    
    base_time = time.time()
    
    for frame_id in range(3):
        t_rx = base_time + frame_id * 0.1
        t_cpu_gpu = t_rx + 0.002  # +2ms
        t_proc = t_cpu_gpu + 0.005  # +5ms  
        t_gpu_cpu = t_proc + 0.003  # +3ms
        t_tx = t_gpu_cpu + 0.001  # +1ms
        
        # Marquer tous les points interstage
        gateway.stats.mark_interstage_rx(frame_id, t_rx)
        gateway.stats.mark_interstage_cpu_to_gpu(frame_id, t_cpu_gpu)
        gateway.stats.mark_interstage_proc_done(frame_id, t_proc)
        gateway.stats.mark_interstage_gpu_to_cpu(frame_id, t_gpu_cpu)
        gateway.stats.mark_interstage_tx(frame_id, t_tx)
        
        print(f"  ✅ Frame #{frame_id:03d}: RX→CPU-GPU→PROC→GPU-CPU→TX marqués")
    
    # 4. Attendre un peu pour que les calculs se fassent
    time.sleep(0.1)
    
    # 5. Vérifier le snapshot direct du gateway
    print("\n📊 VÉRIFICATION SNAPSHOT GATEWAY:")
    print("-" * 40)
    
    gateway_snap = gateway.stats.snapshot()
    
    if gateway_snap.get("interstage_samples", 0) > 0:
        print(f"✅ Échantillons interstage: {gateway_snap.get('interstage_samples', 0)}")
        print(f"✅ RX→CPU-GPU: {gateway_snap.get('interstage_rx_to_cpu_gpu_ms', 0):.2f}ms")
        print(f"✅ CPU-GPU→PROC: {gateway_snap.get('interstage_cpu_gpu_to_proc_ms', 0):.2f}ms") 
        print(f"✅ PROC→GPU-CPU: {gateway_snap.get('interstage_proc_to_gpu_cpu_ms', 0):.2f}ms")
        print(f"✅ GPU-CPU→TX: {gateway_snap.get('interstage_gpu_cpu_to_tx_ms', 0):.2f}ms")
    else:
        print("❌ Aucun échantillon interstage dans le snapshot!")
        print("📋 Contenu snapshot:", list(gateway_snap.keys()))
    
    # 6. Vérifier collect_gateway_metrics()
    print("\n🔗 VÉRIFICATION COLLECT_GATEWAY_METRICS:")
    print("-" * 50)
    
    gw_metrics = collect_gateway_metrics(gateway)
    
    if gw_metrics.get("interstage_samples", 0) > 0:
        print(f"✅ Échantillons: {gw_metrics.get('interstage_samples', 0)}")
        print(f"✅ RX→CPU-GPU: {gw_metrics.get('interstage_rx_to_cpu_gpu_ms', 0):.2f}ms")
        print(f"✅ CPU-GPU→PROC: {gw_metrics.get('interstage_cpu_gpu_to_proc_ms', 0):.2f}ms")
        print(f"✅ PROC→GPU-CPU: {gw_metrics.get('interstage_proc_to_gpu_cpu_ms', 0):.2f}ms")
        print(f"✅ GPU-CPU→TX: {gw_metrics.get('interstage_gpu_cpu_to_tx_ms', 0):.2f}ms")
    else:
        print("❌ Aucun échantillon interstage dans collect_gateway_metrics!")
        print("📋 Clés disponibles:", list(gw_metrics.keys()))
    
    # 7. Vérifier get_aggregated_metrics()
    print("\n📈 VÉRIFICATION GET_AGGREGATED_METRICS:")
    print("-" * 50)
    
    agg_metrics = get_aggregated_metrics()
    
    if agg_metrics and agg_metrics.get("interstage_samples", 0) > 0:
        print(f"✅ Échantillons: {agg_metrics.get('interstage_samples', 0)}")
        print(f"✅ RX→CPU-GPU: {agg_metrics.get('interstage_rx_to_cpu_gpu_ms', 0):.2f}ms")
        print(f"✅ CPU-GPU→PROC: {agg_metrics.get('interstage_cpu_gpu_to_proc_ms', 0):.2f}ms")
        print(f"✅ PROC→GPU-CPU: {agg_metrics.get('interstage_proc_to_gpu_cpu_ms', 0):.2f}ms")
        print(f"✅ GPU-CPU→TX: {agg_metrics.get('interstage_gpu_cpu_to_tx_ms', 0):.2f}ms")
        print("\n🎯 SUCCESS: Le dashboard devrait voir ces métriques!")
    else:
        print("❌ get_aggregated_metrics() ne retourne pas les métriques interstage!")
        if agg_metrics:
            print("📋 Clés disponibles:", list(agg_metrics.keys()))
        else:
            print("📋 get_aggregated_metrics() retourne None!")
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTIC TERMINÉ")

if __name__ == "__main__":
    test_interstage_to_dashboard()