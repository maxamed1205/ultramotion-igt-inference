#!/usr/bin/env python3
"""
🎯 Test final : dashboard avec métriques interstage
==================================================

Ce script teste le correctif du dashboard pour qu'il récupère
les métriques interstage du gateway actif
"""

import sys
import time
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def test_dashboard_with_interstage():
    """Test le dashboard corrigé avec métriques interstage"""
    
    print("🎯 TEST DASHBOARD + MÉTRIQUES INTERSTAGE")
    print("=" * 50)
    
    # Imports
    from service.gateway.manager import IGTGateway
    from core.monitoring.monitor import set_active_gateway
    from service.dashboard_unified import UnifiedMetricsCollector, DashboardConfig
    
    # 1. Créer et enregistrer le gateway
    print("📡 Création et enregistrement du gateway...")
    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    set_active_gateway(gateway)
    
    # 2. Ajouter quelques métriques interstage
    print("⚡ Ajout de métriques interstage...")
    base_time = time.time()
    
    for frame_id in range(5):
        t_rx = base_time + frame_id * 0.1
        t_cpu_gpu = t_rx + 0.002  # +2ms
        t_proc = t_cpu_gpu + 0.005  # +5ms  
        t_gpu_cpu = t_proc + 0.003  # +3ms
        t_tx = t_gpu_cpu + 0.001  # +1ms
        
        gateway.stats.mark_interstage_rx(frame_id, t_rx)
        gateway.stats.mark_interstage_cpu_to_gpu(frame_id, t_cpu_gpu)
        gateway.stats.mark_interstage_proc_done(frame_id, t_proc)
        gateway.stats.mark_interstage_gpu_to_cpu(frame_id, t_gpu_cpu)
        gateway.stats.mark_interstage_tx(frame_id, t_tx)
    
    print(f"✅ {5} frames avec métriques ajoutées")
    
    # 3. Créer le collecteur de dashboard
    print("📊 Test du collecteur de dashboard...")
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # 4. Collecter les métriques
    snapshot = collector.collect()
    
    # 5. Vérifier les résultats
    print("\n🔍 RÉSULTATS:")
    print("-" * 30)
    
    interstage_samples = snapshot.get("interstage_samples", 0)
    
    if interstage_samples > 0:
        print(f"✅ Échantillons interstage: {interstage_samples}")
        print(f"✅ RX→CPU-GPU: {snapshot.get('interstage_rx_to_cpu_gpu_ms', 0):.2f}ms")
        print(f"✅ CPU-GPU→PROC: {snapshot.get('interstage_cpu_gpu_to_proc_ms', 0):.2f}ms")
        print(f"✅ PROC→GPU-CPU: {snapshot.get('interstage_proc_to_gpu_cpu_ms', 0):.2f}ms")
        print(f"✅ GPU-CPU→TX: {snapshot.get('interstage_gpu_cpu_to_tx_ms', 0):.2f}ms")
        print(f"✅ Percentiles P95:")
        print(f"   RX→CPU-GPU: {snapshot.get('interstage_rx_to_cpu_gpu_p95_ms', 0):.2f}ms")
        print(f"   CPU-GPU→PROC: {snapshot.get('interstage_cpu_gpu_to_proc_p95_ms', 0):.2f}ms")
        print(f"   PROC→GPU-CPU: {snapshot.get('interstage_proc_to_gpu_cpu_p95_ms', 0):.2f}ms")
        print(f"   GPU-CPU→TX: {snapshot.get('interstage_gpu_cpu_to_tx_p95_ms', 0):.2f}ms")
        
        print("\n🎯 SUCCESS! Le dashboard voit maintenant les métriques interstage!")
        print("🚀 Les valeurs ne seront plus 0.0 / 0.0 dans l'interface!")
        
    else:
        print("❌ Le dashboard ne voit toujours pas les métriques interstage")
        print(f"📋 Clés disponibles: {list(snapshot.keys())}")
        
        # Debug
        print("\n🔍 DEBUG:")
        print(f"   fps_in: {snapshot.get('fps_in', 'N/A')}")
        print(f"   fps_out: {snapshot.get('fps_out', 'N/A')}")
        print(f"   latency_ms: {snapshot.get('latency_ms', 'N/A')}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_dashboard_with_interstage()