#!/usr/bin/env python3
"""
🔧 Test de validation finale : Dashboard + Gateway dans le même processus
=========================================================================

Ce test confirme que quand le dashboard et le gateway sont dans le même processus,
les métriques interstage sont bien visibles.
"""

import sys
import time
import threading
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def test_same_process_communication():
    """Test dashboard + gateway dans le même processus"""
    
    print("🔧 TEST FINAL : MÊME PROCESSUS")
    print("=" * 50)
    
    # Imports
    from service.gateway.manager import IGTGateway
    from core.monitoring.monitor import set_active_gateway, get_active_gateway
    from service.dashboard_unified import UnifiedMetricsCollector, DashboardConfig
    
    # 1. Créer et enregistrer gateway
    print("📡 Création du gateway...")
    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    set_active_gateway(gateway)
    
    # 2. Vérifier que le gateway est bien enregistré
    print("🔍 Vérification enregistrement...")
    active_gw = get_active_gateway()
    if active_gw is None:
        print("❌ ÉCHEC : Gateway non enregistré")
        return
    else:
        print("✅ Gateway bien enregistré")
    
    # 3. Ajouter métriques interstage
    print("⚡ Ajout de métriques interstage...")
    base_time = time.time()
    
    for frame_id in range(3):
        t_rx = base_time + frame_id * 0.1
        t_cpu_gpu = t_rx + 0.002 
        t_proc = t_cpu_gpu + 0.005
        t_gpu_cpu = t_proc + 0.003
        t_tx = t_gpu_cpu + 0.001
        
        gateway.stats.mark_interstage_rx(frame_id, t_rx)
        gateway.stats.mark_interstage_cpu_to_gpu(frame_id, t_cpu_gpu)
        gateway.stats.mark_interstage_proc_done(frame_id, t_proc)
        gateway.stats.mark_interstage_gpu_to_cpu(frame_id, t_gpu_cpu)
        gateway.stats.mark_interstage_tx(frame_id, t_tx)
    
    print("✅ 3 frames avec métriques ajoutées")
    
    # 4. Créer collecteur dashboard
    print("📊 Test collecteur dashboard...")
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # 5. Collecter métriques
    snapshot = collector.collect()
    
    # 6. Analyser résultats
    print("\n🎯 RÉSULTATS FINAUX:")
    print("-" * 40)
    
    interstage_samples = snapshot.get("interstage_samples", 0)
    rx_to_gpu = snapshot.get("interstage_rx_to_cpu_gpu_ms", 0)
    gpu_to_proc = snapshot.get("interstage_cpu_gpu_to_proc_ms", 0)
    proc_to_cpu = snapshot.get("interstage_proc_to_gpu_cpu_ms", 0)
    cpu_to_tx = snapshot.get("interstage_gpu_cpu_to_tx_ms", 0)
    
    if interstage_samples > 0:
        print(f"✅ SUCCÈS ! Échantillons: {interstage_samples}")
        print(f"✅ RX→CPU-GPU: {rx_to_gpu:.2f}ms")
        print(f"✅ CPU-GPU→PROC: {gpu_to_proc:.2f}ms")
        print(f"✅ PROC→GPU-CPU: {proc_to_cpu:.2f}ms")
        print(f"✅ GPU-CPU→TX: {cpu_to_tx:.2f}ms")
        
        print("\n🎉 VALIDATION RÉUSSIE !")
        print("💡 Le problème était bien la communication inter-processus")
        print("💡 Dans le même processus, tout fonctionne parfaitement")
        
    else:
        print("❌ ÉCHEC : Pas de métriques interstage")
        print("🔍 Debug:")
        print(f"   get_active_gateway(): {get_active_gateway()}")
        print(f"   Available keys: {list(snapshot.keys())[:10]}...")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_same_process_communication()