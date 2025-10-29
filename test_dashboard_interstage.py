#!/usr/bin/env python3
"""
Test rapide du dashboard avec les nouvelles métriques inter-étapes.
"""

import sys
import time
import json
from pathlib import Path

# Configuration du chemin
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from service.dashboard_unified import UnifiedMetricsCollector, DashboardConfig


def test_dashboard_interstage_metrics():
    """Test du dashboard avec métriques inter-étapes simulées."""
    print("🎯 Test du dashboard avec nouvelles métriques inter-étapes...")
    
    # Créer une config de test
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # Simuler des métriques avec nos nouvelles données inter-étapes
    fake_metrics = {
        "fps_in": 25.5,
        "fps_out": 25.2,
        "latency_ms": 12.5,
        "gpu_util": 45.3,
        
        # 🎯 Nouvelles métriques inter-étapes
        "interstage_rx_to_cpu_gpu_ms": 2.1,
        "interstage_cpu_gpu_to_proc_ms": 6.8,
        "interstage_proc_to_gpu_cpu_ms": 2.4,
        "interstage_gpu_cpu_to_tx_ms": 1.2,
        
        "interstage_rx_to_cpu_gpu_p95_ms": 2.8,
        "interstage_cpu_gpu_to_proc_p95_ms": 8.1,
        "interstage_proc_to_gpu_cpu_p95_ms": 3.2,
        "interstage_gpu_cpu_to_tx_p95_ms": 1.8,
        
        "interstage_samples": 15,
    }
    
    # Simuler get_aggregated_metrics() qui retourne nos métriques
    import core.monitoring.monitor as monitor
    original_get_aggregated = monitor.get_aggregated_metrics
    
    def mock_get_aggregated():
        return fake_metrics
    
    monitor.get_aggregated_metrics = mock_get_aggregated
    
    try:
        # Collecter un snapshot
        snapshot = collector.collect()
        
        print("✅ Snapshot collecté avec succès !")
        print(f"📊 Latences inter-étapes trouvées:")
        print(f"   RX → CPU-to-GPU: {snapshot.get('interstage_rx_to_cpu_gpu_ms', 0):.1f}ms")
        print(f"   CPU-to-GPU → PROC: {snapshot.get('interstage_cpu_gpu_to_proc_ms', 0):.1f}ms")
        print(f"   PROC → GPU-to-CPU: {snapshot.get('interstage_proc_to_gpu_cpu_ms', 0):.1f}ms")
        print(f"   GPU-to-CPU → TX: {snapshot.get('interstage_gpu_cpu_to_tx_ms', 0):.1f}ms")
        print(f"   Échantillons: {snapshot.get('interstage_samples', 0)}")
        
        # Vérifier que toutes les nouvelles métriques sont présentes
        required_keys = [
            'interstage_rx_to_cpu_gpu_ms', 'interstage_cpu_gpu_to_proc_ms',
            'interstage_proc_to_gpu_cpu_ms', 'interstage_gpu_cpu_to_tx_ms',
            'interstage_rx_to_cpu_gpu_p95_ms', 'interstage_cpu_gpu_to_proc_p95_ms',
            'interstage_proc_to_gpu_cpu_p95_ms', 'interstage_gpu_cpu_to_tx_p95_ms',
            'interstage_samples'
        ]
        
        missing_keys = [key for key in required_keys if key not in snapshot]
        if missing_keys:
            print(f"❌ Clés manquantes: {missing_keys}")
        else:
            print("✅ Toutes les métriques inter-étapes sont présentes dans le snapshot !")
        
        # Test JSON serialization (important pour WebSocket)
        json_str = json.dumps(snapshot, default=str)
        parsed = json.loads(json_str)
        print("✅ Sérialisation JSON réussie !")
        
        print("\n🎯 Dashboard prêt pour les métriques inter-étapes GPU-résident !")
        
    finally:
        # Restaurer la fonction originale
        monitor.get_aggregated_metrics = original_get_aggregated


if __name__ == "__main__":
    test_dashboard_interstage_metrics()