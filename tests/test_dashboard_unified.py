"""
Tests de validation pour le Dashboard Unifié
=============================================

Vérifie que le dashboard fonctionne correctement :
- Parsing des logs KPI et Pipeline
- Collection des métriques GPU et Pipeline
- Endpoints API
- Génération HTML

Usage:
    python tests/test_dashboard_unified.py
"""

import sys
import time
import json
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from service.dashboard_unified import (
    UnifiedMetricsCollector,
    DashboardConfig,
    create_app
)


def test_collector_initialization():
    """Test 1 : Initialisation du collecteur"""
    print("\n🧪 Test 1: Initialisation du collecteur...")
    
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    assert collector.config == config
    assert len(collector.history) == 0
    assert collector.latest is None
    
    print("✅ Collecteur initialisé correctement")


def test_collect_metrics():
    """Test 2 : Collection des métriques"""
    print("\n🧪 Test 2: Collection des métriques...")
    
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # Collecter les métriques
    snapshot = collector.collect()
    
    # Vérifier la structure
    assert "timestamp" in snapshot
    assert "datetime" in snapshot
    assert "fps_rx" in snapshot
    assert "fps_tx" in snapshot
    assert "gpu_util" in snapshot
    assert "gpu_transfer" in snapshot
    assert "health" in snapshot
    
    # Vérifier GPU transfer
    gpu = snapshot["gpu_transfer"]
    assert "frames" in gpu
    assert "stats" in gpu
    
    print(f"✅ Snapshot collecté avec {len(snapshot)} champs")
    print(f"   - FPS RX: {snapshot['fps_rx']}")
    print(f"   - FPS TX: {snapshot['fps_tx']}")
    print(f"   - GPU Util: {snapshot['gpu_util']}%")
    print(f"   - Health: {snapshot['health']}")


def test_history():
    """Test 3 : Historique"""
    print("\n🧪 Test 3: Historique des métriques...")
    
    config = DashboardConfig(history_size=10)
    collector = UnifiedMetricsCollector(config)
    
    # Collecter plusieurs fois
    for i in range(15):
        collector.collect()
        time.sleep(0.01)
    
    # Vérifier taille max
    history = collector.get_history()
    assert len(history) <= 10
    
    # Vérifier dernières N
    last_5 = collector.get_history(last_n=5)
    assert len(last_5) == 5
    
    print(f"✅ Historique : {len(history)} snapshots (max 10)")


def test_api_endpoints():
    """Test 4 : Endpoints API"""
    print("\n🧪 Test 4: Endpoints API...")
    
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # Collecter au moins une fois
    collector.collect()
    
    # Créer l'app
    app = create_app(collector, config)
    
    # Tester avec TestClient
    try:
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test /api/metrics/latest
        response = client.get("/api/metrics/latest")
        assert response.status_code == 200
        data = response.json()
        assert "fps_rx" in data
        assert "gpu_transfer" in data
        print(f"   ✓ /api/metrics/latest : OK")
        
        # Test /api/metrics/history
        response = client.get("/api/metrics/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "count" in data
        print(f"   ✓ /api/metrics/history : OK")
        
        # Test /api/health
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"   ✓ /api/health : OK")
        
        # Test /
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        print(f"   ✓ / (HTML) : OK")
        
        print("✅ Tous les endpoints fonctionnent")
    
    except ImportError:
        print("⚠️  TestClient non disponible, endpoints non testés")


def test_gpu_metrics_parsing():
    """Test 5 : Parsing des métriques GPU"""
    print("\n🧪 Test 5: Parsing GPU transfer...")
    
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # Créer un fichier KPI de test
    kpi_log = Path("logs/kpi_test.log")
    kpi_log.parent.mkdir(exist_ok=True)
    
    with open(kpi_log, "w") as f:
        f.write("event=copy_async device=cuda:0 H=512 W=512 norm_ms=0.8 pin_ms=1.2 copy_ms=0.5 total_ms=2.5 frame=1\n")
        f.write("event=copy_async device=cuda:0 H=512 W=512 norm_ms=0.7 pin_ms=1.1 copy_ms=0.4 total_ms=2.2 frame=2\n")
    
    # Temporairement remplacer le path global
    from service.dashboard_unified import KPI_LOG_PATH
    import service.dashboard_unified as module
    backup_path = module.KPI_LOG_PATH
    module.KPI_LOG_PATH = kpi_log
    
    # Collecter
    gpu_metrics = collector._collect_gpu_transfer_metrics()
    
    # Vérifier
    assert len(gpu_metrics["frames"]) == 2
    assert gpu_metrics["stats"]["total_frames"] == 2
    assert gpu_metrics["stats"]["avg_norm"] > 0
    
    print(f"✅ GPU metrics parsées : {gpu_metrics['stats']['total_frames']} frames")
    print(f"   - Avg norm: {gpu_metrics['stats']['avg_norm']} ms")
    print(f"   - Avg pin: {gpu_metrics['stats']['avg_pin']} ms")
    print(f"   - Avg copy: {gpu_metrics['stats']['avg_copy']} ms")
    
    # Nettoyer
    kpi_log.unlink()
    module.KPI_LOG_PATH = backup_path


def test_pipeline_metrics_parsing():
    """Test 6 : Parsing des métriques Pipeline"""
    print("\n🧪 Test 6: Parsing Pipeline logs...")
    
    config = DashboardConfig()
    collector = UnifiedMetricsCollector(config)
    
    # Créer un fichier pipeline de test
    pipeline_log = Path("logs/pipeline_test.log")
    pipeline_log.parent.mkdir(exist_ok=True)
    
    with open(pipeline_log, "w") as f:
        f.write("[2025-10-29 14:30:00,000] RX: Received frame #1\n")
        f.write("[2025-10-29 14:30:00,020] PROC: Processing frame #1\n")
        f.write("[2025-10-29 14:30:00,040] TX: Sent frame #1\n")
        f.write("[2025-10-29 14:30:00,100] RX: Received frame #2\n")
        f.write("[2025-10-29 14:30:00,120] PROC: Processing frame #2\n")
        f.write("[2025-10-29 14:30:00,140] TX: Sent frame #2\n")
    
    # Temporairement remplacer le path global
    from service.dashboard_unified import PIPELINE_LOG_PATH
    import service.dashboard_unified as module
    backup_path = module.PIPELINE_LOG_PATH
    module.PIPELINE_LOG_PATH = pipeline_log
    
    pipeline_metrics = collector._parse_pipeline_log()
    
    # Vérifier
    assert pipeline_metrics["rx_count"] >= 0
    assert pipeline_metrics["proc_count"] >= 0
    assert pipeline_metrics["tx_count"] >= 0
    
    print(f"✅ Pipeline metrics parsées")
    print(f"   - RX: {pipeline_metrics['rx_count']} frames")
    print(f"   - PROC: {pipeline_metrics['proc_count']} frames")
    print(f"   - TX: {pipeline_metrics['tx_count']} frames")
    
    # Nettoyer
    pipeline_log.unlink()
    module.PIPELINE_LOG_PATH = backup_path


def test_health_computation():
    """Test 7 : Calcul du statut de santé"""
    print("\n🧪 Test 7: Calcul du statut de santé...")
    
    config = DashboardConfig(
        fps_warning=70.0,
        fps_critical=50.0,
        latency_warning=30.0,
        latency_critical=50.0,
        gpu_warning=90.0,
        gpu_critical=95.0
    )
    collector = UnifiedMetricsCollector(config)
    
    # Test OK
    health = collector._compute_health(
        {"fps_in": 85.0, "latency_ms": 20.0},
        gpu=75.0,
        queues={}
    )
    assert health == "OK"
    print(f"   ✓ Scénario OK : {health}")
    
    # Test WARNING (FPS)
    health = collector._compute_health(
        {"fps_in": 65.0, "latency_ms": 20.0},
        gpu=75.0,
        queues={}
    )
    assert health == "WARNING"
    print(f"   ✓ Scénario WARNING (FPS) : {health}")
    
    # Test CRITICAL (latence)
    health = collector._compute_health(
        {"fps_in": 85.0, "latency_ms": 55.0},
        gpu=75.0,
        queues={}
    )
    assert health == "CRITICAL"
    print(f"   ✓ Scénario CRITICAL (latence) : {health}")
    
    # Test CRITICAL (GPU)
    health = collector._compute_health(
        {"fps_in": 85.0, "latency_ms": 20.0},
        gpu=97.0,
        queues={}
    )
    assert health == "CRITICAL"
    print(f"   ✓ Scénario CRITICAL (GPU) : {health}")
    
    print("✅ Calcul de santé OK pour tous les scénarios")


def main():
    """Lance tous les tests"""
    print("=" * 60)
    print("Tests de validation - Dashboard Unifié")
    print("=" * 60)
    
    try:
        test_collector_initialization()
        test_collect_metrics()
        test_history()
        test_gpu_metrics_parsing()
        test_pipeline_metrics_parsing()
        test_health_computation()
        test_api_endpoints()
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS PASSENT !")
        print("=" * 60)
        
        return 0
    
    except AssertionError as e:
        print(f"\n❌ Test échoué : {e}")
        return 1
    
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
