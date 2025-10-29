"""
Générateur de métriques KPI pour tester le dashboard
====================================================

Ce script génère des métriques KPI simulées directement dans kpi.log
sans avoir besoin de la pipeline complète.

Usage:
    python tests/tests_gateway/generate_kpi_for_dashboard.py
"""

import sys
import time
import random
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Configure logging
import logging.config, yaml
from core.monitoring import async_logging

LOG_CFG = ROOT / "src" / "config" / "logging.yaml"
if LOG_CFG.exists():
    with open(LOG_CFG, "r") as f:
        cfg = yaml.safe_load(f)
        async_logging.setup_async_logging(yaml_cfg=cfg)
        async_logging.start_health_monitor()

from core.monitoring.kpi import safe_log_kpi, format_kpi

LOG = logging.getLogger("igt.kpi.generator")

def generate_realistic_metrics():
    """Génère des métriques réalistes qui varient dans le temps"""
    
    # FPS avec variation naturelle
    base_fps = 90.0
    fps_variation = random.uniform(-5, 5)
    fps = max(50.0, min(100.0, base_fps + fps_variation))
    
    # Latence corrélée inversement au FPS
    base_latency = 15.0
    latency_variation = random.uniform(-5, 5)
    latency = base_latency + latency_variation + (100 - fps) * 0.2
    latency = max(5.0, min(50.0, latency))
    
    # Drops occasionnels
    drops = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])  # Rarement des drops
    
    return {
        "fps_rx": round(fps, 2),
        "fps_tx": round(fps * 0.98, 2),  # TX légèrement plus bas
        "latency_ms": round(latency, 2),
        "drops": drops,
    }

def main():
    """Génère des métriques KPI en continu"""
    
    print("=" * 60)
    print("  GÉNÉRATEUR DE MÉTRIQUES KPI POUR DASHBOARD")
    print("=" * 60)
    print()
    print("  Ce script génère des métriques simulées dans kpi.log")
    print("  pour tester le dashboard sans pipeline complète.")
    print()
    print("  Lancez le dashboard dans un autre terminal avec:")
    print()
    print("      .\\start_dashboard.ps1")
    print()
    print("  Puis ouvrez http://localhost:8050")
    print()
    print("  Appuyez sur Ctrl+C pour arrêter.")
    print("=" * 60)
    print()
    
    LOG.info("Démarrage du générateur de métriques KPI")
    print("✅ Génération de métriques en cours...")
    print("   (consultez logs/kpi.log pour voir les données)\n")
    
    frame_count = 0
    
    try:
        while True:
            # Générer des métriques
            metrics = generate_realistic_metrics()
            
            # Logger comme un événement rx_update (format attendu par le dashboard)
            kpi_msg = format_kpi({
                "event": "rx_update",
                **metrics,
                "total_bytes_rx": frame_count * 512 * 512,
                "bytes_rx": 512 * 512
            })
            
            safe_log_kpi(kpi_msg)
            
            frame_count += 1
            
            # Afficher un résumé toutes les 10 secondes
            if frame_count % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Frame {frame_count:4d} | "
                      f"FPS: {metrics['fps_rx']:5.1f} | "
                      f"Latence: {metrics['latency_ms']:5.1f}ms | "
                      f"Drops: {metrics['drops']}")
            
            time.sleep(1)  # 1 métrique par seconde
            
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt demandé par l'utilisateur...")
        LOG.info(f"Générateur arrêté après {frame_count} frames")
        print(f"✅ {frame_count} métriques générées\n")

if __name__ == "__main__":
    main()
