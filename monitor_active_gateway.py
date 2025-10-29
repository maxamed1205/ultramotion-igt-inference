#!/usr/bin/env python3
"""
🔍 Diagnostic de synchronisation : Test vs Dashboard
===================================================

Problème détecté : test_dashboard_fix.py fonctionne mais test_gateway_dataset_mock.py
ne transmet pas au dashboard.

Hypothèse : les deux tournent dans des processus séparés donc _ACTIVE_GATEWAY
n'est pas partagé entre test et dashboard.
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

def check_active_gateway():
    """Vérifie l'état du gateway actif périodiquement"""
    from core.monitoring.monitor import get_active_gateway, collect_gateway_metrics
    
    print("🔍 MONITORING GATEWAY ACTIF (Ctrl+C pour arrêter)")
    print("=" * 60)
    
    while True:
        try:
            active_gw = get_active_gateway()
            
            if active_gw is None:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ Aucun gateway actif")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ✅ Gateway actif détecté: {type(active_gw).__name__}")
                
                # Vérifier les métriques
                try:
                    gw_metrics = collect_gateway_metrics(active_gw)
                    interstage_samples = gw_metrics.get("interstage_samples", 0)
                    
                    if interstage_samples > 0:
                        print(f"    📊 Échantillons interstage: {interstage_samples}")
                        print(f"    ⏱️  RX→CPU-GPU: {gw_metrics.get('interstage_rx_to_cpu_gpu_ms', 0):.2f}ms")
                        print(f"    ⏱️  CPU-GPU→PROC: {gw_metrics.get('interstage_cpu_gpu_to_proc_ms', 0):.2f}ms") 
                        print(f"    ⏱️  PROC→GPU-CPU: {gw_metrics.get('interstage_proc_to_gpu_cpu_ms', 0):.2f}ms")
                        print(f"    ⏱️  GPU-CPU→TX: {gw_metrics.get('interstage_gpu_cpu_to_tx_ms', 0):.2f}ms")
                    else:
                        print(f"    📊 Pas d'échantillons interstage")
                
                except Exception as e:
                    print(f"    ❌ Erreur collect_gateway_metrics: {e}")
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring arrêté")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Erreur: {e}")
            time.sleep(2)

if __name__ == "__main__":
    print("💡 INSTRUCTIONS:")
    print("1. Lancer ce script dans un terminal")
    print("2. Lancer test_gateway_dataset_mock.py dans un autre terminal")
    print("3. Observer si le gateway devient actif pendant le test")
    print()
    
    check_active_gateway()