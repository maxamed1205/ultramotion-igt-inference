#!/usr/bin/env python
"""
Script de lancement pour Test + Dashboard Web
=============================================

Ce script lance séparément :
1. Le dashboard web sur http://localhost:8050
2. Le test du pipeline avec métriques

Usage:
------
    python launch_test_with_dashboard.py

Puis ouvrir dans le navigateur: http://localhost:8050
"""

import subprocess
import time
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def launch_dashboard():
    """Lance le dashboard web en arrière-plan"""
    print("🌐 Lancement du dashboard web...")
    
    # Configurer l'environnement
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC
    
    # Commande pour lancer le dashboard
    cmd = [
        sys.executable, "-m", "service.dashboard_unified",
        "--port", "8050",
        "--host", "0.0.0.0"
    ]
    
    # Lancer en arrière-plan
    dashboard_process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print(f"📊 Dashboard lancé (PID: {dashboard_process.pid})")
    print("🌐 URL: http://localhost:8050")
    
    return dashboard_process

def launch_test():
    """Lance le test de pipeline avec métriques"""
    print("\n🚀 Lancement du test de pipeline...")
    
    # Configurer l'environnement  
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC
    
    # Commande pour lancer le test
    cmd = [
        sys.executable, 
        "tests/tests_gateway/test_gateway_dataset_mock.py"
    ]
    
    # Lancer le test
    test_process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env
    )
    
    print(f"⚙️ Test pipeline lancé (PID: {test_process.pid})")
    
    return test_process

def main():
    print("=" * 60)
    print("🎯 LANCEMENT ULTRAMOTION IGT: DASHBOARD + TEST")
    print("=" * 60)
    
    # 1. Lancer le dashboard web
    dashboard_proc = launch_dashboard()
    
    # 2. Attendre que le dashboard démarre
    print("⏳ Attente du démarrage du dashboard (3s)...")
    time.sleep(3)
    
    # 3. Lancer le test
    test_proc = launch_test()
    
    print("\n" + "=" * 60)
    print("✅ SERVICES LANCÉS AVEC SUCCÈS")
    print("=" * 60)
    print(f"📊 Dashboard web: http://localhost:8050")
    print(f"⚙️ Test pipeline: PID {test_proc.pid}")
    print("=" * 60)
    print("\n🌐 Ouvrez votre navigateur sur: http://localhost:8050")
    print("📈 Les métriques inter-étapes apparaîtront en temps réel")
    print("\n💡 Appuyez sur Ctrl+C pour arrêter tous les services")
    
    try:
        # Attendre que le test se termine
        test_proc.wait()
        print("\n✅ Test terminé avec succès")
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        
    finally:
        # Nettoyer les processus
        print("🧹 Nettoyage des processus...")
        
        try:
            test_proc.terminate()
            test_proc.wait(timeout=5)
        except:
            test_proc.kill()
            
        try:
            dashboard_proc.terminate() 
            dashboard_proc.wait(timeout=5)
        except:
            dashboard_proc.kill()
            
        print("✅ Tous les services arrêtés")

if __name__ == "__main__":
    main()