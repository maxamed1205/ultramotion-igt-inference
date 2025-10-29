"""
Script de test rapide - Pipeline + Dashboard intégré
==================================================

Test simple pour vérifier que les métriques inter-étapes 
sont bien transmises au dashboard en temps réel.

Usage:
    python test_dashboard_integration.py

Puis ouvrir: http://localhost:8050
"""

import os
import sys
import time
import threading
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def test_integration():
    """Test d'intégration dashboard + pipeline"""
    
    print("=" * 60)
    print("TEST INTÉGRATION DASHBOARD + PIPELINE")
    print("=" * 60)
    
    try:
        # Import du nouveau pipeline intégré
        from test_pipeline_avec_dashboard import PipelineWithDashboard
        
        print("✅ Import réussi")
        
        # Créer et lancer
        pipeline = PipelineWithDashboard()
        
        print("🚀 Démarrage pipeline intégrée...")
        print("📊 Dashboard disponible sur: http://localhost:8050")
        print("🔄 La pipeline va tourner en boucle avec le dataset")
        print("⚡ Vous devriez voir les métriques GPU-résident s'afficher")
        print("🛑 Ctrl+C pour arrêter")
        print("=" * 60)
        
        # Lancer (bloquant)
        pipeline.run()
        
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        print("💡 Vérifiez que tous les modules sont disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()