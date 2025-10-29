#!/usr/bin/env python3
"""
Test rapide des dépendances pour les scripts de validation.
"""

print("🔍 VÉRIFICATION DES DÉPENDANCES")
print("=" * 40)

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"🔧 CUDA: {'✅ Disponible' if cuda_available else '❌ Non disponible'}")
except ImportError:
    print("❌ PyTorch: Non installé")
    torch = None

# Test SciPy
try:
    import scipy
    from scipy.optimize import linear_sum_assignment
    print(f"✅ SciPy: {scipy.__version__}")
except ImportError:
    print("❌ SciPy: Non installé")

# Test NumPy
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy: Non installé")

# Test des imports du projet
print("\n🔍 VÉRIFICATION IMPORTS PROJET")
print("-" * 30)

try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from core.inference.d_fine.matcher import HungarianMatcher
    print("✅ HungarianMatcher: Import réussi")
except ImportError as e:
    print(f"❌ HungarianMatcher: {e}")

print("\n🎯 STATUS GLOBAL")
print("-" * 20)
if torch is not None:
    print("✅ Tests peuvent être exécutés")
else:
    print("⚠️  Tests limités sans PyTorch")