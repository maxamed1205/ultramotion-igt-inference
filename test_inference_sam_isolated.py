#!/usr/bin/env python3
"""
Test isolé pour inference_sam.py sans passer par les __init__.py problématiques
"""

import sys
import os
import numpy as np

# Test d'import direct du fichier
sys.path.insert(0, 'src')

try:
    # Import direct du fichier sans passer par __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inference_sam", 
        "src/core/inference/engine/inference_sam.py"
    )
    inference_sam = importlib.util.module_from_spec(spec)
    
    # On doit d'abord simuler les dépendances
    sys.modules['core.monitoring.kpi'] = type('MockKPI', (), {
        'safe_log_kpi': lambda x: None,
        'format_kpi': lambda x: x
    })()
    
    spec.loader.exec_module(inference_sam)
    
    print("✅ inference_sam.py importé avec succès !")
    
    # Test des fonctions
    run_segmentation = inference_sam.run_segmentation
    
    # Test avec torch si disponible
    try:
        import torch
        torch_available = True
        print("✅ PyTorch détecté")
    except ImportError:
        torch_available = False
        print("⚠️ PyTorch non disponible")
    
    if torch_available:
        # Mock d'un modèle SAM simple
        class MockSAM:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1)
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    
            def set_image(self, img):
                pass
                
            def predict(self, box, multimask_output=False):
                # Retourne un masque numpy comme le vrai SAM
                mask = np.random.rand(100, 100) > 0.5
                return [mask], [0.95], None
        
        # Test du GPU path
        print("\n🔍 Test du GPU path (as_numpy=False)...")
        sam_model = MockSAM()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50])
        
        # Test mode GPU-resident
        result_gpu = run_segmentation(sam_model, test_image, bbox, as_numpy=False)
        if result_gpu is not None:
            print(f"✅ GPU path: {type(result_gpu)}")
            if hasattr(result_gpu, 'device'):
                print(f"   Device: {result_gpu.device}")
            if hasattr(result_gpu, 'shape'):
                print(f"   Shape: {result_gpu.shape}")
        else:
            print("❌ GPU path returned None")
        
        # Test mode numpy
        print("\n🔍 Test du numpy path (as_numpy=True)...")
        result_numpy = run_segmentation(sam_model, test_image, bbox, as_numpy=True)
        if result_numpy is not None:
            print(f"✅ Numpy path: {type(result_numpy)}")
            if hasattr(result_numpy, 'shape'):
                print(f"   Shape: {result_numpy.shape}")
            if hasattr(result_numpy, 'dtype'):
                print(f"   Dtype: {result_numpy.dtype}")
        else:
            print("❌ Numpy path returned None")
            
        print("\n✅ Tests des optimizations SAM terminés avec succès !")
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    import traceback
    traceback.print_exc()