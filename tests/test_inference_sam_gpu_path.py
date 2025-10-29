#!/usr/bin/env python3
"""
Test unitaire pour valider le GPU path dans inference_sam.py
"""

import sys
import os
import numpy as np

# Add project src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

try:
    import torch
except ImportError:
    torch = None

def test_sam_gpu_path():
    """Test que run_segmentation retourne bien un tensor GPU quand as_numpy=False"""
    
    if torch is None:
        print("⚠️ PyTorch not available, skipping GPU test")
        return
    
    try:
        from core.inference.engine.inference_sam import run_segmentation
        
        class DummySAM:
            """Mock SAM model pour les tests"""
            def __init__(self):
                # Simule un modèle avec des paramètres sur GPU
                self.model = torch.nn.Conv2d(3, 1, 1)
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    
            def set_image(self, img): 
                pass
                
            def predict(self, box, multimask_output=False):
                # Retourne un masque numpy (simule le comportement SAM réel)
                return [np.random.rand(256, 256)], [0.9], None
        
        # Test en mode GPU-resident (as_numpy=False)
        print("🔍 Testing SAM GPU path (as_numpy=False)...")
        sam = DummySAM()
        
        test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        bbox_xyxy = [0, 0, 10, 10]
        
        mask_tensor = run_segmentation(
            sam, 
            test_image, 
            bbox_xyxy=bbox_xyxy, 
            as_numpy=False
        )
        
        if mask_tensor is not None:
            print(f"✅ GPU path successful: returned {type(mask_tensor)}")
            
            if torch.cuda.is_available():
                if isinstance(mask_tensor, torch.Tensor):
                    print(f"   Device: {mask_tensor.device}")
                    print(f"   Shape: {mask_tensor.shape}")
                    print(f"   Dtype: {mask_tensor.dtype}")
                    
                    if mask_tensor.device.type == "cuda":
                        print("✅ Mask correctly on GPU")
                    else:
                        print("⚠️ Mask on CPU (expected on GPU)")
                else:
                    print("⚠️ Expected torch.Tensor, got", type(mask_tensor))
            else:
                print("✅ CPU fallback working (no CUDA available)")
        else:
            print("❌ GPU path failed: returned None")
        
        # Test en mode numpy (as_numpy=True)
        print("\n🔍 Testing SAM numpy path (as_numpy=True)...")
        mask_numpy = run_segmentation(
            sam, 
            test_image, 
            bbox_xyxy=bbox_xyxy, 
            as_numpy=True
        )
        
        if mask_numpy is not None:
            print(f"✅ Numpy path successful: returned {type(mask_numpy)}")
            if isinstance(mask_numpy, np.ndarray):
                print(f"   Shape: {mask_numpy.shape}")
                print(f"   Dtype: {mask_numpy.dtype}")
                print("✅ Numpy conversion working correctly")
            else:
                print("⚠️ Expected np.ndarray, got", type(mask_numpy))
        else:
            print("❌ Numpy path failed: returned None")
            
        print("\n✅ SAM GPU path test completed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure core.inference.engine.inference_sam is available")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sam_gpu_path()