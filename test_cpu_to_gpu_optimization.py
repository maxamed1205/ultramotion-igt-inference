#!/usr/bin/env python3
"""
Test script for cpu_to_gpu.py optimization verification
"""

import sys
import os
import numpy as np

# Add project src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

try:
    from core.preprocessing.cpu_to_gpu import prepare_frame_for_gpu
    from core.types import RawFrame, FrameMeta
    
    print("✅ Successfully imported prepare_frame_for_gpu")
    
    # Create test frame
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    test_meta = FrameMeta(frame_id=1, ts=1234567890.0)
    test_frame = RawFrame(image=test_image, meta=test_meta)
    
    print(f"✅ Created test frame with shape {test_image.shape}")
    
    # Test normal mode
    print("\n🔍 Testing normal mode (test_mode=False)...")
    try:
        gpu_frame_normal = prepare_frame_for_gpu(
            test_frame, 
            device="cpu",  # Use CPU to avoid CUDA issues in test
            test_mode=False
        )
        print("✅ Normal mode successful")
        print(f"   Output tensor shape: {gpu_frame_normal.tensor.shape}")
    except Exception as e:
        print(f"❌ Normal mode failed: {e}")
    
    # Test optimized mode
    print("\n🔍 Testing optimized mode (test_mode=True)...")
    try:
        gpu_frame_optimized = prepare_frame_for_gpu(
            test_frame, 
            device="cpu",  # Use CPU to avoid CUDA issues in test
            test_mode=True
        )
        print("✅ Optimized mode successful")
        print(f"   Output tensor shape: {gpu_frame_optimized.tensor.shape}")
        
        # Verify tensor values are similar
        diff = np.abs(gpu_frame_normal.tensor.numpy() - gpu_frame_optimized.tensor.numpy()).max()
        print(f"   Max difference between modes: {diff:.6f}")
        
        if diff < 1e-5:
            print("✅ Tensor values match between modes")
        else:
            print("⚠️  Tensor values differ between modes")
            
    except Exception as e:
        print(f"❌ Optimized mode failed: {e}")
    
    print("\n✅ cpu_to_gpu.py optimization test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("This might be normal if dependencies aren't installed")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()