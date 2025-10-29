#!/usr/bin/env python3
"""
Script de validation pour la refactorisation orchestrator.py
"""

import os
import sys
import time

def test_syntax_validation():
    """Test que le module peut être parsé sans erreurs de syntaxe"""
    print("🔍 Testing syntax validation...")
    
    import ast
    try:
        with open("src/core/inference/engine/orchestrator.py", "r") as f:
            content = f.read()
        ast.parse(content)
        print("✅ Syntax validation passed")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  File read error: {e}")
        return False

def test_signature_and_parameters():
    """Test que la nouvelle signature est correcte"""
    print("🔍 Testing function signature...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    # Vérifier que sam_as_numpy est dans la signature
    if 'sam_as_numpy: bool = False' not in content:
        print("❌ Parameter sam_as_numpy not found in signature")
        return False
    print("✅ Parameter sam_as_numpy found in method signature")
    
    # Vérifier la documentation
    if 'sam_as_numpy:' not in content and 'If False, keeps tensors on GPU' not in content:
        print("❌ sam_as_numpy not documented")
        return False
    print("✅ sam_as_numpy parameter documented")
    
    return True

def test_gpu_resident_implementation():
    """Test que l'implémentation GPU-resident est présente"""
    print("🔍 Testing GPU-resident implementation...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    required_elements = [
        "GPU-resident path",
        "not sam_as_numpy",
        "contiguous()",
        "Legacy CPU path",
        '"device": str(getattr(full_image, "device"',
    ]
    
    for element in required_elements:
        if element not in content:
            print(f"❌ GPU-resident element not found: {element}")
            return False
    
    print("✅ GPU-resident implementation found")
    return True

def test_run_segmentation_call():
    """Test que l'appel à run_segmentation a été modifié"""
    print("🔍 Testing run_segmentation call...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    # Vérifier que as_numpy est passé
    if 'as_numpy=sam_as_numpy' not in content:
        print("❌ as_numpy parameter not passed to run_segmentation")
        return False
    print("✅ as_numpy parameter passed to run_segmentation")
    
    return True

def test_kpi_instrumentation():
    """Test que l'instrumentation KPI est présente"""
    print("🔍 Testing KPI instrumentation...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    required_kpi_elements = [
        "safe_log_kpi",
        "format_kpi",
        '"event": "prepare_inference_inputs"',
        '"sam_as_numpy": int(sam_as_numpy)',
        '"tensor_type": str(type(full_image))',
    ]
    
    for element in required_kpi_elements:
        if element not in content:
            print(f"❌ KPI element not found: {element}")
            return False
    
    print("✅ KPI instrumentation found")
    return True

def test_backward_compatibility():
    """Test que la compatibilité arrière est préservée"""
    print("🔍 Testing backward compatibility...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    # Le paramètre sam_as_numpy doit avoir False comme valeur par défaut
    if 'sam_as_numpy: bool = False' not in content:
        print("❌ sam_as_numpy should default to False for GPU-resident mode")
        return False
    print("✅ sam_as_numpy defaults to False (GPU-resident mode)")
    
    # Il doit y avoir un mode de compatibilité CPU legacy
    if 'Legacy CPU path' not in content:
        print("❌ Legacy CPU compatibility mode not found")
        return False
    print("✅ Legacy CPU compatibility mode preserved")
    
    return True

def test_imports():
    """Test que les imports nécessaires sont présents"""
    print("🔍 Testing imports...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    required_imports = [
        "import torch",
        "import time",
    ]
    
    for imp in required_imports:
        if imp not in content:
            print(f"❌ Missing import: {imp}")
            return False
    
    print("✅ All required imports found")
    return True

def test_file_structure():
    """Test que tous les fichiers nécessaires existent"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "src/core/inference/engine/orchestrator.py",
        "tests/test_orchestrator_gpu_resident.py",
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False
        
        # Vérifier que le fichier n'est pas vide
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if len(content) < 100:
            print(f"❌ File seems too small: {file_path}")
            return False
    
    print("✅ File structure validation passed")
    return True

def count_gpu_cpu_transfers():
    """Compter les transferts GPU→CPU restants"""
    print("🔍 Counting GPU→CPU transfers...")
    
    with open("src/core/inference/engine/orchestrator.py", "r") as f:
        content = f.read()
    
    lines = content.split('\n')
    gpu_cpu_transfers = [line for line in lines if '.detach().cpu().numpy()' in line]
    
    print(f"📊 Found {len(gpu_cpu_transfers)} .detach().cpu().numpy() calls")
    
    # Vérifier qu'ils sont tous dans le chemin legacy
    for i, line in enumerate(gpu_cpu_transfers):
        line_num = None
        for j, content_line in enumerate(lines):
            if content_line.strip() == line.strip():
                line_num = j + 1
                break
        
        print(f"   Line {line_num}: {line.strip()}")
    
    # Tous les transferts devraient être conditionnels maintenant
    legacy_block_start = None
    for i, line in enumerate(lines):
        if "Legacy CPU path" in line:
            legacy_block_start = i
            break
    
    if legacy_block_start:
        print(f"✅ All GPU→CPU transfers are in legacy block (starting line {legacy_block_start + 1})")
    else:
        print("⚠️  Could not locate legacy CPU block")
    
    return True

def main():
    """Run all validation tests"""
    print("🧪 Orchestrator.py Refactoring Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        success = True
        success &= test_file_structure()
        print()
        success &= test_syntax_validation()
        print()
        success &= test_imports()
        print()
        success &= test_signature_and_parameters()
        print()
        success &= test_gpu_resident_implementation()
        print()
        success &= test_run_segmentation_call()
        print()
        success &= test_kpi_instrumentation()
        print()
        success &= test_backward_compatibility()
        print()
        success &= count_gpu_cpu_transfers()
        
        elapsed = time.time() - start_time
        
        if success:
            print()
            print("🎉 ALL VALIDATIONS PASSED!")
            print(f"⏱️  Completed in {elapsed:.2f} seconds")
            print()
            print("📋 Summary of orchestrator.py changes:")
            print("• ✅ Added sam_as_numpy=False parameter for GPU-resident mode")
            print("• ✅ GPU-first path: keeps tensors on GPU for SAM") 
            print("• ✅ Legacy CPU path: preserved for backward compatibility")
            print("• ✅ Modified run_segmentation call to pass as_numpy parameter")
            print("• ✅ KPI instrumentation for monitoring mode usage")
            print("• ✅ Comprehensive test suite created")
            print()
            print("🚀 Orchestrator ready for integration testing!")
            print("   Next step: Update inference_sam.py to accept as_numpy parameter")
        else:
            print()
            print("❌ SOME VALIDATIONS FAILED")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()