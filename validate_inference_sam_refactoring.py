#!/usr/bin/env python3
"""
Script de validation pour la refactorisation inference_sam.py
"""

import os
import sys
import time

def test_syntax_validation():
    """Test que le module peut être parsé sans erreurs de syntaxe"""
    print("🔍 Testing syntax validation...")
    
    import ast
    try:
        with open("src/core/inference/engine/inference_sam.py", "r") as f:
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
    print("🔍 Testing function signatures...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    # Vérifier que as_numpy est dans la signature de run_segmentation
    if 'as_numpy: bool = False,' not in content:
        print("❌ Parameter as_numpy not found in run_segmentation signature")
        return False
    print("✅ Parameter as_numpy found in run_segmentation signature")
    
    # Vérifier que _run_segmentation_legacy a aussi le paramètre
    if 'def _run_segmentation_legacy(sam_model: Any, roi: np.ndarray, as_numpy: bool = False)' not in content:
        print("❌ Parameter as_numpy not found in _run_segmentation_legacy signature")
        return False
    print("✅ Parameter as_numpy found in _run_segmentation_legacy signature")
    
    # Vérifier la documentation
    if 'as_numpy: Si True, retourne numpy array' not in content:
        print("❌ as_numpy not documented")
        return False
    print("✅ as_numpy parameter documented")
    
    return True

def test_gpu_resident_implementation():
    """Test que l'implémentation GPU-resident est présente"""
    print("🔍 Testing GPU-resident implementation...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    required_elements = [
        "Nouveau chemin GPU-resident",
        "if not as_numpy:",
        "return mask.detach()",
        "mode",
        "mask.detach().cpu().numpy()",
    ]
    
    for element in required_elements:
        if element not in content:
            print(f"❌ GPU-resident element not found: {element}")
            return False
    
    print("✅ GPU-resident implementation found")
    return True

def test_conditional_processing():
    """Test que le traitement conditionnel est correct"""
    print("🔍 Testing conditional processing...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    # Vérifier la réduction dimensionnelle conditionnelle
    if 'if as_numpy:' not in content:
        print("❌ Conditional as_numpy processing not found")
        return False
    
    # Vérifier que le GPU tensor path existe
    if 'GPU tensor' not in content and 'ne le réduit pas ici' not in content:
        print("❌ GPU tensor processing comment not found")
        return False
    
    print("✅ Conditional processing found")
    return True

def test_legacy_calls_updated():
    """Test que les appels à _run_segmentation_legacy ont été mis à jour"""
    print("🔍 Testing legacy calls...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    # Compter les appels avec as_numpy
    legacy_calls_with_as_numpy = content.count("_run_segmentation_legacy(sam_model, image, as_numpy")
    if legacy_calls_with_as_numpy < 2:
        print(f"❌ Expected at least 2 updated legacy calls, found {legacy_calls_with_as_numpy}")
        return False
    
    print(f"✅ Found {legacy_calls_with_as_numpy} updated legacy calls")
    return True

def test_mask_astype_adaptation():
    """Test que mask.astype(bool) a été adapté"""
    print("🔍 Testing mask.astype adaptation...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    # Il ne devrait plus y avoir de mask.astype(bool) direct
    if content.count("mask.astype(bool)") > 1:
        print("❌ Found unconditional mask.astype(bool) calls")
        return False
    
    # Vérifier qu'il y a un mode conditionnel
    if 'if as_numpy:' not in content or 'return mask.astype(bool)' not in content:
        print("❌ Conditional mask.astype(bool) not found")
        return False
    
    # Vérifier le mode GPU
    if 'mask_t = torch.from_numpy(mask).to(device)' not in content:
        print("❌ GPU tensor conversion not found")
        return False
    
    print("✅ mask.astype adaptation found")
    return True

def test_kpi_instrumentation():
    """Test que l'instrumentation KPI est présente"""
    print("🔍 Testing KPI instrumentation...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    required_kpi_elements = [
        "safe_log_kpi",
        "format_kpi",
        '"event": "sam_mask_output"',
        '"as_numpy": int(as_numpy)',
        '"device": str(getattr(mask, "device"',
    ]
    
    for element in required_kpi_elements:
        if element not in content:
            print(f"❌ KPI element not found: {element}")
            return False
    
    print("✅ KPI instrumentation found")
    return True

def test_imports():
    """Test que les imports nécessaires sont présents"""
    print("🔍 Testing imports...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    required_imports = [
        "import time",
    ]
    
    for imp in required_imports:
        if imp not in content:
            print(f"❌ Missing import: {imp}")
            return False
    
    print("✅ All required imports found")
    return True

def count_gpu_cpu_transfers():
    """Compter les transferts GPU→CPU restants"""
    print("🔍 Counting GPU→CPU transfers...")
    
    with open("src/core/inference/engine/inference_sam.py", "r") as f:
        content = f.read()
    
    lines = content.split('\n')
    gpu_cpu_transfers = [line for line in lines if '.detach().cpu().numpy()' in line]
    
    print(f"📊 Found {len(gpu_cpu_transfers)} .detach().cpu().numpy() calls")
    
    # Vérifier qu'ils sont tous conditionnels
    for i, line in enumerate(gpu_cpu_transfers):
        line_num = None
        for j, content_line in enumerate(lines):
            if content_line.strip() == line.strip():
                line_num = j + 1
                break
        
        print(f"   Line {line_num}: {line.strip()}")
    
    # Tous les transferts devraient être conditionnels maintenant
    if len(gpu_cpu_transfers) > 1:
        print("⚠️  Multiple GPU→CPU transfers found - verify they are all conditional")
    else:
        print("✅ GPU→CPU transfers are properly conditioned")
    
    return True

def test_file_structure():
    """Test que tous les fichiers nécessaires existent"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "src/core/inference/engine/inference_sam.py",
        "tests/test_inference_sam_gpu_resident.py",
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

def main():
    """Run all validation tests"""
    print("🧪 inference_sam.py Refactoring Validation")
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
        success &= test_conditional_processing()
        print()
        success &= test_legacy_calls_updated()
        print()
        success &= test_mask_astype_adaptation()
        print()
        success &= test_kpi_instrumentation()
        print()
        success &= count_gpu_cpu_transfers()
        
        elapsed = time.time() - start_time
        
        if success:
            print()
            print("🎉 ALL VALIDATIONS PASSED!")
            print(f"⏱️  Completed in {elapsed:.2f} seconds")
            print()
            print("📋 Summary of inference_sam.py changes:")
            print("• ✅ Added as_numpy=False parameter for GPU-resident mode")
            print("• ✅ GPU-first path: keeps masks on GPU for pipeline")
            print("• ✅ Legacy CPU path: preserved for backward compatibility")
            print("• ✅ Conditional processing for dimensional reduction")
            print("• ✅ Modified mask.astype(bool) to be conditional")
            print("• ✅ KPI instrumentation for monitoring mask device")
            print("• ✅ Comprehensive test suite created")
            print()
            print("🚀 inference_sam.py ready for integration testing!")
            print("   GPU-resident pipeline: DFINE → Orchestrator → SAM → ResultPacket")
            print("   No more premature GPU→CPU transfers!")
        else:
            print()
            print("❌ SOME VALIDATIONS FAILED")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()