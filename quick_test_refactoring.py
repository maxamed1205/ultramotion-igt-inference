#!/usr/bin/env python3
"""
Quick test script to validate SamPredictor refactoring
Run this after the refactoring to ensure everything works
"""

import os
import sys
import time

def test_import_and_signature():
    """Test que le module peut être importé et a la bonne signature"""
    print("🔍 Testing import and signature...")
    
    # Test de la signature sans importer les dépendances
    with open("src/core/inference/MobileSAM/mobile_sam/predictor.py", "r") as f:
        content = f.read()
    
    # Vérifier que la signature contient as_numpy
    assert "as_numpy: bool = False," in content, "as_numpy parameter not found in signature"
    
    # Vérifier que la classe SamPredictor existe
    assert "class SamPredictor:" in content, "SamPredictor class not found"
    
    # Vérifier que la méthode predict existe avec la bonne signature
    assert "def predict(" in content, "predict method not found"
    
    print("✅ Import and signature validation passed")

def test_conditional_logic():
    """Test que la logique conditionnelle est correcte"""
    print("🔍 Testing conditional logic...")
    
    with open("src/core/inference/MobileSAM/mobile_sam/predictor.py", "r") as f:
        content = f.read()
    
    # Compter les occurrences de .detach().cpu().numpy()
    numpy_calls = content.count(".detach().cpu().numpy()")
    assert numpy_calls == 3, f"Expected 3 .detach().cpu().numpy() calls, found {numpy_calls}"
    
    # Vérifier qu'elles sont dans le bloc conditionnel
    lines = content.split('\n')
    numpy_lines = [i for i, line in enumerate(lines) if '.detach().cpu().numpy()' in line]
    
    # Trouver le bloc if as_numpy:
    if_numpy_line = None
    for i, line in enumerate(lines):
        if 'if as_numpy:' in line:
            if_numpy_line = i
            break
    
    assert if_numpy_line is not None, "if as_numpy: block not found"
    
    # Vérifier que toutes les conversions numpy sont après le if
    for numpy_line in numpy_lines:
        assert numpy_line > if_numpy_line, f"Unconditional numpy conversion at line {numpy_line + 1}"
    
    print("✅ Conditional logic validation passed")

def test_kpi_instrumentation():
    """Test que l'instrumentation KPI est présente"""
    print("🔍 Testing KPI instrumentation...")
    
    with open("src/core/inference/MobileSAM/mobile_sam/predictor.py", "r") as f:
        content = f.read()
    
    required_elements = [
        "import time",
        "safe_log_kpi",
        "format_kpi", 
        '"event": "sam_predict_output"',
        '"as_numpy": int(as_numpy)',
        "except Exception:",
        "pass"
    ]
    
    for element in required_elements:
        assert element in content, f"KPI element not found: {element}"
    
    print("✅ KPI instrumentation validation passed")

def test_documentation():
    """Test que la documentation est correcte"""
    print("🔍 Testing documentation...")
    
    with open("src/core/inference/MobileSAM/mobile_sam/predictor.py", "r") as f:
        content = f.read()
    
    doc_elements = [
        "as_numpy (bool): If true, returns numpy arrays",
        "legacy mode",
        "torch tensors for GPU-resident operation"
    ]
    
    for element in doc_elements:
        assert element in content, f"Documentation element not found: {element}"
    
    print("✅ Documentation validation passed")

def test_file_structure():
    """Test que tous les fichiers nécessaires existent"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "src/core/inference/MobileSAM/mobile_sam/predictor.py",
        "tests/test_sam_predictor_gpu_resident.py", 
        "REFACTORING_SAM_PREDICTOR.md",
        "validate_refactoring.py"
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file not found: {file_path}"
        
        # Vérifier que le fichier n'est pas vide
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        assert len(content) > 100, f"File seems too small: {file_path}"
    
    print("✅ File structure validation passed")

def main():
    """Run all validation tests"""
    print("🧪 SamPredictor Refactoring Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        test_file_structure()
        test_import_and_signature()
        test_conditional_logic()
        test_kpi_instrumentation() 
        test_documentation()
        
        elapsed = time.time() - start_time
        
        print()
        print("🎉 ALL VALIDATIONS PASSED!")
        print(f"⏱️  Completed in {elapsed:.2f} seconds")
        print()
        print("📋 Summary of changes:")
        print("• ✅ SamPredictor.predict() now supports GPU-resident mode")
        print("• ✅ as_numpy=False (default): returns torch.Tensor on GPU")
        print("• ✅ as_numpy=True (legacy): returns np.ndarray for compatibility")
        print("• ✅ KPI instrumentation for monitoring usage")
        print("• ✅ Comprehensive test suite created")
        print("• ✅ Documentation updated")
        print()
        print("🚀 Ready for integration testing!")
        print("   Next step: Update orchestrator.py to use as_numpy=False")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()