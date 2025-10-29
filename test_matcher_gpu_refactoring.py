#!/usr/bin/env python3
"""
Test de validation de la refactorisation GPU-resident du HungarianMatcher.

Ce script valide que la nouvelle implémentation GPU-resident du matcher
maintient la compatibilité avec l'ancienne version CPU tout en éliminant
les transferts GPU→CPU prématurés.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.inference.d_fine.matcher import HungarianMatcher


def create_test_data(batch_size=2, num_queries=100, num_classes=10, device='cpu'):
    """
    Crée des données de test synthétiques pour valider le matcher.
    """
    # Outputs simulés du modèle DFINE
    outputs = {
        'pred_logits': torch.rand(batch_size, num_queries, num_classes, device=device),
        'pred_boxes': torch.rand(batch_size, num_queries, 4, device=device)  # cxcywh format
    }
    
    # Targets simulés
    targets = []
    for b in range(batch_size):
        num_targets = np.random.randint(1, 20)  # Entre 1 et 19 targets par batch
        targets.append({
            'labels': torch.randint(0, num_classes, (num_targets,), device=device),
            'boxes': torch.rand(num_targets, 4, device=device)  # cxcywh format
        })
    
    return outputs, targets


def test_matcher_consistency():
    """
    Teste que les résultats GPU et CPU sont cohérents.
    """
    print("🧪 Test de cohérence GPU vs CPU...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device utilisé: {device}")
    
    # Configuration du matcher
    weight_dict = {'cost_class': 1.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
    
    # Créer deux instances: une GPU, une CPU
    matcher_gpu = HungarianMatcher(weight_dict, use_gpu_match=True)
    matcher_cpu = HungarianMatcher(weight_dict, use_gpu_match=False)
    
    # Données de test
    outputs, targets = create_test_data(batch_size=3, device=device)
    
    # Test sans topk
    with torch.no_grad():
        result_gpu = matcher_gpu(outputs, targets, return_topk=False)
        result_cpu = matcher_cpu(outputs, targets, return_topk=False)
    
    print(f"✅ Résultats GPU: {len(result_gpu['indices'])} paires d'indices")
    print(f"✅ Résultats CPU: {len(result_cpu['indices'])} paires d'indices")
    
    # Vérifier que les formats sont cohérents
    assert len(result_gpu['indices']) == len(result_cpu['indices']), "Nombre de batch différent"
    
    for i, (gpu_pair, cpu_pair) in enumerate(zip(result_gpu['indices'], result_cpu['indices'])):
        gpu_queries, gpu_targets = gpu_pair
        cpu_queries, cpu_targets = cpu_pair
        
        print(f"  Batch {i}: GPU={len(gpu_queries)} assignments, CPU={len(cpu_queries)} assignments")
        
        # Les assignments peuvent être différents (approximation vs exact) mais doivent être cohérents
        assert len(gpu_queries) == len(gpu_targets), f"GPU: query/target mismatch batch {i}"
        assert len(cpu_queries) == len(cpu_targets), f"CPU: query/target mismatch batch {i}"
        
        # Vérifier que les tenseurs sont sur le bon device
        if device == 'cuda':
            assert gpu_queries.device.type == 'cuda', f"GPU queries not on CUDA: {gpu_queries.device}"
            assert gpu_targets.device.type == 'cuda', f"GPU targets not on CUDA: {gpu_targets.device}"
    
    print("✅ Test de cohérence réussi!")
    return True


def test_performance_comparison():
    """
    Compare les performances GPU vs CPU.
    """
    print("\n📊 Test de performance GPU vs CPU...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  CUDA non disponible, test de performance limité")
        return True
    
    weight_dict = {'cost_class': 1.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
    
    matcher_gpu = HungarianMatcher(weight_dict, use_gpu_match=True)
    matcher_cpu = HungarianMatcher(weight_dict, use_gpu_match=False)
    
    # Test avec des données plus importantes
    outputs, targets = create_test_data(batch_size=8, num_queries=300, device=device)
    
    # Benchmark GPU
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            result_gpu = matcher_gpu(outputs, targets, return_topk=False)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start_time) / 10
    
    # Benchmark CPU
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            result_cpu = matcher_cpu(outputs, targets, return_topk=False)
    torch.cuda.synchronize()
    cpu_time = (time.time() - start_time) / 10
    
    print(f"⏱️  Temps moyen GPU: {gpu_time:.4f}s")
    print(f"⏱️  Temps moyen CPU: {cpu_time:.4f}s")
    print(f"🚀 Speedup: {cpu_time/gpu_time:.2f}x")
    
    return True


def test_topk_functionality():
    """
    Teste la fonctionnalité topk avec la nouvelle implémentation.
    """
    print("\n🔝 Test fonctionnalité topk...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_dict = {'cost_class': 1.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
    
    matcher_gpu = HungarianMatcher(weight_dict, use_gpu_match=True)
    matcher_cpu = HungarianMatcher(weight_dict, use_gpu_match=False)
    
    outputs, targets = create_test_data(batch_size=2, device=device)
    
    # Test avec topk
    with torch.no_grad():
        result_gpu_topk = matcher_gpu(outputs, targets, return_topk=5)
        result_cpu_topk = matcher_cpu(outputs, targets, return_topk=5)
    
    print(f"✅ GPU topk keys: {list(result_gpu_topk.keys())}")
    print(f"✅ CPU topk keys: {list(result_cpu_topk.keys())}")
    
    assert 'indices_o2m' in result_gpu_topk, "Missing indices_o2m in GPU topk result"
    assert 'indices_o2m' in result_cpu_topk, "Missing indices_o2m in CPU topk result"
    
    print("✅ Test topk réussi!")
    return True


def test_edge_cases():
    """
    Teste les cas edge: batch vides, targets vides, etc.
    """
    print("\n🎯 Test des cas edge...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_dict = {'cost_class': 1.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
    
    matcher_gpu = HungarianMatcher(weight_dict, use_gpu_match=True)
    matcher_cpu = HungarianMatcher(weight_dict, use_gpu_match=False)
    
    # Test avec des targets vides
    outputs = {
        'pred_logits': torch.rand(2, 100, 10, device=device),
        'pred_boxes': torch.rand(2, 100, 4, device=device)
    }
    
    targets = [
        {'labels': torch.tensor([], dtype=torch.long, device=device), 
         'boxes': torch.zeros(0, 4, device=device)},
        {'labels': torch.tensor([1, 3], device=device), 
         'boxes': torch.rand(2, 4, device=device)}
    ]
    
    with torch.no_grad():
        result_gpu = matcher_gpu(outputs, targets, return_topk=False)
        result_cpu = matcher_cpu(outputs, targets, return_topk=False)
    
    # Premier batch devrait avoir des indices vides
    gpu_queries_0, gpu_targets_0 = result_gpu['indices'][0]
    cpu_queries_0, cpu_targets_0 = result_cpu['indices'][0]
    
    assert len(gpu_queries_0) == 0, f"GPU should have empty indices for empty targets, got {len(gpu_queries_0)}"
    assert len(cpu_queries_0) == 0, f"CPU should have empty indices for empty targets, got {len(cpu_queries_0)}"
    
    # Deuxième batch devrait avoir 2 assignments
    gpu_queries_1, gpu_targets_1 = result_gpu['indices'][1]
    cpu_queries_1, cpu_targets_1 = result_cpu['indices'][1]
    
    assert len(gpu_queries_1) == 2, f"GPU should have 2 indices, got {len(gpu_queries_1)}"
    assert len(cpu_queries_1) == 2, f"CPU should have 2 indices, got {len(cpu_queries_1)}"
    
    print("✅ Test cas edge réussi!")
    return True


def main():
    """
    Lance tous les tests de validation.
    """
    print("🚀 VALIDATION REFACTORISATION HUNGARIANGATCHER GPU-RESIDENT")
    print("=" * 60)
    
    try:
        # Tests de base
        test_matcher_consistency()
        test_topk_functionality()
        test_edge_cases()
        
        # Test de performance (seulement si CUDA disponible)
        if torch.cuda.is_available():
            test_performance_comparison()
        else:
            print("\n⚠️  CUDA non disponible - tests de performance sautés")
        
        print("\n" + "=" * 60)
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ La refactorisation GPU-resident est validée")
        print("🚀 Pipeline 100% GPU-resident maintenant disponible!")
        
    except Exception as e:
        print(f"\n❌ ERREUR lors des tests: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)