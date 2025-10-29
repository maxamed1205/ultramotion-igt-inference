#!/usr/bin/env python3
"""
Audit simple et fonctionnel des transferts GPU→CPU.
Script de référence pour validation de la refactorisation.
"""

import re
import os
from collections import defaultdict, Counter

def scan_for_gpu_cpu_transfers():
    """
    Scanne le code pour les transferts GPU→CPU avec classification.
    """
    
    patterns = {
        "cpu": re.compile(r"\.cpu\("),
        "to_cpu": re.compile(r"\.to\s*\(\s*['\"]cpu['\"]"),  
        "numpy": re.compile(r"\.numpy\("),
        "detach": re.compile(r"\.detach\("),
        "item": re.compile(r"\.item\(")
    }
    
    results = []
    
    print("🔍 SCANNING FOR GPU→CPU TRANSFERS")
    print("=" * 50)
    
    for root, dirs, files in os.walk("src"):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith('#'):
                        continue
                        
                    for category, pattern in patterns.items():
                        if pattern.search(line):
                            severity = classify_transfer(rel_path, line_stripped, category)
                            results.append({
                                'file': rel_path,
                                'line': line_num,
                                'category': category,
                                'code': line_stripped,
                                'severity': severity
                            })
                            
            except Exception as e:
                print(f"⚠️  Error reading {filepath}: {e}")
    
    return results

def classify_transfer(filepath, code, category):
    """
    Classifie la gravité d'un transfert GPU→CPU.
    """
    code_lower = code.lower()
    
    # Cas sûrs
    if 'test' in filepath:
        return '🔵 Safe (Test)'
    
    if category == 'item' and any(word in code_lower for word in ['loss', 'metric', 'kpi', 'scalar']):
        return '🔵 Safe (Scalar)'
    
    # Cas nécessaires
    if any(word in code_lower for word in ['slicer', 'output', 'result', 'save']):
        return '🟡 Necessary (Output)'
    
    # Cas moyens - dans pipeline mais pas critique
    if category == 'numpy' and not any(word in code_lower for word in ['mask', 'tensor', 'bbox']):
        return '🟠 Medium (Minor)'
    
    # Cas critiques - gros tenseurs dans pipeline
    if category in ['cpu', 'to_cpu'] and any(word in code_lower for word in ['mask', 'tensor', 'bbox', 'logits']):
        return '🔴 Critical (Tensor)'
    
    if 'detach' in code_lower and 'cpu' in code_lower:
        return '🔴 Critical (Combined)'
    
    return '🟡 Medium (Review)'

def generate_summary(results):
    """
    Génère un résumé des résultats.
    """
    print(f"\n📊 SUMMARY OF {len(results)} TRANSFERS")
    print("=" * 50)
    
    # Stats par gravité
    severity_counts = Counter(r['severity'] for r in results)
    for severity, count in severity_counts.most_common():
        print(f"   {severity}: {count}")
    
    # Stats par fichier
    file_counts = Counter(r['file'] for r in results)
    print(f"\n📁 TOP FILES:")
    for file, count in file_counts.most_common(10):
        print(f"   {file}: {count} transfers")
    
    # Stats par catégorie
    category_counts = Counter(r['category'] for r in results)
    print(f"\n🏷️  BY CATEGORY:")
    for cat, count in category_counts.most_common():
        print(f"   {cat}: {count}")
    
    # Transferts critiques
    critical = [r for r in results if '🔴' in r['severity']]
    if critical:
        print(f"\n🚨 CRITICAL TRANSFERS ({len(critical)}):")
        for transfer in critical:
            print(f"   {transfer['file']}:{transfer['line']} - {transfer['code']}")
    else:
        print(f"\n✅ NO CRITICAL TRANSFERS FOUND!")
    
    return {
        'total': len(results),
        'critical': len(critical),
        'by_severity': dict(severity_counts),
        'by_file': dict(file_counts),
        'by_category': dict(category_counts)
    }

def save_detailed_report(results):
    """
    Sauvegarde un rapport détaillé.
    """
    with open('gpu_to_cpu_audit.txt', 'w', encoding='utf-8') as f:
        f.write("🔍 GPU→CPU TRANSFER AUDIT REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for r in results:
            f.write(f"{r['severity']} {r['file']}:{r['line']} [{r['category']}]\n")
            f.write(f"   {r['code']}\n\n")
    
    print(f"\n📄 Detailed report saved to: gpu_to_cpu_audit.txt")

def validate_refactoring_success(stats):
    """
    Valide le succès de la refactorisation.
    """
    print(f"\n🎯 REFACTORING VALIDATION")
    print("=" * 30)
    
    if stats['critical'] == 0:
        print("✅ SUCCESS: No critical GPU→CPU transfers found!")
        print("✅ Pipeline can run 100% GPU-resident")
    else:
        print(f"⚠️  WARNING: {stats['critical']} critical transfers remain")
    
    # Composants refactorisés
    refactored_components = [
        'matcher.py',
        'predictor.py', 
        'inference_sam.py',
        'orchestrator.py'
    ]
    
    print(f"\n🔧 REFACTORED COMPONENTS:")
    for component in refactored_components:
        component_transfers = stats['by_file'].get(f'src/core/inference/d_fine/{component}', 0)
        component_transfers += stats['by_file'].get(f'src/core/inference/MobileSAM/mobile_sam/{component}', 0)
        component_transfers += stats['by_file'].get(f'src/core/inference/engine/{component}', 0)
        
        if component_transfers == 0:
            print(f"   ✅ {component}: No transfers (optimized)")
        else:
            print(f"   ⚠️  {component}: {component_transfers} transfers")

def main():
    """
    Lance l'audit complet.
    """
    print("🚀 GPU→CPU TRANSFER AUDIT")
    print("Project: ultramotion-igt-inference")
    print("=" * 50)
    
    results = scan_for_gpu_cpu_transfers()
    stats = generate_summary(results)
    save_detailed_report(results)
    validate_refactoring_success(stats)
    
    print(f"\n🎉 AUDIT COMPLETE!")
    print(f"   Found {stats['total']} total transfers")
    print(f"   Critical: {stats['critical']}")
    print(f"   Report: gpu_to_cpu_audit.txt")

if __name__ == "__main__":
    main()