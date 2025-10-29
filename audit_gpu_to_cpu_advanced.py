#!/usr/bin/env python3
"""
🧭 DIAGNOSTIC AVANCÉ DES TRANSFERTS GPU→CPU

Ce script analyse les patterns de transfert GPU→CPU dans le projet
et identifie les chaînes de conversions problématiques.
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import argparse

@dataclass
class GPUTransfer:
    """Représente un transfert GPU→CPU détecté"""
    file_path: str
    line_number: int
    function_name: str
    code_line: str
    transfer_type: str  # 'cpu', 'numpy', 'item', 'detach', 'to_cpu'
    severity: str  # 'critical', 'medium', 'low'
    tensor_name: str
    context: List[str]  # Lignes avant/après pour contexte

class GPUTransferAnalyzer:
    """Analyseur de transferts GPU→CPU"""
    
    # Patterns de détection
    PATTERNS = {
        'cpu': r'\.cpu\(\)',
        'to_cpu': r'\.to\s*\(\s*["\']cpu["\']\s*\)',
        'numpy': r'\.numpy\(\)',
        'detach': r'\.detach\(\)',
        'item': r'\.item\(\)'
    }
    
    # Fonctions critiques (performance impact élevé)
    CRITICAL_FUNCTIONS = {
        'predict', 'forward', 'inference', 'run_inference',
        'process_frame', 'predict_torch', 'generate'
    }
    
    # Fichiers critiques pour la performance
    CRITICAL_FILES = {
        'predictor.py', 'dfine_infer.py', 'orchestrator.py',
        'inference_sam.py', 'mobile_sam'
    }

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.transfers: List[GPUTransfer] = []
        
    def scan_files(self, include_tests: bool = True) -> List[GPUTransfer]:
        """Scan tous les fichiers Python pour les transferts GPU→CPU"""
        
        search_paths = [self.root_path / "src"]
        if include_tests:
            search_paths.append(self.root_path / "tests")
            
        print(f"🔍 Scanning {len(search_paths)} directories...")
        
        for search_path in search_paths:
            if search_path.exists():
                self._scan_directory(search_path)
                
        print(f"✅ Found {len(self.transfers)} GPU→CPU transfers")
        return self.transfers
    
    def _scan_directory(self, directory: Path):
        """Scan récursif d'un répertoire"""
        for py_file in directory.rglob("*.py"):
            self._analyze_file(py_file)
    
    def _analyze_file(self, file_path: Path):
        """Analyse un fichier Python pour les transferts"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                self._check_line_for_transfers(file_path, line_num, line, lines)
                
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
    
    def _check_line_for_transfers(self, file_path: Path, line_num: int, line: str, all_lines: List[str]):
        """Vérifie une ligne pour les patterns de transfert"""
        line_clean = line.strip()
        
        for transfer_type, pattern in self.PATTERNS.items():
            if re.search(pattern, line_clean):
                # Extraire le nom du tensor/variable
                tensor_name = self._extract_tensor_name(line_clean, pattern)
                
                # Trouver la fonction englobante
                function_name = self._find_function_name(all_lines, line_num)
                
                # Déterminer la sévérité
                severity = self._assess_severity(file_path, function_name, transfer_type, tensor_name)
                
                # Contexte (lignes avant/après)
                context = self._get_context(all_lines, line_num)
                
                transfer = GPUTransfer(
                    file_path=str(file_path.relative_to(self.root_path)),
                    line_number=line_num,
                    function_name=function_name,
                    code_line=line_clean,
                    transfer_type=transfer_type,
                    severity=severity,
                    tensor_name=tensor_name,
                    context=context
                )
                
                self.transfers.append(transfer)
    
    def _extract_tensor_name(self, line: str, pattern: str) -> str:
        """Extrait le nom du tensor depuis la ligne de code"""
        # Chercher le nom de variable avant le pattern
        match = re.search(r'(\w+)' + pattern.replace('\\', ''), line)
        if match:
            return match.group(1)
        return "unknown"
    
    def _find_function_name(self, lines: List[str], current_line: int) -> str:
        """Trouve le nom de la fonction qui contient la ligne actuelle"""
        for i in range(current_line - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('def '):
                match = re.match(r'def\s+(\w+)', line)
                if match:
                    return match.group(1)
            elif line.startswith('class '):
                # Si on trouve une classe avant une fonction, on s'arrête
                break
        return "unknown"
    
    def _assess_severity(self, file_path: Path, function_name: str, transfer_type: str, tensor_name: str) -> str:
        """Évalue la sévérité du transfert"""
        
        # Fichier critique ?
        is_critical_file = any(critical in str(file_path).lower() for critical in self.CRITICAL_FILES)
        
        # Fonction critique ?
        is_critical_function = function_name.lower() in self.CRITICAL_FUNCTIONS
        
        # Type de transfert coûteux ?
        expensive_transfers = {'numpy', 'cpu', 'to_cpu'}
        is_expensive = transfer_type in expensive_transfers
        
        # Tensor potentiellement lourd ?
        heavy_tensors = {'mask', 'image', 'frame', 'tensor', 'output', 'prediction'}
        is_heavy_tensor = any(heavy in tensor_name.lower() for heavy in heavy_tensors)
        
        # Calcul de sévérité
        if is_critical_file and is_critical_function and is_expensive and is_heavy_tensor:
            return "critical"
        elif (is_critical_file and is_expensive) or (is_critical_function and is_heavy_tensor):
            return "medium"
        else:
            return "low"
    
    def _get_context(self, lines: List[str], line_num: int, context_size: int = 2) -> List[str]:
        """Récupère le contexte autour de la ligne"""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        return [lines[i].rstrip() for i in range(start, end)]
    
    def analyze_chains(self) -> List[Dict]:
        """Analyse les chaînes de conversions (GPU→CPU→GPU)"""
        chains = []
        
        # Grouper par fichier
        files_transfers = {}
        for transfer in self.transfers:
            if transfer.file_path not in files_transfers:
                files_transfers[transfer.file_path] = []
            files_transfers[transfer.file_path].append(transfer)
        
        # Chercher des patterns de chaînes
        for file_path, file_transfers in files_transfers.items():
            if len(file_transfers) > 1:
                # Trier par numéro de ligne
                file_transfers.sort(key=lambda x: x.line_number)
                
                # Chercher des transferts proches (probablement liés)
                for i in range(len(file_transfers) - 1):
                    curr = file_transfers[i]
                    next_transfer = file_transfers[i + 1]
                    
                    # Si les transferts sont dans la même fonction et proches
                    if (curr.function_name == next_transfer.function_name and 
                        next_transfer.line_number - curr.line_number < 10):
                        
                        chains.append({
                            'file': file_path,
                            'function': curr.function_name,
                            'transfers': [curr, next_transfer],
                            'severity': 'critical' if any(t.severity == 'critical' for t in [curr, next_transfer]) else 'medium'
                        })
        
        return chains
    
    def generate_report(self) -> Dict:
        """Génère un rapport complet"""
        
        # Statistiques par sévérité
        severity_stats = {'critical': 0, 'medium': 0, 'low': 0}
        for transfer in self.transfers:
            severity_stats[transfer.severity] += 1
        
        # Statistiques par type
        type_stats = {}
        for transfer in self.transfers:
            type_stats[transfer.transfer_type] = type_stats.get(transfer.transfer_type, 0) + 1
        
        # Top fichiers problématiques
        file_stats = {}
        for transfer in self.transfers:
            file_stats[transfer.file_path] = file_stats.get(transfer.file_path, 0) + 1
        
        top_files = sorted(file_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Chaînes de conversions
        chains = self.analyze_chains()
        
        # Transferts critiques
        critical_transfers = [t for t in self.transfers if t.severity == 'critical']
        
        return {
            'summary': {
                'total_transfers': len(self.transfers),
                'critical_count': severity_stats['critical'],
                'medium_count': severity_stats['medium'],
                'low_count': severity_stats['low'],
                'chains_detected': len(chains)
            },
            'severity_breakdown': severity_stats,
            'transfer_types': type_stats,
            'top_problematic_files': top_files,
            'critical_transfers': [
                {
                    'file': t.file_path,
                    'line': t.line_number,
                    'function': t.function_name,
                    'code': t.code_line,
                    'tensor': t.tensor_name,
                    'type': t.transfer_type
                }
                for t in critical_transfers
            ],
            'conversion_chains': chains,
            'recommendations': self._generate_recommendations(critical_transfers, chains)
        }
    
    def _generate_recommendations(self, critical_transfers: List[GPUTransfer], chains: List[Dict]) -> List[str]:
        """Génère des recommandations automatiques"""
        recommendations = []
        
        # Recommandations basées sur les transferts critiques
        mobile_sam_transfers = [t for t in critical_transfers if 'mobile_sam' in t.file_path.lower()]
        if mobile_sam_transfers:
            recommendations.append(
                "🔥 PRIORITÉ 1: Modifier MobileSAM predictor pour retourner des tenseurs GPU au lieu de NumPy arrays"
            )
        
        orchestrator_transfers = [t for t in critical_transfers if 'orchestrator' in t.file_path.lower()]
        if orchestrator_transfers:
            recommendations.append(
                "🔥 PRIORITÉ 1: Éliminer conversion GPU→CPU dans orchestrator.py - passer tenseurs GPU directement à SAM"
            )
        
        dfine_transfers = [t for t in critical_transfers if 'dfine' in t.file_path.lower()]
        if dfine_transfers:
            recommendations.append(
                "⚡ PRIORITÉ 2: Optimiser D-FINE - différer conversions CPU jusqu'à la sortie finale"
            )
        
        # Recommandations basées sur les chaînes
        if chains:
            recommendations.append(
                f"🔄 CHAÎNES DÉTECTÉES: {len(chains)} chaînes GPU→CPU→GPU identifiées - fusionner en pipeline GPU-resident"
            )
        
        # Recommandation architecturale
        recommendations.append(
            "🎯 OBJECTIF: Implémenter pipeline entièrement GPU-resident avec conversion unique finale"
        )
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="Diagnostic avancé des transferts GPU→CPU")
    parser.add_argument("--no-tests", action="store_true", help="Exclure les fichiers de tests")
    parser.add_argument("--output", default="gpu_transfer_diagnostic.json", help="Fichier de sortie JSON")
    parser.add_argument("--verbose", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    print("🧭 DIAGNOSTIC AVANCÉ DES TRANSFERTS GPU→CPU")
    print("=" * 50)
    
    analyzer = GPUTransferAnalyzer()
    
    # Scanner les fichiers
    transfers = analyzer.scan_files(include_tests=not args.no_tests)
    
    # Générer le rapport
    report = analyzer.generate_report()
    
    # Afficher le résumé
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Total transferts: {report['summary']['total_transfers']}")
    print(f"   🔴 Critiques: {report['summary']['critical_count']}")
    print(f"   🟠 Moyens: {report['summary']['medium_count']}")
    print(f"   🟢 Faibles: {report['summary']['low_count']}")
    print(f"   🔄 Chaînes: {report['summary']['chains_detected']}")
    
    # Afficher les transferts critiques
    if report['critical_transfers']:
        print(f"\n🚨 TRANSFERTS CRITIQUES:")
        for ct in report['critical_transfers'][:5]:  # Top 5
            print(f"   ⚠️  {ct['file']}:{ct['line']} - {ct['function']}() - {ct['code'][:50]}...")
    
    # Afficher les recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    # Sauvegarder le rapport
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Rapport détaillé sauvegardé: {args.output}")
    print(f"🔄 Pour réexécuter: python audit_gpu_to_cpu_advanced.py [--no-tests] [--verbose]")

if __name__ == "__main__":
    main()