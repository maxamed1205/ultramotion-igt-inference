#!/usr/bin/env python3
"""
AUDIT GLOBAL PHASE 3 - POST-FIX GPU
Diagnostic complet des conversions GPU↔CPU après optimisations
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
from collections import defaultdict
import json

class GPUTransfer(NamedTuple):
    file_path: str
    line_num: int
    line_content: str
    expression: str
    context: str  # prod/test/postprocess
    severity: str  # 🔴/🟠/🟢
    recommendation: str

class GPUAuditor:
    """Auditeur avancé pour détecter les transferts GPU↔CPU"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.transfers = []
        
        # Patterns critiques à détecter
        self.gpu_cpu_patterns = {
            r'\.cpu\(\)': 'cpu_method',
            r'\.numpy\(\)': 'numpy_method', 
            r'\.to\("cpu"\)': 'to_cpu_explicit',
            r'\.to\(device="cpu"\)': 'to_cpu_device',
            r'\.to\(torch\.device\("cpu"\)\)': 'to_cpu_torch_device',
            r'\.detach\(\)': 'detach_method',
            r'\.item\(\)': 'item_method',
            r'torch\.from_numpy\(': 'from_numpy',
            r'np\.array\([^)]*\.cpu\(\)': 'np_array_cpu',
            r'device\s*=\s*"cpu"': 'device_cpu',
            r'device\s*=\s*torch\.device\("cpu"\)': 'device_torch_cpu'
        }
        
        # Contextes légitimes (patterns moins critiques)
        self.legitimate_contexts = [
            r'# .*test.*',
            r'# .*debug.*', 
            r'# .*kpi.*',
            r'# .*slicer.*',
            r'# .*visualization.*',
            r'# .*export.*',
            r'LOG\.',
            r'print\(',
            r'assert\s',
            r'if.*test',
            r'def test_',
            r'class.*Test',
            r'pytest',
            r'unittest'
        ]
    
    def analyze_file(self, file_path: Path) -> List[GPUTransfer]:
        """Analyse un fichier Python pour détecter les transferts GPU↔CPU"""
        transfers = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️ Erreur lecture {file_path}: {e}")
            return transfers
            
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Skip empty lines and pure comments
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            # Detect GPU↔CPU patterns
            for pattern, pattern_type in self.gpu_cpu_patterns.items():
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    expression = match.group(0)
                    
                    # Determine context and severity
                    context = self._determine_context(line, line_num, lines, file_path)
                    severity = self._determine_severity(expression, context, file_path)
                    recommendation = self._get_recommendation(expression, context, severity)
                    
                    transfer = GPUTransfer(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_num=line_num,
                        line_content=line_stripped,
                        expression=expression,
                        context=context,
                        severity=severity,
                        recommendation=recommendation
                    )
                    transfers.append(transfer)
                    
        return transfers
    
    def _determine_context(self, line: str, line_num: int, lines: List[str], file_path: Path) -> str:
        """Détermine le contexte d'utilisation (prod/test/postprocess)"""
        
        # Check file path for test indicators
        if 'test' in str(file_path).lower():
            return 'test'
            
        # Check current line for legitimate contexts
        for pattern in self.legitimate_contexts:
            if re.search(pattern, line, re.IGNORECASE):
                return 'test/debug'
                
        # Check surrounding lines for context
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 2)
        context_lines = lines[start:end]
        
        for context_line in context_lines:
            for pattern in self.legitimate_contexts:
                if re.search(pattern, context_line, re.IGNORECASE):
                    return 'test/debug'
                    
        # Check for specific function contexts
        if 'kpi' in line.lower() or 'log' in line.lower():
            return 'monitoring'
        if 'slicer' in line.lower() or 'export' in line.lower():
            return 'postprocess'
        if 'result' in line.lower() and 'packet' in line.lower():
            return 'postprocess'
            
        # Default: production inference
        return 'production'
    
    def _determine_severity(self, expression: str, context: str, file_path: Path) -> str:
        """Détermine la gravité de la conversion"""
        
        # Critical patterns in production inference
        critical_patterns = ['.cpu()', '.numpy()', '.to("cpu")', '.to(device="cpu")']
        medium_patterns = ['.detach()', '.item()']
        
        # Context-based severity
        if context in ['test', 'test/debug']:
            return '🟢'  # Acceptable in tests
        elif context in ['monitoring', 'postprocess']:
            if any(p in expression for p in critical_patterns):
                return '🟠'  # Tolerated for monitoring/export
            else:
                return '🟢'  # OK for monitoring
        else:  # production context
            if any(p in expression for p in critical_patterns):
                return '🔴'  # Critical in production
            elif any(p in expression for p in medium_patterns):
                return '🟠'  # Medium severity
            else:
                return '🟢'  # Low priority
                
        return '🟠'  # Default medium
    
    def _get_recommendation(self, expression: str, context: str, severity: str) -> str:
        """Génère une recommandation d'action"""
        
        if severity == '🟢':
            return "Acceptable - pas d'action requise"
        elif severity == '🟠':
            if context == 'monitoring':
                return "Toléré pour KPI/logs - surveiller performance"
            elif context == 'postprocess':
                return "Légitime pour export/Slicer - ok"
            else:
                return "Optimiser si possible - utiliser torch.no_grad()"
        else:  # 🔴
            if '.cpu()' in expression or '.numpy()' in expression:
                return "CRITIQUE - éliminer ou remplacer par GPU-resident"
            elif '.detach()' in expression:
                return "CRITIQUE - remplacer par torch.no_grad()"
            elif '.item()' in expression:
                return "CRITIQUE - vectoriser ou éliminer sync GPU→CPU"
            else:
                return "CRITIQUE - optimiser pour GPU pipeline"
    
    def scan_directories(self, directories: List[str]) -> List[GPUTransfer]:
        """Scanne les répertoires spécifiés"""
        all_transfers = []
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                print(f"⚠️ Répertoire non trouvé: {dir_path}")
                continue
                
            print(f"🔍 Scanning {directory}...")
            
            # Scan all Python files recursively
            for py_file in dir_path.rglob("*.py"):
                transfers = self.analyze_file(py_file)
                all_transfers.extend(transfers)
                
        return all_transfers
    
    def generate_report(self, transfers: List[GPUTransfer]) -> Dict:
        """Génère un rapport détaillé"""
        
        # Group by file and severity
        by_file = defaultdict(list)
        by_severity = defaultdict(list)
        
        for transfer in transfers:
            by_file[transfer.file_path].append(transfer)
            by_severity[transfer.severity].append(transfer)
        
        # Statistics
        total_transfers = len(transfers)
        critical_count = len(by_severity['🔴'])
        medium_count = len(by_severity['🟠'])
        low_count = len(by_severity['🟢'])
        
        # Progress assessment
        files_with_critical = len([f for f in by_file.keys() 
                                 if any(t.severity == '🔴' for t in by_file[f])])
        
        # Create report
        report = {
            'summary': {
                'total_transfers': total_transfers,
                'critical_transfers': critical_count,
                'medium_transfers': medium_count,
                'low_transfers': low_count,
                'files_scanned': len(by_file),
                'files_with_critical': files_with_critical,
                'progress_percent': max(0, 100 - (critical_count * 10))  # Rough estimate
            },
            'by_file': dict(by_file),
            'by_severity': dict(by_severity),
            'critical_files': [f for f in by_file.keys() 
                             if any(t.severity == '🔴' for t in by_file[f])]
        }
        
        return report
    
    def print_detailed_report(self, report: Dict):
        """Affiche un rapport détaillé formaté"""
        
        print("\n" + "="*80)
        print("🚀 AUDIT GLOBAL PHASE 3 - POST-FIX GPU")
        print("="*80)
        
        summary = report['summary']
        print(f"\n📊 RÉSUMÉ GLOBAL:")
        print(f"   Total conversions détectées: {summary['total_transfers']}")
        print(f"   🔴 Critiques (production):   {summary['critical_transfers']}")
        print(f"   🟠 Tolérées (monitoring):    {summary['medium_transfers']}")
        print(f"   🟢 Acceptables (tests):      {summary['low_transfers']}")
        print(f"   Fichiers scannés:            {summary['files_scanned']}")
        print(f"   Fichiers avec critiques:     {summary['files_with_critical']}")
        print(f"   Progression GPU pipeline:    {summary['progress_percent']}%")
        
        # Critical transfers details
        if summary['critical_transfers'] > 0:
            print(f"\n🔴 CONVERSIONS CRITIQUES À CORRIGER:")
            print("-" * 60)
            for transfer in report['by_severity']['🔴']:
                print(f"📄 {transfer.file_path}:{transfer.line_num}")
                print(f"   Expression: {transfer.expression}")
                print(f"   Ligne: {transfer.line_content[:80]}...")
                print(f"   Contexte: {transfer.context}")
                print(f"   Recommandation: {transfer.recommendation}")
                print()
        
        # File-by-file breakdown
        print(f"\n📋 DÉTAIL PAR FICHIER:")
        print("-" * 60)
        for file_path, transfers in report['by_file'].items():
            critical = sum(1 for t in transfers if t.severity == '🔴')
            medium = sum(1 for t in transfers if t.severity == '🟠')
            low = sum(1 for t in transfers if t.severity == '🟢')
            
            status = "🔴" if critical > 0 else "🟠" if medium > 0 else "🟢"
            print(f"{status} {file_path}: {critical}🔴 {medium}🟠 {low}🟢")
        
        # Recommendations
        print(f"\n🎯 RECOMMANDATIONS PRIORITAIRES:")
        print("-" * 60)
        if summary['critical_transfers'] == 0:
            print("✅ Aucune conversion critique détectée!")
            print("🎉 Pipeline GPU-resident opérationnelle!")
        else:
            print(f"1. Corriger {summary['critical_transfers']} conversions critiques")
            print(f"2. Focus sur {summary['files_with_critical']} fichiers prioritaires")
            print("3. Utiliser torch.no_grad() au lieu de .detach()")
            print("4. Vectoriser les .item() en boucles")
            print("5. Maintenir les tenseurs sur GPU jusqu'au ResultPacket")
            
def main():
    """Point d'entrée principal"""
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Directories to scan
    directories = [
        "src/core/inference",
        "src/core/preprocessing"
    ]
    
    print("🔍 DÉMARRAGE AUDIT GLOBAL PHASE 3")
    print(f"📁 Projet: {project_root}")
    
    auditor = GPUAuditor(project_root)
    transfers = auditor.scan_directories(directories)
    
    # Generate and display report
    report = auditor.generate_report(transfers)
    auditor.print_detailed_report(report)
    
    # Save JSON report
    report_file = "gpu_audit_phase3_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        # Convert NamedTuples to dicts for JSON serialization
        json_report = {
            'summary': report['summary'],
            'transfers': [t._asdict() for t in transfers],
            'critical_files': report['critical_files']
        }
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Rapport détaillé sauvegardé: {report_file}")
    
    # Return status code
    return 0 if report['summary']['critical_transfers'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())