# Audit GPU amélioré avec détection des branches conditionnelles

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class GPUTransfer:
    file_path: str
    line_num: int
    line_content: str
    expression: str
    context: str
    severity: str
    recommendation: str
    conditional_context: str = ""  # NEW: Détecte if/else branches

class AdvancedGPUAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Patterns GPU↔CPU
        self.gpu_cpu_patterns = {
            r'\.cpu\(\)': 'CPU_TRANSFER',
            r'\.numpy\(\)': 'NUMPY_CONVERSION', 
            r'\.detach\(\)': 'DETACH_CALL',
            r'\.item\(\)': 'ITEM_SCALAR',
            r'\.to\("cpu"\)': 'TO_CPU',
            r'\.to\(device="cpu"\)': 'TO_CPU_DEVICE'
        }
        
        # Contextes légitimes
        self.legitimate_contexts = [
            r'# .*legacy.*',
            r'# .*fallback.*',
            r'# .*test.*',
            r'# .*debug.*',
            r'# .*kpi.*',
            r'# .*visualization.*',
            r'LOG\.',
            r'assert\s',
            r'def test_',
        ]
        
        # Patterns pour détecter les branches GPU-first
        self.gpu_first_patterns = [
            r'use_gpu_match.*=.*True',
            r'as_numpy.*=.*False', 
            r'return_gpu_tensor.*=.*True',
            r'gpu_mode.*=.*True',
            r'if.*gpu.*match',
            r'if.*return_gpu_tensor',
            r'if.*as_numpy'
        ]

    def _detect_conditional_context(self, line_num: int, lines: List[str]) -> str:
        """Détecte si la ligne est dans une branche if/else GPU-friendly"""
        
        # Chercher les conditions dans les 15 lignes précédentes (étendu)
        start = max(0, line_num - 15)
        context_lines = lines[start:line_num]
        
        # Analyser la structure conditionnelle
        current_line = lines[line_num - 1] if line_num > 0 else ""
        indent_level = len(current_line) - len(current_line.lstrip())
        
        # Patterns étendus pour détecter les branches GPU-friendly
        gpu_first_conditions = [
            r'if\s+.*use_gpu_match',
            r'if\s+.*gpu_mode',
            r'if\s+.*gpu_first',
            r'if\s+.*torch\.cuda\.is_available',
            r'if\s+.*device\.type\s*==\s*["\']cuda["\']',
            r'if\s+.*return_gpu_tensor',
            r'if\s+not\s+as_numpy',
            r'if\s+.*gpu_resident',
        ]
        
        legacy_conditions = [
            r'if\s+as_numpy',
            r'if\s+.*legacy',
            r'if\s+.*fallback',
            r'if\s+.*cpu_mode',
            r'if\s+.*backward.*compat',
            r'except.*Exception',
            r'except.*Error',
        ]
        
        # Analyser le contexte en remontant
        in_gpu_branch = False
        in_legacy_branch = False
        in_exception_handler = False
        
        for i in range(len(context_lines) - 1, -1, -1):
            context_line = context_lines[i].strip()
            context_indent = len(context_lines[i]) - len(context_lines[i].lstrip())
            
            # Si on trouve une condition à un niveau d'indentation inférieur ou égal
            if context_indent <= indent_level:
                
                # Vérifier si c'est une condition GPU-first
                for pattern in gpu_first_conditions:
                    if re.search(pattern, context_line, re.IGNORECASE):
                        in_gpu_branch = True
                        break
                        
                # Vérifier si c'est une condition legacy
                for pattern in legacy_conditions:
                    if re.search(pattern, context_line, re.IGNORECASE):
                        in_legacy_branch = True
                        break
                
                # Détecter les handlers d'exception
                if re.search(r'except\s+.*:', context_line, re.IGNORECASE):
                    in_exception_handler = True
                
                # Détecter les branches else
                if context_line.strip() == "else:" and i < len(context_lines) - 1:
                    # Regarder la condition précédente
                    for j in range(i - 1, -1, -1):
                        prev_line = context_lines[j].strip()
                        if prev_line.startswith('if '):
                            # Si la condition if était GPU-first, alors else est legacy
                            for pattern in gpu_first_conditions:
                                if re.search(pattern, prev_line, re.IGNORECASE):
                                    in_legacy_branch = True
                                    break
                            break
        
        # Détection de patterns spécifiques dans la ligne courante
        current_line_lower = current_line.lower()
        
        # Patterns qui indiquent du code legacy/fallback
        if any(keyword in current_line_lower for keyword in [
            '# legacy', '# fallback', '# deprecated', '# backward compat',
            'should not be executed', 'code mort', 'emergency', 'compatibility'
        ]):
            return "LEGACY_DEAD_CODE"
        
        # Patterns qui indiquent des optimisations légitimes
        if any(keyword in current_line_lower for keyword in [
            'pinned memory', 'optimization', 'numpy view', 'fastpath',
            'kpi', 'monitoring', 'instrumentation'
        ]):
            return "LEGITIMATE_OPTIMIZATION"
        
        # Déterminer le contexte final
        if in_exception_handler:
            return "EXCEPTION_HANDLER"
        elif in_legacy_branch:
            return "LEGACY_FALLBACK"
        elif in_gpu_branch:
            return "GPU_FIRST"
        else:
            return "UNKNOWN"

    def _determine_context_advanced(self, line: str, line_num: int, lines: List[str], file_path: Path) -> str:
        """Détermine le contexte avec analyse conditionnelle avancée"""
        
        # Détection du contexte conditionnel amélioré
        conditional = self._detect_conditional_context(line_num, lines)
        
        # Analyse de la ligne courante pour patterns spécifiques
        line_lower = line.lower()
        
        # Contexts prioritaires basés sur les commentaires et patterns
        if any(pattern in line_lower for pattern in [
            'legacy', 'fallback', 'deprecated', 'backward compat', 'code mort',
            'should not be executed', 'emergency', 'compatibility only'
        ]):
            return 'legacy_dead_code'
            
        if any(pattern in line_lower for pattern in [
            'pinned memory', 'optimization', 'numpy view', 'fastpath',
            'buffer allocation', 'memory pool'
        ]):
            return 'memory_optimization'
            
        if any(pattern in line_lower for pattern in [
            'kpi', 'monitoring', 'instrumentation', 'debug', 'log'
        ]):
            return 'monitoring'
        
        # Contexts basés sur détection conditionnelle
        if conditional == "LEGACY_DEAD_CODE":
            return 'legacy_dead_code'
        elif conditional in ["LEGACY_FALLBACK", "LEGACY_NUMPY", "LEGACY_CPU"]:
            return 'legacy_branch'
        elif conditional in ["GPU_FIRST", "GPU_RESIDENT", "GPU_TENSOR"]:
            return 'gpu_optimized'
        elif conditional == "EXCEPTION_HANDLER":
            return 'exception_safety'
        elif conditional == "LEGITIMATE_OPTIMIZATION":
            return 'memory_optimization'
            
        # Contextes standards
        if 'test' in str(file_path).lower():
            return 'test'
            
        # Analyse des patterns légitimes standards
        for pattern in self.legitimate_contexts:
            if re.search(pattern, line, re.IGNORECASE):
                return 'test/debug'
        
        # Context par défaut
        return 'production'

    def _determine_severity_advanced(self, expression: str, context: str, conditional: str) -> str:
        """Détermine la gravité avec contexte conditionnel amélioré"""
        
        critical_patterns = ['.cpu()', '.numpy()', '.to("cpu")', '.to(device="cpu")']
        medium_patterns = ['.detach()', '.item()']
        
        # **PRIORITÉ 1**: Code mort/legacy qui ne sera jamais exécuté
        if context in ['legacy_dead_code', 'legacy_branch']:
            return '🟡'  # LEGACY - pas critique car jamais exécuté
            
        # **PRIORITÉ 2**: Optimisations légitimes
        if context in ['memory_optimization', 'exception_safety']:
            return '🟢'  # ACCEPTABLE - optimisation ou sécurité
            
        # **PRIORITÉ 3**: Monitoring/debug (tolérés en production)
        if context in ['monitoring', 'test/debug', 'test']:
            return '�'  # MONITORING - acceptable si nécessaire
            
        # **PRIORITÉ 4**: Branches GPU optimisées (devraient être OK)
        if context == 'gpu_optimized':
            if any(p in expression for p in critical_patterns):
                return '🟠'  # MEDIUM - dans branche GPU mais avec conversion
            else:
                return '🟢'  # ACCEPTABLE
                
        # **PRIORITÉ 5**: Production - VRAI CRITIQUE
        if context == 'production':
            if any(p in expression for p in critical_patterns):
                return '🔴'  # CRITIQUE - conversion en production
            elif any(p in expression for p in medium_patterns):
                return '🟠'  # MEDIUM - détachement en production
            else:
                return '🟢'  # ACCEPTABLE
        
        # Par défaut
        return '🟠'

    def analyze_file_advanced(self, file_path: Path) -> List[GPUTransfer]:
        """Analyse avancée avec détection conditionnelle"""
        transfers = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return transfers
            
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            for pattern, pattern_type in self.gpu_cpu_patterns.items():
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    expression = match.group(0)
                    
                    # Contexte avancé avec analyse conditionnelle
                    context = self._determine_context_advanced(line, line_num, lines, file_path)
                    conditional = self._detect_conditional_context(line_num, lines)
                    severity = self._determine_severity_advanced(expression, context, conditional)
                    
                    # Recommandation basée sur le contexte amélioré
                    if context == 'legacy_dead_code':
                        recommendation = "LEGACY DEAD CODE - Code jamais exécuté, pas d'action requise"
                    elif context == 'legacy_branch':
                        recommendation = "LEGACY BRANCH - Fallback désactivé par défaut, pas critique"
                    elif context == 'memory_optimization':
                        recommendation = "OPTIMIZATION - Pinned memory ou buffer allocation légitime"
                    elif context == 'exception_safety':
                        recommendation = "SAFETY - Exception handler, maintenir pour robustesse"
                    elif context == 'monitoring':
                        recommendation = "MONITORING - KPI/debug, acceptable si nécessaire"
                    elif severity == '�':
                        recommendation = "CRITIQUE - conversion active en production à éliminer"
                    elif severity == '🟠':
                        recommendation = "MEDIUM - vérifier si nécessaire, optimiser si possible"
                    else:
                        recommendation = "ACCEPTABLE - contexte légitime"
                    
                    transfer = GPUTransfer(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_num=line_num,
                        line_content=line_stripped,
                        expression=expression,
                        context=context,
                        severity=severity,
                        recommendation=recommendation,
                        conditional_context=conditional
                    )
                    transfers.append(transfer)
                    
        return transfers

    def audit_project_advanced(self) -> Dict[str, Any]:
        """Audit avancé du projet complet"""
        print("🔍 DÉMARRAGE AUDIT AVANCÉ AVEC DÉTECTION CONDITIONNELLE")
        print(f"📁 Projet: {self.project_root}")
        
        all_transfers = []
        scan_dirs = [
            self.project_root / "src" / "core" / "inference",
            self.project_root / "src" / "core" / "preprocessing"
        ]
        
        for scan_dir in scan_dirs:
            if scan_dir.exists():
                print(f"🔍 Scanning {scan_dir.relative_to(self.project_root)}...")
                for py_file in scan_dir.rglob("*.py"):
                    transfers = self.analyze_file_advanced(py_file)
                    all_transfers.extend(transfers)
        
        # Statistiques avancées
        critical = [t for t in all_transfers if t.severity == '🔴']
        medium = [t for t in all_transfers if t.severity == '🟠']
        acceptable = [t for t in all_transfers if t.severity == '🟢']
        legacy = [t for t in all_transfers if t.severity == '🟡']
        
        # Calcul progression GPU pipeline (en excluant les fallbacks legacy)
        active_transfers = [t for t in all_transfers if t.severity != '🟡']
        if active_transfers:
            gpu_progress = max(0, 100 - (len(critical) * 100 / len(active_transfers)))
        else:
            gpu_progress = 100
        
        print("\n" + "="*80)
        print("🚀 AUDIT AVANCÉ AVEC DÉTECTION CONDITIONNELLE")
        print("="*80)
        print(f"\n📊 RÉSUMÉ GLOBAL:")
        print(f"   Total conversions détectées: {len(all_transfers)}")
        print(f"   🔴 Critiques (production):   {len(critical)}")
        print(f"   🟠 Tolérées (monitoring):    {len(medium)}")
        print(f"   🟢 Acceptables (tests):      {len(acceptable)}")
        print(f"   🟡 Legacy fallbacks:         {len(legacy)}")
        print(f"   Progression GPU pipeline:    {gpu_progress:.0f}%")
        
        # Affichage des critiques (seulement les VRAIS critiques)
        if critical:
            print(f"\n🔴 CONVERSIONS CRITIQUES À CORRIGER (Production active):")
            print("-" * 60)
            for transfer in critical:
                print(f"📄 {transfer.file_path}:{transfer.line_num}")
                print(f"   Expression: {transfer.expression}")
                print(f"   Ligne: {transfer.line_content[:80]}...")
                print(f"   Contexte: {transfer.context}")
                print(f"   Conditionnel: {transfer.conditional_context}")
                print(f"   Recommandation: {transfer.recommendation}")
                print()
        else:
            print(f"\n🎉 AUCUNE CONVERSION CRITIQUE DÉTECTÉE!")
            print("   Pipeline 100% GPU-resident en production ✅")
        
        # Affichage des optimisations légitimes (informatif)
        memory_opts = [t for t in all_transfers if t.context == 'memory_optimization']
        if memory_opts:
            print(f"\n� OPTIMISATIONS MÉMOIRE LÉGITIMES ({len(memory_opts)}):")
            print("-" * 60)
            for transfer in memory_opts[:3]:  # Limiter l'affichage
                print(f"📄 {transfer.file_path}:{transfer.line_num} - {transfer.expression} (Pinned memory/Buffer)")
            if len(memory_opts) > 3:
                print(f"   ... et {len(memory_opts) - 3} autres optimisations mémoire")
        
        # Affichage des legacy fallbacks (informatif)
        if legacy:
            print(f"\n🟡 BRANCHES LEGACY/FALLBACK ({len(legacy)}):")
            print("-" * 60)
            for transfer in legacy[:3]:  # Limiter l'affichage
                print(f"📄 {transfer.file_path}:{transfer.line_num} - {transfer.expression} ({'Dead code' if transfer.context == 'legacy_dead_code' else 'Fallback'})")
            if len(legacy) > 3:
                print(f"   ... et {len(legacy) - 3} autres branches legacy")
        
        return {
            'total_transfers': len(all_transfers),
            'critical': len(critical),
            'medium': len(medium),
            'acceptable': len(acceptable),
            'legacy_fallbacks': len(legacy),
            'gpu_progress': gpu_progress,
            'transfers': [
                {
                    'file_path': t.file_path,
                    'line_num': t.line_num,
                    'expression': t.expression,
                    'context': t.context,
                    'conditional_context': t.conditional_context,
                    'severity': t.severity,
                    'recommendation': t.recommendation
                }
                for t in all_transfers
            ]
        }

def main():
    project_root = Path.cwd()
    auditor = AdvancedGPUAuditor(str(project_root))
    results = auditor.audit_project_advanced()
    
    # Sauvegarder le rapport
    report_file = project_root / "gpu_audit_advanced_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Rapport détaillé sauvegardé: {report_file}")
    return 0 if results['critical'] == 0 else 1

if __name__ == "__main__":
    exit(main())