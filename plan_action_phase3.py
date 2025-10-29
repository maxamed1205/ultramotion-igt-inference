#!/usr/bin/env python3
"""
PLAN D'ACTION PHASE 3 - CORRECTIONS CRITIQUES
Basé sur l'audit gpu_audit_phase3_comprehensive.py
"""

print("🎯 PLAN D'ACTION PHASE 3 - CORRECTIONS CRITIQUES GPU")
print("=" * 60)

corrections_critiques = [
    {
        "fichier": "src/core/inference/dfine_infer.py",
        "problemes": [
            {"ligne": 219, "expression": ".cpu().numpy()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 299, "expression": ".cpu().numpy()", "priorite": "🔴 CRITIQUE"}
        ],
        "solution": "Ajouter paramètre return_gpu_tensor=True par défaut",
        "impact": "Élimination des conversions finales en mode production"
    },
    {
        "fichier": "src/core/inference/d_fine/matcher.py", 
        "problemes": [
            {"ligne": 185, "expression": "C.cpu()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 197, "expression": ".cpu().numpy()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 209, "expression": "C.cpu()", "priorite": "🔴 CRITIQUE"}
        ],
        "solution": "Activer use_gpu_match=True par défaut + éliminer fallback CPU",
        "impact": "Matching 100% GPU-resident, gain performance majeur"
    },
    {
        "fichier": "src/core/inference/engine/orchestrator.py",
        "problemes": [
            {"ligne": 151, "expression": ".detach().cpu().numpy()", "priorite": "🔴 CRITIQUE"}
        ],
        "solution": "Remplacer par GPU tensor jusqu'à ResultPacket final",
        "impact": "Pipeline orchestration 100% GPU"
    },
    {
        "fichier": "src/core/inference/MobileSAM/mobile_sam/predictor.py",
        "problemes": [
            {"ligne": 195, "expression": ".detach().cpu().numpy()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 196, "expression": ".detach().cpu().numpy()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 197, "expression": ".detach().cpu().numpy()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 202, "expression": ".detach().cpu().numpy()", "priorite": "🔴 CRITIQUE"},
            {"ligne": 203, "expression": ".detach().cpu().numpy()", "priorite": "🔴 CRITIQUE"}
        ],
        "solution": "Ajouter flag return_tensors=True pour mode GPU-resident",
        "impact": "MobileSAM outputs restent sur GPU"
    },
    {
        "fichier": "src/core/preprocessing/cpu_to_gpu.py",
        "problemes": [
            {"ligne": 426, "expression": "buf.numpy()", "priorite": "🔴 CRITIQUE"}
        ],
        "solution": "Finaliser optimisation fastpath avec gestion test_mode",
        "impact": "Transfert CPU→GPU optimal sans référence NumPy persistante"
    }
]

print("📋 CORRECTIONS PRIORITAIRES:")
print("-" * 40)

for i, correction in enumerate(corrections_critiques, 1):
    print(f"\n{i}. 📄 {correction['fichier']}")
    print(f"   🎯 Solution: {correction['solution']}")
    print(f"   💡 Impact: {correction['impact']}")
    print(f"   🔧 Problèmes:")
    for probleme in correction['problemes']:
        print(f"      L{probleme['ligne']}: {probleme['expression']} - {probleme['priorite']}")

print(f"\n📊 STATISTIQUES:")
print(f"   Total fichiers critiques: {len(corrections_critiques)}")
total_problemes = sum(len(c['problemes']) for c in corrections_critiques)
print(f"   Total conversions critiques: {total_problemes}")
print(f"   Temps estimé: ~2-3 heures")
print(f"   Gain performance attendu: ~25-30%")

print(f"\n⚡ ORDRE D'EXÉCUTION RECOMMANDÉ:")
print("1. dfine_infer.py - Return statements (impact immédiat)")
print("2. matcher.py - GPU matching (gain performance majeur)")  
print("3. predictor.py - MobileSAM outputs (pipeline SAM)")
print("4. orchestrator.py - Pipeline coordination")
print("5. cpu_to_gpu.py - Finalisation fastpath")

print(f"\n🎯 OBJECTIF FINAL:")
print("✅ 0 conversions critiques en production")
print("✅ Pipeline 100% GPU-resident jusqu'à ResultPacket")
print("✅ Unique .cpu().numpy() final pour export Slicer")
print("✅ Performance optimale RTX 4090")

print(f"\n🚀 PRÊT POUR CORRECTIONS!")