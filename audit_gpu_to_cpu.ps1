# ===============================================
# 🧭 SCRIPT D'AUDIT GPU→CPU AUTOMATISÉ
# ===============================================
# Ce script analyse automatiquement les transferts GPU→CPU dans le projet
# et génère un rapport complet pour optimisation

param(
    [string]$OutputDir = ".",
    [switch]$IncludeTests = $true,
    [switch]$Verbose = $false
)

Write-Host "🧭 AUDIT DES TRANSFERTS GPU→CPU - ULTRAMOTION IGT" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Définir les chemins de recherche
$SearchPaths = @("src")
if ($IncludeTests) {
    $SearchPaths += "tests"
    Write-Host "🔍 Recherche dans: src/ et tests/" -ForegroundColor Green
} else {
    Write-Host "🔍 Recherche dans: src/ uniquement" -ForegroundColor Green
}

# Patterns de recherche pour conversions GPU→CPU
$Patterns = @(
    "\.cpu\(",
    "\.to\s*\(\s*['""]cpu['""]",
    "\.numpy\(",
    "\.detach\(",
    "\.item\("
)

Write-Host "🔎 Patterns recherchés: $($Patterns -join ', ')" -ForegroundColor Yellow

# Exécuter la recherche
Write-Host "`n⚡ Exécution du scan..." -ForegroundColor Magenta

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$auditFile = Join-Path $OutputDir "gpu_to_cpu_audit_$timestamp.txt"
$reportFile = Join-Path $OutputDir "gpu_to_cpu_report_$timestamp.csv"

try {
    # Recherche des conversions
    $results = Get-ChildItem -Recurse -Include *.py -Path $SearchPaths | 
               Select-String -Pattern $Patterns | 
               Select-Object Path, LineNumber, Line

    # Sauvegarder résultats bruts
    $results | Format-Table -Wrap | Out-File -Encoding utf8 $auditFile
    
    Write-Host "✅ Résultats sauvegardés: $auditFile" -ForegroundColor Green
    Write-Host "📊 Nombre total de conversions détectées: $($results.Count)" -ForegroundColor Yellow

    # Analyse par catégorie
    $stats = @{
        "MobileSAM" = 0
        "D-FINE" = 0
        "Orchestrator" = 0
        "Tests" = 0
        "Autres" = 0
    }

    foreach ($result in $results) {
        $path = $result.Path
        if ($path -like "*MobileSAM*") { $stats["MobileSAM"]++ }
        elseif ($path -like "*dfine*") { $stats["D-FINE"]++ }
        elseif ($path -like "*orchestrator*") { $stats["Orchestrator"]++ }
        elseif ($path -like "*test*") { $stats["Tests"]++ }
        else { $stats["Autres"]++ }
    }

    Write-Host "`n📈 RÉPARTITION PAR COMPOSANT:" -ForegroundColor Cyan
    foreach ($component in $stats.Keys) {
        if ($stats[$component] -gt 0) {
            $indicator = if ($stats[$component] -gt 3) { "🔴" } elseif ($stats[$component] -gt 1) { "🟠" } else { "🟢" }
            Write-Host "$indicator $component : $($stats[$component]) conversions" -ForegroundColor White
        }
    }

    # Identifier les zones critiques
    Write-Host "`n🚨 ZONES CRITIQUES IDENTIFIÉES:" -ForegroundColor Red
    $criticalFiles = $results | Where-Object { 
        $_.Path -like "*predictor*" -or 
        $_.Path -like "*orchestrator*" -or
        ($_.Path -like "*dfine_infer*" -and $_.Line -like "*detach().cpu().numpy()*")
    }

    if ($criticalFiles.Count -gt 0) {
        foreach ($critical in $criticalFiles) {
            $filename = Split-Path $critical.Path -Leaf
            Write-Host "⚠️  $filename:$($critical.LineNumber) - $($critical.Line.Trim())" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✅ Aucune zone critique majeure détectée" -ForegroundColor Green
    }

    # Recommandations automatiques
    Write-Host "`n💡 RECOMMANDATIONS AUTOMATIQUES:" -ForegroundColor Cyan
    
    if ($stats["MobileSAM"] -gt 2) {
        Write-Host "🔥 PRIORITÉ 1: Optimiser MobileSAM - $($stats['MobileSAM']) conversions détectées" -ForegroundColor Red
        Write-Host "   → Garder tenseurs GPU jusqu'au ResultPacket final" -ForegroundColor White
    }
    
    if ($stats["Orchestrator"] -gt 0) {
        Write-Host "🔥 PRIORITÉ 1: Optimiser Orchestrator - Chain GPU→CPU→GPU détectée" -ForegroundColor Red
        Write-Host "   → Passer tenseurs GPU directement à SAM" -ForegroundColor White
    }
    
    if ($stats["D-FINE"] -gt 2) {
        Write-Host "⚡ PRIORITÉ 2: Vérifier D-FINE - Conversions intermédiaires possibles" -ForegroundColor Yellow
        Write-Host "   → Différer conversions jusqu'à la sortie finale" -ForegroundColor White
    }

    Write-Host "`n🎯 OBJECTIF: Pipeline entièrement GPU-resident" -ForegroundColor Green
    Write-Host "   Input → D-FINE(GPU) → SAM(GPU) → PostProcess(GPU) → ResultPacket.cpu().numpy()" -ForegroundColor White

} catch {
    Write-Host "❌ Erreur lors de l'exécution: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ AUDIT TERMINÉ" -ForegroundColor Green
Write-Host "📋 Fichiers générés:" -ForegroundColor White
Write-Host "   - $auditFile" -ForegroundColor Gray
if (Test-Path $reportFile) {
    Write-Host "   - $reportFile" -ForegroundColor Gray
}

Write-Host "`n🔄 Pour réexécuter: .\audit_gpu_to_cpu.ps1 [-IncludeTests] [-Verbose]" -ForegroundColor Blue