# ============================================================================
# Script PowerShell : Lancer Pipeline + Dashboard en Parallèle
# ============================================================================
# Usage:
#   .\run_test_with_dashboard.ps1
#
# Ce script lance :
#   1. Dashboard FastAPI sur http://localhost:8050
#   2. Test de pipeline avec dataset reel
#
# Appuyez sur Ctrl+C pour arreter les deux processus
# ============================================================================

$separator = "=" * 80
Write-Host $separator -ForegroundColor Cyan
Write-Host "LANCEMENT PIPELINE + DASHBOARD" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan

# Chemin racine du projet
$ROOT = "C:\Users\maxam\Desktop\TM\ultramotion-igt-inference"
Set-Location $ROOT

# Configuration Python
$env:PYTHONPATH = "$ROOT\src"

# ============================================================================
# Job 1 : Lancer le Dashboard (arriere-plan)
# ============================================================================
Write-Host "[1/2] Demarrage du Dashboard (http://localhost:8050)..." -ForegroundColor Green

$dashboardJob = Start-Job -ScriptBlock {
    param($rootPath)
    Set-Location $rootPath
    $env:PYTHONPATH = "$rootPath\src"
    python -m service.dashboard_service
} -ArgumentList $ROOT

# Attendre 2 secondes pour que le dashboard demarre
Start-Sleep -Seconds 2

# Verifier que le dashboard est lance
if ($dashboardJob.State -eq "Running") {
    Write-Host "Dashboard demarre (Job ID: $($dashboardJob.Id))" -ForegroundColor Green
} else {
    Write-Host "Echec du demarrage du dashboard" -ForegroundColor Red
    Stop-Job $dashboardJob
    Remove-Job $dashboardJob
    exit 1
}

# ============================================================================
# Job 2 : Lancer le Test de Pipeline (avant-plan)
# ============================================================================
Write-Host "[2/2] Demarrage du test de pipeline avec dataset..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard accessible sur : http://localhost:8050" -ForegroundColor Yellow
Write-Host "Test en cours... (appuyez sur Ctrl+C pour arreter)" -ForegroundColor Yellow
Write-Host ""

try {
    # Lancer le test en avant-plan
    python tests\tests_gateway\test_gateway_dataset_mock.py
    
    # Test termine - garder le dashboard actif
    Write-Host ""
    Write-Host "Test termine ! Dashboard toujours actif sur http://localhost:8050" -ForegroundColor Green
    Write-Host "Appuyez sur Ctrl+C pour arreter le dashboard et quitter" -ForegroundColor Yellow
    Write-Host ""
    
    # Attendre indefiniment (jusqu'a Ctrl+C)
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    # Arreter le dashboard proprement
    Write-Host ""
    Write-Host "Arret du dashboard..." -ForegroundColor Yellow
    Stop-Job $dashboardJob -ErrorAction SilentlyContinue
    Remove-Job $dashboardJob -ErrorAction SilentlyContinue
    Write-Host "Tous les processus arretes" -ForegroundColor Green
}

Write-Host ""
Write-Host $separator -ForegroundColor Cyan
Write-Host "FIN" -ForegroundColor Cyan
Write-Host $separator -ForegroundColor Cyan
