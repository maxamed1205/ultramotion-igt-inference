# ═══════════════════════════════════════════════════════════
# Script : Lancer Test Pipeline 1s + Dashboard
# ═══════════════════════════════════════════════════════════
# Ce script :
#  1. Lance test_gateway_real_pipeline_mock.py (1 seconde)
#  2. Lance le dashboard web sur http://localhost:8050
# ═══════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

# Configuration
$ROOT = $PSScriptRoot
$SRC = Join-Path $ROOT "src"
$TESTS = Join-Path $ROOT "tests"
$TESTS_GATEWAY = Join-Path $TESTS "tests_gateway"
$PIPELINE_TEST = Join-Path $TESTS_GATEWAY "test_gateway_real_pipeline_mock.py"
$SERVICE_DIR = Join-Path $SRC "service"
$DASHBOARD_SERVICE = Join-Path $SERVICE_DIR "dashboard_service.py"

# Définir PYTHONPATH
$env:PYTHONPATH = $SRC

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  TEST PIPELINE 1s + DASHBOARD" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ce script va :" -ForegroundColor Yellow
Write-Host "  1. Lancer test pipeline 1 seconde" -ForegroundColor Yellow
Write-Host "  2. Lancer dashboard web" -ForegroundColor Yellow
Write-Host ""
Write-Host "Dashboard accessible : http://localhost:8050" -ForegroundColor Green
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Étape 1 : Lancer le test de 1 seconde
Write-Host "[1/2] Lancement du test pipeline (1 seconde)..." -ForegroundColor Yellow
Write-Host "      Fichier: $PIPELINE_TEST" -ForegroundColor Gray

try {
    python $PIPELINE_TEST
    Write-Host "      Test termine avec succes" -ForegroundColor Green
} catch {
    Write-Host "      ERREUR lors du test: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "      Attente de 2 secondes avant dashboard..." -ForegroundColor Gray
Start-Sleep -Seconds 2

# Étape 2 : Lancer le dashboard
Write-Host ""
Write-Host "[2/2] Demarrage du dashboard web..." -ForegroundColor Yellow
Write-Host "      URL: http://localhost:8050" -ForegroundColor Green
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Appuyez sur Ctrl+C pour arreter le dashboard" -ForegroundColor Yellow
Write-Host ""

# Lancer le dashboard (bloquant)
python -m service.dashboard_service

Write-Host ""
Write-Host "Dashboard arrete." -ForegroundColor Yellow
