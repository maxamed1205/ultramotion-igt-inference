# =====================================================================
# Script: run_dashboard_continuous.ps1
# Lance le dashboard unifié et le garde actif en continu
# =====================================================================

# Avoid weird encoding in console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Dashboard Unifié - Mode Continu" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Python est installé
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python trouvé: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    exit 1
}

# Créer le répertoire logs si nécessaire
$logsDir = "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "✓ Répertoire logs créé" -ForegroundColor Green
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Démarrage du Dashboard..." -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard accessible sur: http://localhost:8050" -ForegroundColor Green
Write-Host "Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Yellow
Write-Host ""

# Lancer le dashboard directement (pas en arrière-plan)
try {
    python -m service.dashboard_unified --port 8050 --host 0.0.0.0 --interval 1.0
} catch {
    Write-Host ""
    Write-Host "Dashboard arrêté par l'utilisateur" -ForegroundColor Yellow
} finally {
    Write-Host ""
    Write-Host "Dashboard fermé !" -ForegroundColor Green
}