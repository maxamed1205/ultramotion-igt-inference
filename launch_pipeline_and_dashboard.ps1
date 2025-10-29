# Script pour lancer la pipeline ET le dashboard ensemble
# Usage: .\launch_pipeline_and_dashboard.ps1

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  LANCEMENT PIPELINE + DASHBOARD" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

Write-Host "`nCe script va lancer :" -ForegroundColor Yellow
Write-Host "  1. Generateur de metriques KPI" -ForegroundColor White
Write-Host "  2. Dashboard web (port 8050)" -ForegroundColor White

Write-Host "`nConsultez ensuite : http://localhost:8050" -ForegroundColor Green
Write-Host "`nPour arreter : Ctrl+C dans cette fenetre" -ForegroundColor Yellow
Write-Host "=========================================`n" -ForegroundColor Cyan

# Verifier les dependances
$missing = @()
foreach ($dep in @("fastapi", "uvicorn")) {
    $installed = python -c "import $dep" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $dep
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Installation des dependances manquantes..." -ForegroundColor Yellow
    pip install -r requirements-dashboard.txt -q
}

# Ajouter src au PYTHONPATH
$env:PYTHONPATH = "$PSScriptRoot\src"

# Créer le dossier logs si nécessaire
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

# Lancer le générateur KPI en arrière-plan dans un nouveau terminal
Write-Host "[1/2] Demarrage du generateur de metriques..." -ForegroundColor Cyan
$pipelineProcess = Start-Process -FilePath "python" `
    -ArgumentList "tests\tests_gateway\generate_kpi_for_dashboard.py" `
    -PassThru `
    -WindowStyle Hidden

Write-Host "      Generateur PID: $($pipelineProcess.Id)" -ForegroundColor Gray

# Attendre que le générateur démarre et génère des logs
Write-Host "      Attente demarrage (3s)..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Vérifier que le générateur tourne
if ($pipelineProcess.HasExited) {
    Write-Host "`nERREUR: Le generateur s'est arrete immediatement !" -ForegroundColor Red
    Write-Host "Verifiez les logs pour les details" -ForegroundColor Yellow
    exit 1
}

# Lancer le dashboard
Write-Host "`n[2/2] Demarrage du dashboard web..." -ForegroundColor Cyan
Write-Host "      URL: http://localhost:8050" -ForegroundColor Green
Write-Host "`n=========================================`n" -ForegroundColor Cyan
Write-Host "Appuyez sur Ctrl+C pour arreter le dashboard ET le generateur`n" -ForegroundColor Yellow

try {
    # Lancer le dashboard et attendre
    python -m service.dashboard_service --port 8050 --host 0.0.0.0 --interval 1.0
}
catch {
    Write-Host "`nErreur: $_" -ForegroundColor Red
}
finally {
    # Arrêter le générateur à la sortie
    Write-Host "`n`nArret du generateur..." -ForegroundColor Yellow
    
    if (-not $pipelineProcess.HasExited) {
        Stop-Process -Id $pipelineProcess.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
    
    Write-Host "Generateur arrete." -ForegroundColor Green
    Write-Host "`nAu revoir !`n" -ForegroundColor Cyan
}
