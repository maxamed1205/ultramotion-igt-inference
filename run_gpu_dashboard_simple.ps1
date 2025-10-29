# Script simplifie pour lancer test GPU + dashboard
# =====================================================

Write-Host "`n=== GPU Transfer Test + Dashboard ===" -ForegroundColor Cyan
Write-Host ""

# Nettoyer les anciens logs
Write-Host "[1/3] Nettoyage logs..." -ForegroundColor Yellow
if (Test-Path "logs") {
    Remove-Item "logs\*.log" -Force -ErrorAction SilentlyContinue
    Write-Host "  OK" -ForegroundColor Green
}

# Demarrer dashboard en arriere-plan
Write-Host "`n[2/3] Demarrage dashboard sur http://localhost:8051..." -ForegroundColor Yellow
$dashboard = Start-Process python -ArgumentList "src\service\dashboard_gpu_transfer.py" -NoNewWindow -PassThru
Start-Sleep -Seconds 2
Write-Host "  Dashboard PID: $($dashboard.Id)" -ForegroundColor Green

# Executer le test
Write-Host "`n[3/3] Execution du test (213 images completes)..." -ForegroundColor Yellow
Write-Host ""
python tests\tests_gateway\test_validate_step1_step2.py
$exitCode = $LASTEXITCODE

# Resultats
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "TEST REUSSI" -ForegroundColor Green
    Write-Host "`nDashboard actif sur: http://localhost:8051" -ForegroundColor Cyan
    Write-Host "Appuyez sur Ctrl+C pour arreter..." -ForegroundColor Yellow
    
    # Attendre Ctrl+C
    try {
        while ($true) { Start-Sleep -Seconds 1 }
    } catch {
        Write-Host "`nArret..." -ForegroundColor Yellow
    }
} else {
    Write-Host "TEST ECHOUE" -ForegroundColor Red
}

# Arreter dashboard
Write-Host "`nArret dashboard..." -ForegroundColor Yellow
Stop-Process -Id $dashboard.Id -Force -ErrorAction SilentlyContinue
Write-Host "Termine.`n" -ForegroundColor Green

exit $exitCode
