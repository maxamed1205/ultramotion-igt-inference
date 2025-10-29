# Script PowerShell pour lancer Dashboard + Test Pipeline
# ========================================================

Write-Host "🎯 ULTRAMOTION IGT - DASHBOARD + PIPELINE TEST" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$ROOT = Get-Location
$env:PYTHONPATH = "$ROOT\src"

Write-Host "📂 Répertoire: $ROOT" -ForegroundColor Green
Write-Host "🐍 PYTHONPATH: $($env:PYTHONPATH)" -ForegroundColor Green

# 1. Lancer le dashboard en arrière-plan
Write-Host "`n🌐 Lancement du dashboard web..." -ForegroundColor Yellow

$dashboardJob = Start-Job -ScriptBlock {
    param($pythonPath, $rootPath)
    $env:PYTHONPATH = $pythonPath
    Set-Location $rootPath
    python -m service.dashboard_unified --port 8050 --host 0.0.0.0
} -ArgumentList $env:PYTHONPATH, $ROOT

Write-Host "📊 Dashboard lancé (Job ID: $($dashboardJob.Id))" -ForegroundColor Green
Write-Host "🌐 URL: http://localhost:8050" -ForegroundColor Cyan

# 2. Attendre le démarrage du dashboard
Write-Host "`n⏳ Attente du démarrage du dashboard (3s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 3. Lancer le test avec dashboard désactivé (pour éviter conflit)
Write-Host "`n🚀 Lancement du test de pipeline..." -ForegroundColor Yellow

try {
    # Modifier temporairement le test pour désactiver son dashboard interne
    $testFile = "$ROOT\tests\tests_gateway\test_gateway_dataset_mock.py"
    $content = Get-Content $testFile -Raw
    $modifiedContent = $content -replace 'ENABLE_DASHBOARD = True', 'ENABLE_DASHBOARD = False'
    $modifiedContent | Set-Content "$testFile.temp"
    
    Write-Host "⚙️ Exécution du test..." -ForegroundColor Green
    python "$testFile.temp"
    
    Write-Host "`n✅ Test terminé avec succès!" -ForegroundColor Green
    
} catch {
    Write-Host "`n❌ Erreur lors du test: $_" -ForegroundColor Red
} finally {
    # Nettoyer le fichier temporaire
    if (Test-Path "$testFile.temp") {
        Remove-Item "$testFile.temp" -Force
    }
}

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "✅ SERVICES ACTIFS" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "📊 Dashboard web: http://localhost:8050" -ForegroundColor Cyan
Write-Host "📈 Les métriques sont maintenant visibles dans votre navigateur" -ForegroundColor Green
Write-Host "`n💡 Le dashboard continue de fonctionner en arrière-plan" -ForegroundColor Yellow
Write-Host "🛑 Pour l'arrêter, utilisez: Stop-Job $($dashboardJob.Id); Remove-Job $($dashboardJob.Id)" -ForegroundColor Yellow

Write-Host "`n🌐 Ouvrez maintenant votre navigateur sur:" -ForegroundColor Magenta
Write-Host "   http://localhost:8050" -ForegroundColor White -BackgroundColor Blue