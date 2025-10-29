# Script de lancement du Dashboard Unifié
# Démarre le dashboard combinant GPU Transfer + Pipeline metrics

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Dashboard Unifié - Ultramotion IGT" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Python est installé
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python trouvé: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    exit 1
}

# Vérifier les dépendances
Write-Host ""
Write-Host "Vérification des dépendances..." -ForegroundColor Yellow

$dependencies = @("fastapi", "uvicorn", "torch")
$missingDeps = @()

foreach ($dep in $dependencies) {
    $result = python -c "import $dep" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missingDeps += $dep
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Host "✗ Dépendances manquantes: $($missingDeps -join ', ')" -ForegroundColor Red
    Write-Host "  Installation en cours..." -ForegroundColor Yellow
    pip install fastapi uvicorn
}

# Créer le répertoire logs si nécessaire
$logsDir = "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "✓ Répertoire logs créé" -ForegroundColor Green
}

# Lancer le dashboard
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Démarrage du Dashboard Unifié..." -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard accessible sur: http://localhost:8050" -ForegroundColor Green
Write-Host "Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Yellow
Write-Host ""

# Lancer le service
python -m service.dashboard_unified --port 8050 --host 0.0.0.0 --interval 1.0
