# Script de demarrage du Dashboard Ultramotion IGT
# Usage: .\start_dashboard.ps1 [port]

param(
    [int]$Port = 8050,
    [string]$HostAddress = "0.0.0.0",
    [double]$Interval = 1.0
)

Write-Host "`nDemarrage du Dashboard Ultramotion IGT..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Verification des dependances
Write-Host "`nVerification des dependances..." -ForegroundColor Yellow

$dependencies = @("fastapi", "uvicorn", "websockets")
$missing = @()

foreach ($dep in $dependencies) {
    $installed = python -c "import $dep" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $dep
    }
}

if ($missing.Count -gt 0) {
    Write-Host "ATTENTION: Dependances manquantes detectees : $($missing -join ', ')" -ForegroundColor Red
    Write-Host "`nInstallation automatique..." -ForegroundColor Yellow
    
    pip install -r requirements-dashboard.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERREUR: Echec de l'installation des dependances" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "OK: Dependances installees avec succes" -ForegroundColor Green
} else {
    Write-Host "OK: Toutes les dependances sont installees" -ForegroundColor Green
}

# Verification que le port est disponible
$portInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "`nATTENTION: Le port $Port est deja utilise !" -ForegroundColor Red
    Write-Host "Processus : $($portInUse.OwningProcess)" -ForegroundColor Yellow
    
    $response = Read-Host "`nVoulez-vous utiliser un autre port ? (O/N)"
    if ($response -eq 'O' -or $response -eq 'o') {
        $Port = Read-Host "Nouveau port"
    } else {
        Write-Host "STOP: Arret du script" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nConfiguration :" -ForegroundColor Cyan
Write-Host "  - Host : $HostAddress" -ForegroundColor White
Write-Host "  - Port : $Port" -ForegroundColor White
Write-Host "  - Interval : ${Interval}s" -ForegroundColor White
Write-Host "  - URL : http://localhost:$Port" -ForegroundColor Green

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "Demarrage du serveur..." -ForegroundColor Yellow
Write-Host ""

# Ajouter le repertoire src au PYTHONPATH
$env:PYTHONPATH = "$PSScriptRoot\src"

# Demarrage du dashboard
python -m service.dashboard_service --port $Port --host $HostAddress --interval $Interval

# Si le script se termine
Write-Host "`nDashboard arrete" -ForegroundColor Yellow
