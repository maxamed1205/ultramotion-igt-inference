# ═══════════════════════════════════════════════════════════════════════
#  Script de Lancement : Test GPU Transfer + Dashboard
# ═══════════════════════════════════════════════════════════════════════
#
#  Ce script lance simultanément :
#  1. Le test de validation GPU (test_validate_step1_step2.py)
#  2. Le dashboard de visualisation (dashboard_gpu_transfer.py)
#
#  Le dashboard sera accessible sur http://localhost:8051
#
# ═══════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

# Couleurs
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Error-Custom { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning-Custom { Write-Host $args -ForegroundColor Yellow }

Write-Host ""
Write-Info "════════════════════════════════════════════════════════════════"
Write-Info "  GPU Transfer Test + Dashboard Launcher"
Write-Info "════════════════════════════════════════════════════════════════"
Write-Host ""

# ───────────────────────────────────────────────
#  1. Vérifier l'environnement Python
# ───────────────────────────────────────────────
Write-Info "[1/4] Vérification de l'environnement Python..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error-Custom "❌ Python n'est pas installé ou n'est pas dans le PATH"
    exit 1
}

$pythonVersion = python --version 2>&1
Write-Success "  ✅ Python trouvé : $pythonVersion"

# ───────────────────────────────────────────────
#  2. Nettoyer les anciens logs
# ───────────────────────────────────────────────
Write-Info "`n[2/4] Nettoyage des anciens logs..."

$logsDir = "logs"
if (Test-Path $logsDir) {
    $logFiles = Get-ChildItem -Path $logsDir -Filter "*.log" -ErrorAction SilentlyContinue
    
    if ($logFiles) {
        foreach ($file in $logFiles) {
            try {
                Remove-Item $file.FullName -Force -ErrorAction Stop
                Write-Success "  ✅ Supprimé : $($file.Name)"
            } catch {
                Write-Warning-Custom "  ⚠️  Impossible de supprimer $($file.Name) (fichier verrouillé)"
            }
        }
    } else {
        Write-Info "  ℹ️  Aucun fichier log à supprimer"
    }
} else {
    Write-Info "  ℹ️  Dossier logs/ n'existe pas encore"
}

# ───────────────────────────────────────────────
#  3. Démarrer le Dashboard en arrière-plan
# ───────────────────────────────────────────────
Write-Info "`n[3/4] Démarrage du dashboard GPU Transfer..."

$dashboardScript = "src\service\dashboard_gpu_transfer.py"

if (-not (Test-Path $dashboardScript)) {
    Write-Error-Custom "❌ Script dashboard introuvable : $dashboardScript"
    exit 1
}

Write-Info "  🚀 Lancement du serveur FastAPI sur http://localhost:8051"

$dashboardProcess = Start-Process -FilePath "python" `
    -ArgumentList $dashboardScript `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput "logs\dashboard_stdout.log" `
    -RedirectStandardError "logs\dashboard_stderr.log"

# Attendre que le serveur démarre
Start-Sleep -Seconds 3

if ($dashboardProcess.HasExited) {
    Write-Error-Custom "❌ Le dashboard a crashé au démarrage"
    Write-Error-Custom "   Vérifiez logs\dashboard_stderr.log pour les détails"
    exit 1
}

Write-Success "  ✅ Dashboard démarré (PID: $($dashboardProcess.Id))"
Write-Info "  🌐 URL: http://localhost:8051"

# ───────────────────────────────────────────────
#  4. Exécuter le test GPU
# ───────────────────────────────────────────────
Write-Info "`n[4/4] Exécution du test de validation GPU..."

$testScript = "tests\tests_gateway\test_validate_step1_step2.py"

if (-not (Test-Path $testScript)) {
    Write-Error-Custom "❌ Script de test introuvable : $testScript"
    
    # Arrêter le dashboard
    Write-Info "`n🛑 Arrêt du dashboard..."
    Stop-Process -Id $dashboardProcess.Id -Force
    exit 1
}

Write-Info "  Lancement du test (10 frames)..."
Write-Host ""

# Executer le test (bloquant)
$testExitCode = 0
try {
    python $testScript
    $testExitCode = $LASTEXITCODE
} catch {
    $testExitCode = 1
}

# ───────────────────────────────────────────────
#  5. Résultats
# ───────────────────────────────────────────────
Write-Host ""
Write-Info "════════════════════════════════════════════════════════════════"

if ($testExitCode -eq 0) {
    Write-Success "✅ TEST RÉUSSI"
    Write-Host ""
    Write-Info "📊 Le dashboard est toujours actif sur http://localhost:8051"
    Write-Info "   Vous pouvez visualiser les métriques en temps réel"
    Write-Host ""
    Write-Warning-Custom "⚠️  Appuyez sur Ctrl+C pour arrêter le dashboard"
    
    # Garder le dashboard actif
    Write-Host ""
    Write-Info "En attente... (le dashboard reste actif)"
    
    try {
        # Attendre indefiniment (jusqu a Ctrl+C)
        while ($true) {
            Start-Sleep -Seconds 1
            
            # Verifier si le dashboard est toujours en cours
            if ($dashboardProcess.HasExited) {
                Write-Warning-Custom "`nLe dashboard s'est arrete"
                break
            }
        }
    } catch {
        # Ctrl+C intercepte
        Write-Info "`nArret demande par l'utilisateur"
    }
    
} else {
    Write-Error-Custom "TEST ECHOUE (code: $testExitCode)"
    Write-Error-Custom "   Verifiez les logs pour plus de details"
}

# ---------------------------------------------------
#  6. Nettoyage : Arreter le dashboard
# ---------------------------------------------------
Write-Host ""
Write-Info "Arret du dashboard..."

if (-not $dashboardProcess.HasExited) {
    Stop-Process -Id $dashboardProcess.Id -Force
    Write-Success "  ✅ Dashboard arrêté"
} else {
    Write-Info "  ℹ️  Dashboard déjà arrêté"
}

Write-Host ""
Write-Info "════════════════════════════════════════════════════════════════"
Write-Host ""

exit $testExitCode
