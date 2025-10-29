# =====================================================================
# Script: run_test_with_unified_dashboard.ps1
# Start unified dashboard + run pipeline test
# =====================================================================

# Avoid weird encoding in console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Test Pipeline + Unified Dashboard" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Clean old logs
Write-Host "Cleaning logs..." -ForegroundColor Yellow
if (Test-Path "logs/kpi.log") { Remove-Item "logs/kpi.log" -Force }
if (Test-Path "logs/pipeline.log") { Remove-Item "logs/pipeline.log" -Force }
Write-Host "Logs cleaned" -ForegroundColor Green
Write-Host ""

# Start dashboard in background
Write-Host "Starting dashboard..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m service.dashboard_unified --port 8050 --host 0.0.0.0"

# Wait a bit for dashboard to come up
Write-Host "Waiting for dashboard to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Probe dashboard (compatible with Windows PowerShell 5.x)
$dashOk = $false
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8050/api/health" -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { $dashOk = $true }
} catch { }
if ($dashOk) {
    Write-Host "Dashboard is up at http://localhost:8050" -ForegroundColor Green
} else {
    Write-Host "Dashboard not ready yet, continuing..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Dashboard URL: http://localhost:8050" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Run pipeline test
Write-Host "Starting pipeline test..." -ForegroundColor Yellow
Write-Host ""

# Use call operator & to avoid parsing issues
& python "tests/tests_gateway/test_gateway_dataset_mock.py"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Test Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard is still running at http://localhost:8050" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the dashboard when done." -ForegroundColor Yellow
Write-Host ""

# Keep dashboard running until user presses Ctrl+C
try {
    Write-Host "Dashboard is running in background. Waiting..." -ForegroundColor Green
    Write-Host "You can:" -ForegroundColor Yellow
    Write-Host "  - Open http://localhost:8050 in your browser" -ForegroundColor White
    Write-Host "  - Run more tests in another terminal" -ForegroundColor White
    Write-Host "  - Press Ctrl+C here to stop the dashboard" -ForegroundColor White
    Write-Host ""
    
    # Wait until user presses Ctrl+C or job fails
    while ($dashboardJob.State -eq "Running") {
        Start-Sleep -Seconds 2
        
        # Check if job is still running
        if ($dashboardJob.State -ne "Running") {
            Write-Host "Dashboard job ended unexpectedly" -ForegroundColor Red
            break
        }
    }
} catch {
    Write-Host "Dashboard interrupted by user" -ForegroundColor Yellow
}

# Stop dashboard job
Write-Host ""
Write-Host "Stopping dashboard..." -ForegroundColor Yellow
if ($null -ne $dashboardJob) {
    Stop-Job -Job $dashboardJob -ErrorAction SilentlyContinue
    Remove-Job -Job $dashboardJob -ErrorAction SilentlyContinue
    Write-Host "Dashboard stopped" -ForegroundColor Green
}

Write-Host ""

Write-Host "Test finished" -ForegroundColor Green
Write-Host "Appuyez sur Entrée pour fermer cette fenêtre..." -ForegroundColor Yellow
Read-Host
