# EdgeOps AI Platform - Automated Multi-Service Launcher (Windows PowerShell)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   ⚡ EdgeOps AI Platform: Launching Multi-Service Stack   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Backend API
Write-Host "[1/3] Starting EdgeOps FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Green
$backendJob = Start-Process python -ArgumentList "-m uvicorn api:app --host 127.0.0.1 --port 8000 --reload" -PassThru -NoNewWindow

Start-Sleep -Seconds 3

# 2. Start Frontend Vite Dev Server
Write-Host "[2/3] Starting EdgeOps React Dashboard on http://localhost:5173..." -ForegroundColor Green
$frontendJob = Start-Process npm -ArgumentList "run dev" -WorkingDirectory "edgeops-dashboard" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🚀 Stack is running!" -ForegroundColor Green
Write-Host " - FastAPI Docs:     http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host " - React Dashboard:  http://localhost:5173" -ForegroundColor Yellow
Write-Host " - Embedded Grafana: http://localhost:3000 (if running docker compose)" -ForegroundColor Yellow
Write-Host "Press Ctrl+C or kill background jobs to stop." -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Cyan
