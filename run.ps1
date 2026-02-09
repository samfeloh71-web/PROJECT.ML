# ========================================
# PROJECT AI - One-Click Startup (PowerShell)
# ========================================

Write-Host "✅ Starting PROJECT AI..." -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Check and create virtual environment if needed
$venvPath = "$projectRoot\backend\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    Set-Location backend
    python -m venv .venv
    Set-Location ..
}

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
Set-Location backend
pip install -q -r requirements.txt

# Run Flask app
Write-Host ""
Write-Host "🚀 Launching Flask application..." -ForegroundColor Green
Write-Host "📍 Access the app at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "⚡ Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host ""

python app.py

Set-Location ..
