@echo off
REM ========================================
REM  PROJECT AI - One-Click Startup
REM ========================================

cd /d "%~dp0"
echo.
echo ✅ Starting PROJECT AI...
echo.

REM Check if virtual environment exists
if not exist "backend\.venv" (
    echo 📦 Creating virtual environment...
    cd backend
    python -m venv .venv
    cd ..
)

REM Activate virtual environment
call backend\.venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing dependencies...
cd backend
pip install -q -r requirements.txt

REM Run Flask app
echo.
echo 🚀 Launching Flask application...
echo 📍 Access the app at: http://localhost:5000
echo.
python app.py

pause
