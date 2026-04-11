@echo off
setlocal

:: JARVIS Windows Startup Script
echo [JARVIS] Starting Native Windows Backend...

:: Go to the backend directory
cd /d "%~dp0jarvis-backend"

:: Check for venv
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in jarvis-backend\venv.
    echo Please create it first: python -m venv venv
    pause
    exit /b 1
)

:: Start Uvicorn
echo [JARVIS] Initializing Orchestrator...
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
