@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ===================================================
echo     Jarvis Native Windows Backend Initialization
echo ===================================================

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

set ROOT_DIR=%SCRIPT_DIR%..
for %%I in ("%ROOT_DIR%") do set ROOT_DIR=%%~fI
set VENV_DIR=%ROOT_DIR%\.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

echo [1/4] Checking Python availability...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.11+.
    popd
    endlocal
    exit /b 1
)

echo [2/4] Ensuring root virtual environment exists...
if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment at "%VENV_DIR%"...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment at "%VENV_DIR%".
        popd
        endlocal
        exit /b 1
    )
)

echo [3/4] Activating environment and installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    popd
    endlocal
    exit /b 1
)

echo [4/4] Starting Uvicorn API Server...
echo Make sure Ollama and Kokoro TTS are accessible at localhost.
echo ===================================================

where ffmpeg >nul 2>&1
if errorlevel 1 (
    set "FFMPEG_BIN="
    for /f "delims=" %%F in ('powershell -NoProfile -Command "Get-ChildItem \"$env:LOCALAPPDATA\Microsoft\WinGet\Packages\" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName"') do set "FFMPEG_BIN=%%F"
    if defined FFMPEG_BIN (
        for %%D in ("!FFMPEG_BIN!") do set "FFMPEG_DIR=%%~dpD"
        set "PATH=!FFMPEG_DIR!;!PATH!"
        echo [INFO] Added ffmpeg to PATH from winget package: !FFMPEG_DIR!
    ) else (
        echo [WARN] ffmpeg was not found in PATH. Audio decode features may fail.
    )
)

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags -TimeoutSec 2 ^| Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel% neq 0 (
    echo [WARN] Ollama API is not reachable at http://localhost:11434.
    echo [WARN] Start Ollama service before using chat features.
)

:: Override runtime URLs for native mode
set OLLAMA_BASE_URL=http://localhost:11434
set JARVIS_TTS_FALLBACK_URL=http://localhost:8880/v1/audio/speech
set JARVIS_HOST_OPTIMIZER_URL=http://localhost:8765/optimize

:: Ensure backend package imports resolve from this folder
set PYTHONPATH=%CD%

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

popd
endlocal
