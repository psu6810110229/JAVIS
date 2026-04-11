# start-jarvis-full.ps1 — Start JARVIS with all native Windows services
# This script starts: TTS worker, then the JARVIS backend

param(
    [switch]$SkipTTS = $false,
    [switch]$SkipOllamaCheck = $false,
    [bool]$RequireVoice = $true,
    [int]$TtsHealthTimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

Write-Host @"
    ___    ___  _________ ________  ________  ________ 
   |\  \  /  /||\___   ___\\   __  \|\   __  \|\   ____\
   \ \  \/  / /\|___ \  \_\ \  \|\  \ \  \|\  \ \  \___| 
    \ \    / /      \ \  \  \   __  \ \   _  _\ \_____  \
     /     \/        \ \  \  \  \ \  \ \  \\  \\|____|\  \
    /  /\   \         \ \__\  \__\ \__\ \__\\ _\ ____\_\  \
   /__/ /\ __\         \|__|\|__|\|__|\|__|\|__||\_________\
   |__|/ \|__|                                  \|_______|
"@ -ForegroundColor Cyan

Write-Host "`n[JARVIS] Starting native Windows mode...`n" -ForegroundColor Green

function Test-TtsHealth {
    param(
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8870/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
            # Retry until deadline
        }
        Start-Sleep -Seconds 1
    }

    return $false
}

function Show-RecentTtsLogs {
    param(
        [string]$StdOutPath,
        [string]$StdErrPath,
        [int]$TailLines = 30
    )

    if (Test-Path $StdOutPath) {
        Write-Host "[TTS] Recent worker stdout:" -ForegroundColor DarkYellow
        Get-Content -Path $StdOutPath -Tail $TailLines | ForEach-Object { Write-Host "  $_" }
    }

    if (Test-Path $StdErrPath) {
        Write-Host "[TTS] Recent worker stderr:" -ForegroundColor DarkYellow
        Get-Content -Path $StdErrPath -Tail $TailLines | ForEach-Object { Write-Host "  $_" }
    }
}

# Check Ollama
if (-not $SkipOllamaCheck) {
    Write-Host "[CHECK] Verifying Ollama is running..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop | Out-Null
        Write-Host "[CHECK] Ollama is running!" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Ollama is not running on port 11434!" -ForegroundColor Red
        Write-Host "[ERROR] Start Ollama first with: ollama serve" -ForegroundColor Red
        exit 1
    }
}

# Start TTS Worker
if (-not $SkipTTS) {
    Write-Host "[TTS] Starting native Windows TTS worker..." -ForegroundColor Yellow
    
    # Check if port is already listening
    $portListener = Get-NetTCPConnection -LocalPort 8870 -State Listen -ErrorAction SilentlyContinue
    if ($portListener) {
        Write-Host "[TTS] Port 8870 already in use - TTS worker may already be running" -ForegroundColor Yellow
        $ttsHealthy = Test-TtsHealth -TimeoutSeconds $TtsHealthTimeoutSeconds
        if ($ttsHealthy) {
            Write-Host "[TTS] Existing TTS worker is healthy." -ForegroundColor Green
        } elseif ($RequireVoice) {
            Write-Host "[ERROR] Port 8870 is occupied but TTS health check failed." -ForegroundColor Red
            exit 1
        } else {
            Write-Host "[WARN] Port 8870 is occupied but TTS health check failed. Continuing because RequireVoice=false." -ForegroundColor Yellow
        }
    } else {
        $ttsScript = Join-Path $projectRoot "scripts\start-tts-native.ps1"
        if (Test-Path $ttsScript) {
            $ttsLogDir = Join-Path $projectRoot "scripts\logs"
            New-Item -ItemType Directory -Path $ttsLogDir -Force | Out-Null
            $ttsStdOutLog = Join-Path $ttsLogDir "tts-worker.stdout.log"
            $ttsStdErrLog = Join-Path $ttsLogDir "tts-worker.stderr.log"
            Remove-Item -Path $ttsStdOutLog, $ttsStdErrLog -ErrorAction SilentlyContinue

            $ttsProcess = Start-Process powershell.exe -ArgumentList "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$ttsScript`"" -WindowStyle Normal -PassThru -RedirectStandardOutput $ttsStdOutLog -RedirectStandardError $ttsStdErrLog
            Write-Host "[TTS] TTS worker starting in new window..." -ForegroundColor Green

            $modelCacheDir = Join-Path $projectRoot "scripts\.cache\kokoro"
            $modelFile = Join-Path $modelCacheDir "kokoro-v1.0.onnx"
            $voicesFile = Join-Path $modelCacheDir "voices-v1.0.bin"
            $firstRunDownload = (-not (Test-Path $modelFile)) -or (-not (Test-Path $voicesFile))
            $effectiveTimeout = $TtsHealthTimeoutSeconds
            if ($firstRunDownload -and $effectiveTimeout -lt 120) {
                $effectiveTimeout = 120
                Write-Host "[TTS] First run detected (model assets missing). Extending health timeout to $effectiveTimeout seconds." -ForegroundColor Yellow
            }

            Write-Host "[TTS] Waiting for TTS health endpoint..." -ForegroundColor Yellow
            $ttsHealthy = Test-TtsHealth -TimeoutSeconds $effectiveTimeout
            if ($ttsHealthy) {
                Write-Host "[TTS] TTS health check passed (http://127.0.0.1:8870/health)." -ForegroundColor Green
            } elseif ($RequireVoice) {
                if ($ttsProcess.HasExited) {
                    Write-Host "[ERROR] TTS worker exited early with code $($ttsProcess.ExitCode)." -ForegroundColor Red
                } else {
                    Write-Host "[ERROR] TTS worker did not become healthy within $effectiveTimeout seconds." -ForegroundColor Red
                }
                Show-RecentTtsLogs -StdOutPath $ttsStdOutLog -StdErrPath $ttsStdErrLog
                Write-Host "[ERROR] TTS health check failed and voice is required." -ForegroundColor Red
                exit 1
            } else {
                Write-Host "[WARN] TTS health check failed. Continuing because RequireVoice=false." -ForegroundColor Yellow
                Show-RecentTtsLogs -StdOutPath $ttsStdOutLog -StdErrPath $ttsStdErrLog
            }
        } else {
            if ($RequireVoice) {
                Write-Host "[ERROR] TTS startup script not found at $ttsScript" -ForegroundColor Red
                exit 1
            }
            Write-Host "[WARN] TTS startup script not found at $ttsScript" -ForegroundColor Yellow
            Write-Host "[WARN] Audio will not work without TTS worker!" -ForegroundColor Yellow
        }
    }
} elseif ($RequireVoice) {
    Write-Host "[TTS] SkipTTS requested. Verifying an existing TTS worker is healthy..." -ForegroundColor Yellow
    $ttsHealthy = Test-TtsHealth -TimeoutSeconds $TtsHealthTimeoutSeconds
    if (-not $ttsHealthy) {
        Write-Host "[ERROR] Voice is required but TTS health endpoint is unavailable." -ForegroundColor Red
        exit 1
    }
    Write-Host "[TTS] Existing TTS worker is healthy." -ForegroundColor Green
}

# Start JARVIS Backend
Write-Host "`n[BACKEND] Starting JARVIS backend..." -ForegroundColor Green
$backendDir = Join-Path $projectRoot "jarvis-backend"

if (-not (Test-Path $backendDir)) {
    Write-Error "Backend directory not found at: $backendDir"
    exit 1
}

# Check for venv
$venvPython = Join-Path $backendDir "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found! Run: python -m venv venv"
    exit 1
}

# Check for dependencies
Write-Host "[BACKEND] Checking dependencies..." -ForegroundColor Yellow
& $venvPython -c "import fastapi, httpx, uvicorn, onnxruntime, tzdata" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BACKEND] Installing dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install -r (Join-Path $backendDir "requirements.txt")
}

Write-Host "[BACKEND] Starting server on http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "[BACKEND] Press Ctrl+C to stop`n" -ForegroundColor Cyan

Set-Location $backendDir
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

Write-Host "`n[JARVIS] Backend stopped." -ForegroundColor Yellow
