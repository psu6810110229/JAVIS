param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path $PSScriptRoot -Parent
$backendDir = Join-Path $projectRoot 'jarvis-backend'
$venvPythonRoot = Join-Path $projectRoot '.venv\Scripts\python.exe'
$venvPythonBackend = Join-Path $backendDir 'venv\Scripts\python.exe'

if (Test-Path $venvPythonRoot) {
    $pythonExe = $venvPythonRoot
} elseif (Test-Path $venvPythonBackend) {
    $pythonExe = $venvPythonBackend
} else {
    throw "Python venv not found. Expected '$venvPythonRoot' or '$venvPythonBackend'."
}

Write-Host "[CLEAN] Stopping existing backend listeners on port $Port..." -ForegroundColor Yellow
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    try {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction Stop
        Write-Host "[CLEAN] Stopped PID $($listener.OwningProcess)" -ForegroundColor Green
    } catch {
        Write-Host "[CLEAN] Could not stop PID $($listener.OwningProcess): $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

Write-Host "[CLEAN] Stopping stale uvicorn python processes..." -ForegroundColor Yellow
$stalePython = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'uvicorn' -and
        $_.CommandLine -match 'app\.main:app'
    }

foreach ($proc in $stalePython) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        Write-Host "[CLEAN] Stopped stale python PID $($proc.ProcessId)" -ForegroundColor Green
    } catch {
        Write-Host "[CLEAN] Could not stop stale python PID $($proc.ProcessId): $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

Start-Sleep -Milliseconds 300

Write-Host "[START] Launching fresh backend on http://127.0.0.1:$Port" -ForegroundColor Cyan
Set-Location $backendDir
& $pythonExe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
