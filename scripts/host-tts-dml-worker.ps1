param(
    [int]$Port = 8870
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[TTS-DML] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[TTS-DML] $Message" -ForegroundColor Yellow
}

$pythonScript = Join-Path (Split-Path -Parent $PSCommandPath) "host_tts_dml_worker.py"
if (-not (Test-Path $pythonScript)) {
    throw "host_tts_dml_worker.py not found at $pythonScript"
}

$pythonCmd = "python"
try {
    $null = & $pythonCmd --version
} catch {
    Write-Warn "Python was not found on PATH. Use your venv python explicitly."
    throw
}

Write-Info "Starting host TTS worker on port $Port"
& $pythonCmd $pythonScript --port $Port
