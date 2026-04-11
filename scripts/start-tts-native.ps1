# start-tts-native.ps1 — Start the Windows native Kokoro TTS worker
# Run this before starting JARVIS to enable audio output

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$workerPath = Join-Path $projectRoot "scripts\host_tts_dml_worker.py"
$rootVenvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$backendVenvPython = Join-Path $projectRoot "jarvis-backend\venv\Scripts\python.exe"
$activeVenvPython = $null
if ($env:VIRTUAL_ENV) {
    $candidateActive = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path $candidateActive) {
        $activeVenvPython = $candidateActive
    }
}

Write-Host "[JARVIS TTS] Starting native Windows TTS worker on port 8870..." -ForegroundColor Cyan

# Select Python interpreter (prefer currently active venv, then root venv, backend venv, then PATH)
$pythonCandidates = @()
if ($activeVenvPython) {
    $pythonCandidates += [PSCustomObject]@{ Label = "active venv"; Path = $activeVenvPython }
}
if ((Test-Path $rootVenvPython) -and ($pythonCandidates.Path -notcontains $rootVenvPython)) {
    $pythonCandidates += [PSCustomObject]@{ Label = "root venv"; Path = $rootVenvPython }
}
if (Test-Path $backendVenvPython) {
    $pythonCandidates += [PSCustomObject]@{ Label = "backend venv"; Path = $backendVenvPython }
}
$pathPython = Get-Command python -ErrorAction SilentlyContinue
if ($pathPython -and ($pythonCandidates.Path -notcontains $pathPython.Source)) {
    $pythonCandidates += [PSCustomObject]@{ Label = "PATH python"; Path = $pathPython.Source }
}

if ($pythonCandidates.Count -eq 0) {
    Write-Error "No Python interpreter found. Activate a venv or install Python."
    exit 1
}

$pythonExe = $null
$failedCandidates = @()
foreach ($candidate in $pythonCandidates) {
    & $candidate.Path -c "import kokoro_onnx, onnxruntime, soundfile" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = $candidate.Path
        Write-Host "[JARVIS TTS] Using $($candidate.Label) python: $pythonExe" -ForegroundColor Green
        break
    }
    $failedCandidates += "$($candidate.Label): $($candidate.Path)"
}

if (-not $pythonExe) {
    Write-Error "No Python interpreter with required TTS dependencies found. Tried:`n - $($failedCandidates -join "`n - ")"
    exit 1
}

# Check if worker file exists
if (-not (Test-Path $workerPath)) {
    Write-Error "TTS worker not found at: $workerPath"
    exit 1
}

# Check if port is already listening
$portListener = Get-NetTCPConnection -LocalPort 8870 -State Listen -ErrorAction SilentlyContinue
if ($portListener) {
    Write-Host "[JARVIS TTS] Port 8870 is already in use. TTS worker may already be running." -ForegroundColor Yellow
    exit 0
}

# Start the TTS worker
Write-Host "[JARVIS TTS] Starting worker..." -ForegroundColor Green
& $pythonExe $workerPath

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "[JARVIS TTS] Worker exited with code $LASTEXITCODE" -ForegroundColor Red
}
