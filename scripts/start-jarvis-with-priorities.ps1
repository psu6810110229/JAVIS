param(
    [switch]$SkipBuild,
    [switch]$StartHostOptimizer,
    [int]$HostOptimizerPort = 8765,
    [switch]$StartHostTtsWorker,
    [int]$HostTtsPort = 8870
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Set-ContainerNice {
    param(
        [string]$ContainerName,
        [int]$NiceValue
    )

    try {
        docker exec $ContainerName sh -lc "renice -n $NiceValue -p 1" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Set Linux nice=$NiceValue for PID 1 in container '$ContainerName'."
        } else {
            Write-Warn "Could not set Linux nice=$NiceValue for '$ContainerName' (exit code $LASTEXITCODE)."
        }
    } catch {
        Write-Warn "Unable to set Linux nice for container '$ContainerName': $($_.Exception.Message)"
    }
}

function Set-ContainerHostPriority {
    param(
        [string]$ContainerName,
        [string]$PriorityClass,
        [int]$MaxAttempts = 12,
        [int]$RetryDelayMs = 500
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt += 1) {
        try {
            $pidText = docker inspect -f "{{.State.Pid}}" $ContainerName
            $pid = 0
            if (-not [int]::TryParse($pidText, [ref]$pid) -or $pid -le 0) {
                throw "Container PID unavailable."
            }

            $process = Get-Process -Id $pid -ErrorAction Stop
            $process.PriorityClass = $PriorityClass
            Write-Info "Set host process priority '$PriorityClass' for '$ContainerName' (PID $pid)."
            return
        } catch {
            if ($attempt -eq $MaxAttempts) {
                Write-Warn "Host priority for '$ContainerName' could not be applied after $MaxAttempts attempts. This can happen under Docker Desktop/WSL2 isolation."
                return
            }
            Start-Sleep -Milliseconds $RetryDelayMs
        }
    }
}

$scriptRoot = Split-Path -Parent $PSCommandPath
$projectRoot = Resolve-Path (Join-Path $scriptRoot "..")
Set-Location $projectRoot

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Warn "Run this script as Administrator for best-effort host priority changes."
}

if ($SkipBuild) {
    Write-Info "Starting containers without rebuild..."
    docker compose up -d
} else {
    Write-Info "Starting containers with rebuild..."
    docker compose up --build -d
}

Set-ContainerNice -ContainerName "ollama" -NiceValue -10
Set-ContainerNice -ContainerName "kokoro" -NiceValue -5

if ($isAdmin) {
    Set-ContainerHostPriority -ContainerName "ollama" -PriorityClass "High"
    Set-ContainerHostPriority -ContainerName "kokoro" -PriorityClass "AboveNormal"
}

if ($StartHostOptimizer) {
    $optimizerScript = Join-Path $projectRoot "scripts\host-optimizer-listener.ps1"
    if (Test-Path $optimizerScript) {
        Start-Process -FilePath "powershell.exe" -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $optimizerScript,
            "-Port",
            $HostOptimizerPort
        ) | Out-Null
        Write-Info "Started host optimizer listener on port $HostOptimizerPort."
        Write-Info "Set JARVIS_HOST_OPTIMIZER_URL=http://host.docker.internal:$HostOptimizerPort/optimize in backend env."
    } else {
        Write-Warn "Host optimizer listener script not found at '$optimizerScript'."
    }
}

if ($StartHostTtsWorker) {
    $ttsScript = Join-Path $projectRoot "scripts\host-tts-dml-worker.ps1"
    if (Test-Path $ttsScript) {
        Start-Process -FilePath "powershell.exe" -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $ttsScript,
            "-Port",
            $HostTtsPort
        ) | Out-Null
        Write-Info "Started host TTS worker on port $HostTtsPort."
        Write-Info "Set JARVIS_TTS_URL=http://host.docker.internal:$HostTtsPort/v1/audio/speech in backend env."
    } else {
        Write-Warn "Host TTS worker launcher not found at '$ttsScript'."
    }
}

Write-Info "Priority tuning step complete."
