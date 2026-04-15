param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

$wrapperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Split-Path -Parent $wrapperDir
$projectRoot = Split-Path -Parent $backendDir
$rootScript = Join-Path $projectRoot 'scripts\start-backend-clean.ps1'

if (-not (Test-Path $rootScript)) {
    throw "Root clean-start script not found at '$rootScript'."
}

& $rootScript -Port $Port
