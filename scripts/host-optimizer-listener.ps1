param(
    [int]$Port = 8765,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[HOST-OPT] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[HOST-OPT] $Message" -ForegroundColor Yellow
}

function Invoke-StandbyFlush {
    param([string]$ScriptRoot)

    $emptyStandby = Join-Path $ScriptRoot "EmptyStandbyList.exe"
    if (Test-Path $emptyStandby) {
        & $emptyStandby workingsets | Out-Null
        & $emptyStandby standbylist | Out-Null
        return "standbylist-flushed"
    }

    Write-Warn "EmptyStandbyList.exe not found in scripts folder; standby flush skipped."
    return "standbylist-tool-missing"
}

$scriptRoot = Split-Path -Parent $PSCommandPath
$prefix = "http://127.0.0.1:$Port/optimize/"
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add($prefix)
$listener.Start()

Write-Info "Listening on $prefix"

try {
    do {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response

        if ($request.HttpMethod -ne "POST") {
            $response.StatusCode = 405
            $bytes = [System.Text.Encoding]::UTF8.GetBytes('{"ok":false,"error":"method-not-allowed"}')
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
            $response.Close()
            continue
        }

        $result = Invoke-StandbyFlush -ScriptRoot $scriptRoot
        $payload = @{
            ok = $true
            result = $result
        } | ConvertTo-Json -Compress
        $response.StatusCode = 200
        $response.ContentType = "application/json"
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
        $response.OutputStream.Write($bytes, 0, $bytes.Length)
        $response.Close()

        Write-Info "Handled optimization callback with result '$result'."
    } while (-not $Once)
} finally {
    $listener.Stop()
    $listener.Close()
}
