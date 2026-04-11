param(
    [string]$BackendBaseUrl = "http://127.0.0.1:8000",
    [switch]$IncludeTtsFailureRecovery = $false,
    [int]$TtsHealthTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[SMOKE] $Message" -ForegroundColor Cyan
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host "[SMOKE] $Message" -ForegroundColor Yellow
}

function Write-ErrMsg {
    param([string]$Message)
    Write-Host "[SMOKE] $Message" -ForegroundColor Red
}

function Test-HttpOk {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 8 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Wait-TtsHealth {
    param([int]$TimeoutSeconds = 30)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpOk -Url "http://127.0.0.1:8870/health") {
            return $true
        }
        Start-Sleep -Seconds 1
    }

    return $false
}

function Invoke-ChatStream {
    param(
        [string]$Message,
        [bool]$AutoSpeak = $true
    )

    $url = "$BackendBaseUrl/v1/chat"
    $body = @{
        message = $Message
        auto_speak = $AutoSpeak
    } | ConvertTo-Json -Compress

    $response = Invoke-WebRequest -Uri $url -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 120 -ErrorAction Stop
    $events = @()

    foreach ($line in ($response.Content -split "`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed) {
            continue
        }

        try {
            $events += ($trimmed | ConvertFrom-Json)
        } catch {
            # Ignore malformed lines and continue collecting others.
        }
    }

    return $events
}

function Get-FinalEvent {
    param([array]$Events)

    $final = $null
    foreach ($event in $Events) {
        if ($event.type -eq "final") {
            $final = $event
        }
    }
    return $final
}

function Test-ClaimAlignment {
    param(
        [string]$CommandName,
        [string]$FinalText,
        [object]$ToolOutcome
    )

    if (-not $ToolOutcome) {
        throw "Missing last_tool_outcome for '$CommandName'."
    }

    $status = [string]$ToolOutcome.status
    $verified = $ToolOutcome.verified

    $positiveClaim = $FinalText -match '(?i)\b(success|successfully|done|completed|opened|launched|playing|paused|transferred|killed|closed)\b'
    $cautiousClaim = $FinalText -match '(?i)\b(cannot|could not|unable|cannot confirm|not confirm|failed|error|did not|might|may not)\b'

    $failedStatuses = @('failed', 'error', 'timeout', 'not_found', 'crashed', 'blocked')
    $uncertainStatuses = @('partial', 'unverified', 'opened_no_window', 'no_windows')

    if ($failedStatuses -contains $status.ToLower()) {
        if ($positiveClaim) {
            throw "Contradictory claim: status='$status' but final text sounds successful."
        }
        return
    }

    if (($uncertainStatuses -contains $status.ToLower()) -or ($verified -eq $false)) {
        if ($positiveClaim -and -not $cautiousClaim) {
            throw "Overconfident claim: status='$status' verified='$verified' without caution wording."
        }
    }
}

function Test-ProcessRunningByName {
    param([string]$ProcessName)

    $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
    return $null -ne $proc
}

function Test-ChatCommand {
    param(
        [string]$Name,
        [string]$Message,
        [string]$PostCheck = ""
    )

    Write-Info "Running chat test: $Name"
    $events = Invoke-ChatStream -Message $Message -AutoSpeak $true

    $hasFinal = $false
    $hasToolUnavailable = $false
    $finalEvent = $null
    foreach ($event in $events) {
        if ($event.type -eq "final") {
            $hasFinal = $true
            $finalEvent = $event
            if ($event.text -and ($event.text -match "Tool '.+' is unavailable")) {
                $hasToolUnavailable = $true
            }
        }

        if ($event.type -eq "error" -and $event.message -and ($event.message -match "Tool '.+' is unavailable")) {
            $hasToolUnavailable = $true
        }
    }

    if (-not $hasFinal) {
        throw "No final event returned for '$Name'."
    }

    if ($hasToolUnavailable) {
        throw "Tool unavailable regression detected for '$Name'."
    }

    if (-not $finalEvent) {
        throw "No final event payload found for '$Name'."
    }

    $finalText = if ($finalEvent.text) { [string]$finalEvent.text } else { "" }
    Test-ClaimAlignment -CommandName $Name -FinalText $finalText -ToolOutcome $finalEvent.last_tool_outcome

    if ($PostCheck -eq "notepad-open") {
        if (-not (Test-ProcessRunningByName -ProcessName "notepad")) {
            throw "Postcondition failed: Notepad process was not found after open request."
        }
    }

    if ($PostCheck -eq "spotify-open") {
        $spotifyRunning = (Test-ProcessRunningByName -ProcessName "Spotify") -or (Test-ProcessRunningByName -ProcessName "SpotifyLauncher")
        $status = [string]$finalEvent.last_tool_outcome.status
        if ($status -eq "opened" -and -not $spotifyRunning) {
            throw "Postcondition failed: tool reported opened but Spotify process was not found."
        }
    }

    Write-Info "PASS: $Name"
}

$failures = New-Object System.Collections.Generic.List[string]

Write-Info "Checking backend base URL: $BackendBaseUrl"

try {
    if (-not (Test-HttpOk -Url "$BackendBaseUrl/health")) {
        throw "Backend /health is not reachable."
    }

    $statusResponse = Invoke-WebRequest -Uri "$BackendBaseUrl/v1/system/status" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
    $status = $statusResponse.Content | ConvertFrom-Json
    Write-Info "Status: mode=$($status.active_mode) model=$($status.active_model) tools_registered=$($status.tools_registered) ollama_ready=$($status.system_load.ollama_ready)"
    Write-Info "TTS: mode=$($status.tts.mode) provider=$($status.tts.provider) status=$($status.tts.status)"
} catch {
    Write-ErrMsg $_.Exception.Message
    exit 1
}

$tests = @(
    @{ Name = "Open Spotify"; Message = "open spotify"; PostCheck = "spotify-open" },
    @{ Name = "Pause Spotify"; Message = "pause spotify"; PostCheck = "" },
    @{ Name = "Play Spotify"; Message = "play spotify"; PostCheck = "" },
    @{ Name = "Open Notepad"; Message = "open notepad"; PostCheck = "notepad-open" },
    @{ Name = "Date Time"; Message = "what is the current date and time"; PostCheck = "" }
)

foreach ($test in $tests) {
    try {
        Test-ChatCommand -Name $test.Name -Message $test.Message -PostCheck $test.PostCheck
    } catch {
        $message = "FAIL: $($test.Name) -> $($_.Exception.Message)"
        $failures.Add($message)
        Write-ErrMsg $message
    }
}

if ($IncludeTtsFailureRecovery) {
    Write-Info "Running TTS failure-recovery scenario"

    try {
        $listener = Get-NetTCPConnection -LocalPort 8870 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $listener) {
            throw "No listening process found on port 8870."
        }

        $ttsPid = [int]$listener.OwningProcess
        Write-Info "Stopping TTS process PID $ttsPid"
        Stop-Process -Id $ttsPid -Force -ErrorAction Stop

        $eventsDown = Invoke-ChatStream -Message "say this is a tts failure recovery check" -AutoSpeak $true
        $hasFinalDown = $false
        $hasTtsError = $false
        foreach ($event in $eventsDown) {
            if ($event.type -eq "final") {
                $hasFinalDown = $true
            }
            if ($event.type -eq "error" -and $event.stage -eq "tts") {
                $hasTtsError = $true
            }
        }

        if (-not $hasFinalDown) {
            throw "No final event after forcing TTS down."
        }
        if (-not $hasTtsError) {
            throw "Expected a TTS stage error when TTS worker is down."
        }

        Write-Info "Restarting native TTS worker"
        $root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
        $ttsScript = Join-Path $root "scripts\start-tts-native.ps1"
        Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$ttsScript`"" -WindowStyle Normal | Out-Null

        if (-not (Wait-TtsHealth -TimeoutSeconds $TtsHealthTimeoutSeconds)) {
            throw "TTS health did not recover in time."
        }

        $eventsUp = Invoke-ChatStream -Message "say this is a tts recovery validation" -AutoSpeak $true
        $hasFinalUp = $false
        foreach ($event in $eventsUp) {
            if ($event.type -eq "final") {
                $hasFinalUp = $true
                break
            }
        }

        if (-not $hasFinalUp) {
            throw "No final event after TTS restart."
        }

        Write-Info "PASS: TTS failure-recovery scenario"
    } catch {
        $message = "FAIL: TTS failure-recovery -> $($_.Exception.Message)"
        $failures.Add($message)
        Write-ErrMsg $message
    }
}

if ($failures.Count -gt 0) {
    Write-ErrMsg "Smoke tests completed with failures:"
    foreach ($failure in $failures) {
        Write-ErrMsg " - $failure"
    }
    exit 1
}

Write-Info "All smoke tests passed."
exit 0
