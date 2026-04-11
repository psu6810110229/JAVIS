# Windows-Native Tauri Runbook

This runbook is the canonical path for running Jarvis on Windows native (no Docker) with Tauri desktop as the primary client.

## Scope
- Host OS: Windows
- Backend: FastAPI (native Python venv)
- Frontend: Tauri desktop (Vite + Tauri v2)
- Ollama: native service on localhost
- TTS: native worker on localhost

## 1. Preconditions
1. Install Ollama and make sure it is running.
2. Create backend venv and install backend dependencies:
```powershell
Set-Location jarvis-backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```
3. Install frontend dependencies:
```powershell
Set-Location ..\jarvis-frontend
npm install
```
4. Ensure root .env is configured for native localhost endpoints:
- OLLAMA_BASE_URL=http://localhost:11434
- JARVIS_TTS_URL=http://localhost:8870/v1/audio/speech
- JARVIS_TTS_FALLBACK_URL=http://localhost:8880/v1/audio/speech
- JARVIS_HOST_OPTIMIZER_URL=http://localhost:8765/optimize

## 2. Startup Order (Tauri First)
1. Start backend with native guardrails and required voice:
```powershell
Set-Location D:\JAVIS\source_code\project_JAVIS
.\start-jarvis-full.ps1 -RequireVoice $true
```
2. In another terminal, start Tauri desktop:
```powershell
Set-Location D:\JAVIS\source_code\project_JAVIS\jarvis-frontend
npm run tauri dev
```

## 3. Health Checks
1. Ollama tags endpoint:
```powershell
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
```
2. Backend health endpoint:
```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```
3. Backend status endpoint:
```powershell
Invoke-WebRequest http://127.0.0.1:8000/v1/system/status -UseBasicParsing
```
4. Native TTS health endpoint:
```powershell
Invoke-WebRequest http://127.0.0.1:8870/health -UseBasicParsing
```

## 4. Command Acceptance Set (Tauri UI)
Run these from the Tauri chat box:
1. open spotify
2. pause spotify
3. play spotify
4. open notepad
5. what is the current date and time

Expected behavior:
- Backend should not produce tool-unavailable fallbacks for valid mapped tools.
- TTS failures should appear as voice warnings and text response should still complete.

## 5. Automated Smoke Test Script
Use the native smoke-test helper from project root:
```powershell
.\scripts\smoke-test-native.ps1
```
Run optional TTS failure-recovery check:
```powershell
.\scripts\smoke-test-native.ps1 -IncludeTtsFailureRecovery
```

## 6. Troubleshooting Fast Path
1. If backend fails to start, verify venv packages and Ollama reachability.
2. If voice fails, run:
```powershell
.\scripts\start-tts-native.ps1
```
3. If Spotify tools fail, re-check SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and redirect URI in .env.
4. If Tauri cannot connect, verify backend is listening on 127.0.0.1:8000 and check VITE_JARVIS_API_URL / VITE_JARVIS_WS_URL overrides.
