# JAVIS Recovery Plan (Functional + Stable)

Date: 2026-04-10
Scope: backend, frontend, Tauri desktop shell, local run workflow, repository hygiene.

## 1) Current Fact-Based Status

What is working now:
- Frontend build is passing (`npm run build` in `jarvis-frontend`).
- Tauri Rust check is passing (`cargo check` in `jarvis-frontend/src-tauri`).
- Backend imports now resolve in `.venv` (`httpx`, `psutil`, `speech_recognition`, `pydub`).
- Backend app import works (`from app.main import app`).

What is risky or messy now:
- Uncommitted core backend edits in:
  - `jarvis-backend/app/actuators/system_tools.py`
  - `jarvis-backend/app/brain/memory_guardian.py`
  - `jarvis-backend/app/brain/orchestrator.py`
- Untracked local-only artifacts:
  - `scripts/.cache/` (large model/audio binaries)
  - `.vscode/`
  - `jarvis-backend/run_jarvis.bat`
- Runtime warning in backend import: duplicate tool registration for `open_local_app`.
- Runtime warning from `pydub`: ffmpeg not found (audio decoding may fail at runtime).

## 2) Recovery Objectives (Definition of Done)

Recovery is complete only when all are true:
1. Backend starts cleanly with no critical warnings/errors.
2. Core chat + tool-call path works end-to-end.
3. Frontend + Tauri build pass.
4. Audio/TTS path works (including ffmpeg dependency).
5. Git tree is clean except intentional tracked changes.
6. Local cache/model binaries are not accidentally tracked.

## 3) Phase Plan (Priority Order)

### Phase A - Protect Current Work (No Data Loss)

Actions:
- Create a safety branch before any cleanup.
- Snapshot current changed files (patch or commit).

Commands:
- `git checkout -b recovery/2026-04-10-stabilize`
- `git status --short`

Exit criteria:
- You can always return to this exact messy state if needed.

---

### Phase B - Repository Hygiene (Stop Future Mess)

Actions:
- Update `.gitignore` to ignore local cache artifacts.
- Ensure `.vscode/settings.json` is either intentionally tracked or ignored.
- Keep large artifacts out of git history.

Recommended additions to `.gitignore`:
- `scripts/.cache/`
- `jarvis-backend/venv/`
- optional: `.vscode/` (if team does not share VS Code settings)

Exit criteria:
- `git status --short` no longer lists cache binaries as untracked noise.

---

### Phase C - Backend Stabilization (Highest Runtime Risk)

Actions:
- Fix duplicate tool registration in `system_tools.py`:
  - Remove accidental `@tool(...)` decorator from helper `_resolve_app_path(...)`.
  - Keep only one registered `open_local_app` tool function.
- Align environment usage:
  - Standardize on `.venv` (current active environment).
  - If keeping `run_jarvis.bat`, make it use `.venv` (currently it creates/uses `venv`).
- Install/verify ffmpeg on Windows to satisfy `pydub` runtime dependency.

Smoke checks:
- `python -c "from app.main import app; print('ok')"`
- Start backend and check `/health`.
- Trigger one tool call (`get_system_status`, `open_local_app` with safe app) and verify response shape.

Exit criteria:
- Backend imports cleanly.
- No duplicate tool registration warning.
- Audio decode path no longer warns about missing ffmpeg.

---

### Phase D - Functional Verification Matrix

Run and record pass/fail for each:

Backend:
- Health endpoint
- Chat endpoint basic prompt
- Tool loop (at least 2 tools)
- Mode switch behavior
- TTS fallback endpoint path

Frontend:
- `npm run build`
- app launch and API connectivity

Desktop (Tauri):
- `cargo check`
- desktop app starts and calls backend

Exit criteria:
- All matrix rows pass.

---

### Phase E - Regression Safety Net

Actions:
- Add minimal automated checks if missing:
  - backend smoke script (import + health ping)
  - frontend build check
  - tauri check
- Put these in one command runner (PowerShell script or CI workflow).

Exit criteria:
- One command can validate baseline health after any future edits.

---

### Phase F - Finalize and Lock In

Actions:
- Commit in logical chunks:
  1) gitignore/repo hygiene
  2) backend tool registration fixes
  3) startup/run script normalization
  4) verification scripts
- Tag stable checkpoint.

Exit criteria:
- Clean git status.
- Reproducible local startup.
- Stable checkpoint available for rollback.

## 4) Suggested Execution Order (Fastest Safe Path)

1. Phase A + B (20-30 min)
2. Phase C backend fixes (30-60 min)
3. Phase D full verification (20-40 min)
4. Phase E/F hardening and commit (20-30 min)

Estimated total: 1.5 to 2.5 hours depending on runtime dependency installs.

## 5) Immediate Next Step

Start with Phase A and Phase B first, then fix duplicate tool registration in `jarvis-backend/app/actuators/system_tools.py` before any new feature work.
