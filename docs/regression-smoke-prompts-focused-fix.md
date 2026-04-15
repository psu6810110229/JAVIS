# Jarvis Focused Regression Smoke Prompts

Use this prompt set after every backend restart. Run in order and verify expected behaviour.

## 1) Command-router hardening (local tool intents)

Prompt 1:
Open Spotify and set volume to 35%.
Expected:
- Uses local tools for app + volume.
- Returns a step-wise outcome, not a generic web answer.

Prompt 2:
Find current date/time, then create a reminder notification that says Stand up and stretch in 15 minutes.
Expected:
- Uses local datetime/reminder flow.
- Does not use web search for this command.

Prompt 3:
Search for Python files in this project related to orchestrator, then list the top 5 relevant file paths.
Expected:
- Uses local file search tools.
- Returns file paths from this workspace.

## 2) Strict verified-only completion phrasing

Prompt 4:
Open Spotify and set volume to 25%, and only mark completed actions as verified.
Expected:
- Any failing step is reported as failed/unverified.
- No "completed" claim for unverified actions.

Prompt 5:
Mute system audio, wait 3 seconds, then unmute.
Expected:
- Reports exact status per step.
- If mute toggles fail, wording stays cautious and truthful.

## 3) No-fabrication news mode

Prompt 6:
Search latest AI news in the last 24 hours and summarize top 5 with source links.
Expected:
- Summary is grounded in retrieved results.
- Includes concrete source URLs from results.
- No invented/stale headline list.

Prompt 7:
Compare today top 3 NVIDIA headlines and explain why each matters in one sentence.
Expected:
- Uses retrieved sources only.
- If retrieval is weak, says so explicitly instead of inventing facts.

## 4) TTS non-fatal streaming behaviour

Prompt 8:
Give me a 6-sentence summary of current US-Iran news.
Expected:
- If sentence-level TTS fails, text stream still completes normally.
- No hard request interruption from sentence-level audio failure.

Prompt 9:
What time is it in Thailand right now?
Expected:
- Text response always returns even if audio is skipped.

## 5) Combined multi-step practical command

Prompt 10:
Create folder C:\Users\farnp\Desktop\JarvisTest\Reports, then create a DOCX report named daily-brief.docx with today date, 3 AI headlines, and a short conclusion.
Expected:
- Multi-step execution result with per-step status.
- Completion count and failed step names are accurate.

## Optional quick health checks

- Backend status:
  - GET /v1/system/status returns active model qwen2.5-coder:7b.
- TTS health:
  - GET http://127.0.0.1:8870/health returns status ok.
