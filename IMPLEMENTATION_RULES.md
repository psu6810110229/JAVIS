# Jarvis Implementation Rules

This file exists to keep the implementation aligned with the project contract while coding.

## Non-Negotiable Guardrails
1. No Hallucination: Do not invent libraries, frameworks, or APIs that do not exist or were not requested.
2. Zero Over-engineering: Write exactly what is needed to fulfill the requested step. Do not add unnecessary abstractions or premature optimizations.
3. Strict Tech Stack: Use FastAPI for the backend and Tauri v2 + React + Tailwind for the frontend. Do not substitute Kivy, Tkinter, Electron, or unrelated stacks.
4. Scope Containment: Stop exactly where the active phase ends. Do not implement Phase 2 work during Phase 1.
5. No Spaghetti Code: Keep the code modular, cohesive, and easy to follow.
6. Type Safety: Use strict Python type hints and Pydantic models. Use TypeScript for React code.
7. Async by Default: Keep backend I/O asynchronous, including WebSocket handling and Gemini requests.
8. Error Handling: Never use bare `except:` blocks. Catch specific exceptions and log them clearly.
9. No Silly Placeholders: Core logic must be functional. Avoid placeholder comments for required behavior.
10. Self-Correction: Re-check generated code against these rules before considering the implementation complete.

## Phase 1 Scope
- Build the decoupled `jarvis-backend/` and `jarvis-frontend/` scaffold.
- Add backend infrastructure files for FastAPI + Docker.
- Add the FastAPI entrypoint with a JSON-envelope WebSocket route.
- Add the `JarvisBrain` controller with Gemini integration, chat-session memory, and safe demo tool schemas.
- Add the minimal React + Tailwind shell that connects to the backend over WebSocket.

## Explicit Phase 1 Exclusions
- No microphone capture pipeline.
- No wake-word detection.
- No system tray integration.
- No desktop automation or OS control tools.
- No Electron-based shell.
