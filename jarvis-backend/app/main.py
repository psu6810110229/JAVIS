from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from .brain import JarvisBrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jarvis Backend", version="0.1.0")
brain = JarvisBrain()


class SessionStartPayload(BaseModel):
    client_name: str | None = None


class TextInputPayload(BaseModel):
    text: str = Field(min_length=1)


class MessageEnvelope(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None
    session_id: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_envelope(
    event_type: str,
    payload: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    return {
        "type": event_type,
        "payload": payload,
        "timestamp": utc_timestamp(),
        "session_id": session_id,
    }


@app.on_event("startup")
async def startup_event() -> None:
    await brain.initialize()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "jarvis-backend",
        "timestamp": utc_timestamp(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid4())
    await brain.create_session(session_id)

    await websocket.send_json(
        build_envelope(
            event_type="session.started",
            payload={
                "message": "Jarvis session established.",
                "available_tools": [schema.model_dump() for schema in brain.tool_schemas],
            },
            session_id=session_id,
        )
    )

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                decoded_message = json.loads(raw_message)
                envelope = MessageEnvelope.model_validate(decoded_message)
            except (json.JSONDecodeError, ValidationError) as error:
                logger.warning("Invalid WebSocket envelope received: %s", error)
                await websocket.send_json(
                    build_envelope(
                        event_type="error",
                        payload={"message": "Invalid message envelope."},
                        session_id=session_id,
                    )
                )
                continue

            if envelope.type == "session.start":
                try:
                    payload = SessionStartPayload.model_validate(envelope.payload)
                except ValidationError as error:
                    logger.warning("Invalid session payload received: %s", error)
                    await websocket.send_json(
                        build_envelope(
                            event_type="error",
                            payload={"message": "The session payload is invalid."},
                            session_id=session_id,
                        )
                    )
                    continue

                await websocket.send_json(
                    build_envelope(
                        event_type="session.ready",
                        payload={
                            "message": "Jarvis is ready.",
                            "client_name": payload.client_name,
                        },
                        session_id=session_id,
                    )
                )
                continue

            if envelope.type == "text.input":
                try:
                    payload = TextInputPayload.model_validate(envelope.payload)
                except ValidationError as error:
                    logger.warning("Invalid text payload received: %s", error)
                    await websocket.send_json(
                        build_envelope(
                            event_type="error",
                            payload={"message": "The text payload is invalid."},
                            session_id=session_id,
                        )
                    )
                    continue

                await websocket.send_json(
                    build_envelope(
                        event_type="text.received",
                        payload={"text": payload.text},
                        session_id=session_id,
                    )
                )

                reply = await brain.handle_text(session_id=session_id, user_text=payload.text)
                await websocket.send_json(
                    build_envelope(
                        event_type="text.output",
                        payload={
                            "text": reply.text,
                            "tool_schemas": [schema.model_dump() for schema in reply.tool_schemas],
                        },
                        session_id=session_id,
                    )
                )
                continue

            if envelope.type == "audio.chunk":
                try:
                    audio_payload = brain.parse_audio_payload(envelope.payload)
                except ValueError as error:
                    logger.warning("Invalid audio payload received: %s", error)
                    await websocket.send_json(
                        build_envelope(
                            event_type="error",
                            payload={"message": "The audio payload is invalid."},
                            session_id=session_id,
                        )
                    )
                    continue

                acknowledgement = await brain.handle_audio_chunk(audio_payload)
                await websocket.send_json(
                    build_envelope(
                        event_type="audio.ack",
                        payload=acknowledgement.model_dump(),
                        session_id=session_id,
                    )
                )
                continue

            if envelope.type == "session.end":
                await websocket.send_json(
                    build_envelope(
                        event_type="session.closed",
                        payload={"message": "Jarvis session closed."},
                        session_id=session_id,
                    )
                )
                break

            await websocket.send_json(
                build_envelope(
                    event_type="error",
                    payload={"message": f"Unsupported message type '{envelope.type}'."},
                    session_id=session_id,
                )
            )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from session '%s'.", session_id)
    finally:
        await brain.close_session(session_id)
