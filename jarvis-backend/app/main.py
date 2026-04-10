from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Literal
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from .brain import AudioProcessingError, JarvisBrain, OllamaModelError, OllamaResponseError, OllamaUnavailableError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jarvis Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
brain = JarvisBrain()


class SessionStartPayload(BaseModel):
    client_name: str | None = None


class TextInputPayload(BaseModel):
    text: str = Field(min_length=1)


class SystemModelPatchPayload(BaseModel):
    mode: Literal["eco", "performance"]


class ChatRequestPayload(BaseModel):
    message: str = Field(min_length=1)
    auto_speak: bool = Field(default=True)
    session_id: str | None = None


class SynthesizeRequestPayload(BaseModel):
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


async def send_assistant_response(
    websocket: WebSocket,
    session_id: str,
    response_text: str,
    tool_schemas: list[dict[str, Any]],
    assistant_audio: dict[str, Any] | None,
    audio_error: str | None,
) -> None:
    await websocket.send_json(
        build_envelope(
            event_type="text.output",
            payload={
                "text": response_text,
                "tool_schemas": tool_schemas,
            },
            session_id=session_id,
        )
    )

    if assistant_audio is not None:
        await websocket.send_json(
            build_envelope(
                event_type="assistant_audio",
                payload=assistant_audio,
                session_id=session_id,
            )
        )
    elif audio_error is not None:
        await websocket.send_json(
            build_envelope(
                event_type="error",
                payload={"message": audio_error},
                session_id=session_id,
            )
        )


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


@app.get("/v1/system/status")
async def get_system_status() -> dict[str, Any]:
    try:
        return await brain.get_system_status()
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.patch("/v1/system/model")
async def patch_system_model(payload: SystemModelPatchPayload) -> dict[str, Any]:
    try:
        result = await brain.set_active_mode(mode=payload.mode, prewarm=False)
        asyncio.create_task(brain.prewarm_model_non_blocking(result["active_model"]))
        result["prewarm_attempted"] = True
        result["prewarm_warning"] = None
        result["prewarm_started"] = True
        return result
    except OllamaModelError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/synthesize")
async def post_synthesize(payload: SynthesizeRequestPayload) -> dict[str, Any]:
    try:
        audio_payload = await brain.synthesize_manual_text(payload.text)
        return {
            "audio_base64": audio_payload.audio_base64,
            "mime_type": audio_payload.mime_type,
            "voice": audio_payload.voice,
        }
    except AudioProcessingError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/v1/chat")
async def post_chat(payload: ChatRequestPayload) -> StreamingResponse:
    session_id = payload.session_id or str(uuid4())
    stream_id = str(uuid4())

    async def event_stream() -> Any:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        text_index = 0

        async def on_text_chunk(delta: str) -> None:
            nonlocal text_index
            await queue.put(
                {
                    "type": "text.chunk",
                    "stream_id": stream_id,
                    "session_id": session_id,
                    "index": text_index,
                    "delta": delta,
                    "timestamp": utc_timestamp(),
                }
            )
            text_index += 1

        async def on_sentence_audio(
            sentence_index: int,
            sentence_text: str,
            audio_payload: Any,
            error_message: str | None,
        ) -> None:
            if audio_payload is None:
                await queue.put(
                    {
                        "type": "error",
                        "stream_id": stream_id,
                        "session_id": session_id,
                        "stage": "tts",
                        "sentence_index": sentence_index,
                        "sentence_text": sentence_text,
                        "message": error_message or "Failed to synthesize sentence audio.",
                        "timestamp": utc_timestamp(),
                    }
                )
                return

            await queue.put(
                {
                    "type": "audio.ready",
                    "stream_id": stream_id,
                    "session_id": session_id,
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "audio_base64": audio_payload.audio_base64,
                    "mime_type": audio_payload.mime_type,
                    "voice": audio_payload.voice,
                    "timestamp": utc_timestamp(),
                }
            )

        async def produce() -> None:
            try:
                final_text = await brain.handle_http_chat_streaming(
                    session_id=session_id,
                    user_text=payload.message,
                    auto_speak=payload.auto_speak,
                    on_text_chunk=on_text_chunk,
                    on_sentence_audio=on_sentence_audio,
                )
                await queue.put(
                    {
                        "type": "final",
                        "stream_id": stream_id,
                        "session_id": session_id,
                        "text": final_text,
                        "timestamp": utc_timestamp(),
                    }
                )
            except (OllamaUnavailableError, OllamaModelError, OllamaResponseError) as error:
                await queue.put(
                    {
                        "type": "error",
                        "stream_id": stream_id,
                        "session_id": session_id,
                        "message": str(error),
                        "timestamp": utc_timestamp(),
                    }
                )
            except RuntimeError as error:
                await queue.put(
                    {
                        "type": "error",
                        "stream_id": stream_id,
                        "session_id": session_id,
                        "message": str(error),
                        "timestamp": utc_timestamp(),
                    }
                )
            finally:
                await queue.put(None)

        producer_task = asyncio.create_task(produce())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"{json.dumps(event, ensure_ascii=False)}\n".encode("utf-8")
        except asyncio.CancelledError:
            if not producer_task.done():
                producer_task.cancel()
            raise
        finally:
            if not producer_task.done():
                producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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

                stream_id = str(uuid4())
                stream_index = 0

                async def on_stream_start(model_name: str) -> None:
                    await websocket.send_json(
                        build_envelope(
                            event_type="text.stream.start",
                            payload={"stream_id": stream_id, "model": model_name},
                            session_id=session_id,
                        )
                    )

                async def on_stream_delta(delta: str) -> None:
                    nonlocal stream_index
                    stream_index += 1
                    await websocket.send_json(
                        build_envelope(
                            event_type="text.stream.delta",
                            payload={"stream_id": stream_id, "index": stream_index, "delta": delta},
                            session_id=session_id,
                        )
                    )

                async def on_stream_end(final_text: str) -> None:
                    await websocket.send_json(
                        build_envelope(
                            event_type="text.stream.end",
                            payload={"stream_id": stream_id, "text": final_text},
                            session_id=session_id,
                        )
                    )

                reply = await brain.handle_text_streaming(
                    session_id=session_id,
                    user_text=payload.text,
                    on_stream_start=on_stream_start,
                    on_stream_delta=on_stream_delta,
                    on_stream_end=on_stream_end,
                )
                await send_assistant_response(
                    websocket=websocket,
                    session_id=session_id,
                    response_text=reply.text,
                    tool_schemas=[schema.model_dump() for schema in reply.tool_schemas],
                    assistant_audio=reply.assistant_audio.model_dump() if reply.assistant_audio else None,
                    audio_error=reply.audio_error,
                )
                continue

            if envelope.type == "audio.chunk":
                try:
                    audio_payload = brain.parse_audio_payload(envelope.payload)
                    result = await brain.handle_audio_chunk(session_id=session_id, payload=audio_payload)
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
                except AudioProcessingError as error:
                    logger.warning("Audio processing failed for session '%s': %s", session_id, error)
                    await websocket.send_json(
                        build_envelope(
                            event_type="error",
                            payload={"message": str(error)},
                            session_id=session_id,
                        )
                    )
                    continue

                await websocket.send_json(
                    build_envelope(
                        event_type="audio.ack",
                        payload=result.acknowledgement.model_dump(),
                        session_id=session_id,
                    )
                )
                await websocket.send_json(
                    build_envelope(
                        event_type="speech.transcript",
                        payload=result.transcript.model_dump(),
                        session_id=session_id,
                    )
                )
                await send_assistant_response(
                    websocket=websocket,
                    session_id=session_id,
                    response_text=result.response.text,
                    tool_schemas=[schema.model_dump() for schema in result.response.tool_schemas],
                    assistant_audio=result.response.assistant_audio.model_dump()
                    if result.response.assistant_audio
                    else None,
                    audio_error=result.response.audio_error,
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
