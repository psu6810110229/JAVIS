"""brain/models.py — Pydantic data-transfer objects for the Jarvis brain."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """JSON schema descriptor for a callable tool (used in Phase 4 function calling)."""

    name: str
    description: str
    parameters: dict[str, Any]


class AssistantAudioPayload(BaseModel):
    """Base64-encoded audio produced by the TTS engine."""

    audio_base64: str
    mime_type: str
    voice: str


class BrainResponse(BaseModel):
    """Unified response from the brain for both text and voice interactions."""

    text: str
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    assistant_audio: AssistantAudioPayload | None = None
    audio_error: str | None = None


class AudioChunkAcknowledgement(BaseModel):
    """Confirmation payload returned after receiving an audio chunk."""

    accepted: bool
    chunk_size: int
    mime_type: str
    detail: str


class AudioChunkPayload(BaseModel):
    """Inbound base64-encoded audio sent from the browser/client."""

    mime_type: str = Field(default="audio/webm")
    data: str
    is_final: bool = Field(default=True)


class SpeechTranscriptPayload(BaseModel):
    """Transcribed text from a voice input chunk."""

    text: str
    source: str = Field(default="stt")


class VoiceInteractionResult(BaseModel):
    """Composite result returned after processing a voice audio chunk."""

    acknowledgement: AudioChunkAcknowledgement
    transcript: SpeechTranscriptPayload
    response: BrainResponse
