"""tts — Kokoro TTS host-worker client package."""

from .host_client import TtsEngineError, TtsHostClient, TtsSynthesisPayload

__all__ = [
    "TtsEngineError",
    "TtsHostClient",
    "TtsSynthesisPayload",
]
