"""brain/exceptions.py — Custom exception hierarchy for the Jarvis brain."""
from __future__ import annotations


class OllamaUnavailableError(Exception):
    """Raised when the Ollama runtime cannot be reached or returns service errors."""


class OllamaModelError(Exception):
    """Raised when the configured model is missing from Ollama."""


class OllamaResponseError(Exception):
    """Raised when Ollama response payload does not contain expected fields."""


class AudioProcessingError(Exception):
    """Raised when voice input or TTS output processing fails."""
