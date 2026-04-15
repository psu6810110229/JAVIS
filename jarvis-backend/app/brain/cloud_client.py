"""Async client for Groq/OpenAI-compatible cloud chat APIs."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

_DEFAULT_MAX_RETRIES = 2
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 1024


class CloudLLMError(Exception):
    """Base error for cloud LLM operations."""


class CloudLLMTimeoutError(CloudLLMError):
    """Cloud request timed out."""


class CloudLLMAuthError(CloudLLMError):
    """API key invalid or missing."""


class CloudLLMRateLimitError(CloudLLMError):
    """Rate limited by provider."""


class CloudLLMUnavailableError(CloudLLMError):
    """Cloud service unavailable."""


@dataclass
class CloudResponse:
    text: str
    model: str
    usage: dict[str, int]
    finish_reason: str


class CloudClient:
    """Simple resilient cloud chat client with retry and streaming support."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 15.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise CloudLLMAuthError("Cloud API key is required.")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        response_format: dict[str, str] | None = None,
    ) -> CloudResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            body["response_format"] = response_format

        data = await self._request_with_retry(body)
        return self._parse_response(data)

    async def chat_streaming(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> CloudResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        client = await self._ensure_client()
        collected_text: list[str] = []
        finish_reason = "stop"
        model_name = self._model

        try:
            async with client.stream("POST", "/chat/completions", json=body) as response:
                self._check_status(response.status_code, "")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    model_name = chunk.get("model", model_name)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        collected_text.append(content)
                        if on_delta is not None:
                            await on_delta(content)
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

        except httpx.TimeoutException as err:
            raise CloudLLMTimeoutError(f"Cloud streaming timed out: {err}") from err
        except httpx.HTTPError as err:
            raise CloudLLMUnavailableError(f"Cloud streaming failed: {err}") from err

        return CloudResponse(
            text="".join(collected_text),
            model=model_name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason=finish_reason,
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.text.strip()
        try:
            decoded = json.loads(text)
            if isinstance(decoded, (dict, list)):
                return decoded
            raise CloudLLMError("Cloud JSON response has unsupported type.")
        except json.JSONDecodeError as err:
            raise CloudLLMError(f"Cloud returned invalid JSON: {err}") from err

    async def _request_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                client = await self._ensure_client()
                response = await client.post("/chat/completions", json=body)
                self._check_status(response.status_code, response.text)
                return response.json()
            except (CloudLLMRateLimitError, CloudLLMUnavailableError) as err:
                last_error = err
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                    continue
                raise
            except httpx.TimeoutException as err:
                raise CloudLLMTimeoutError(f"Cloud request timed out after {self._timeout}s: {err}") from err
            except httpx.HTTPError as err:
                raise CloudLLMUnavailableError(f"Cloud HTTP error: {err}") from err

        raise last_error or CloudLLMError("All retry attempts exhausted.")

    @staticmethod
    def _check_status(status_code: int, response_text: str) -> None:
        if 200 <= status_code < 300:
            return
        if status_code == 401:
            raise CloudLLMAuthError("Invalid cloud API key (401).")
        if status_code == 429:
            raise CloudLLMRateLimitError("Cloud rate limit exceeded (429).")
        if status_code >= 500:
            raise CloudLLMUnavailableError(f"Cloud server error ({status_code}).")
        raise CloudLLMError(f"Cloud API error {status_code}: {response_text[:200]}")

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> CloudResponse:
        choices = data.get("choices", [])
        if not choices:
            raise CloudLLMError("Cloud returned empty choices.")

        message = choices[0].get("message", {})
        text = str(message.get("content", "")).strip()
        finish_reason = str(choices[0].get("finish_reason", "stop"))
        model = str(data.get("model", "unknown"))

        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        return CloudResponse(text=text, model=model, usage=usage, finish_reason=finish_reason)
