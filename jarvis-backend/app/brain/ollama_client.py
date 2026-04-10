"""brain/ollama_client.py — Robust async Ollama HTTP client.

Responsibilities
----------------
- All network communication with the Ollama daemon.
- Streaming and non-streaming inference.
- Retry logic: EOF retry, KV-cache fallback, runner memory-pressure fallback,
  emergency eco-model downgrade.
- Connection pooling via a single long-lived httpx.AsyncClient.
- Performance benchmarking: logs TTFT (Time-to-First-Token).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.brain.exceptions import (
    OllamaModelError,
    OllamaResponseError,
    OllamaUnavailableError,
)
from app.config.settings import (
    DEFAULT_EOF_RETRY_DELAY_SECONDS,
    DEFAULT_KV_CACHE_TYPE,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_TEMPERATURE,
    FALLBACK_PROFILE,
    HardwareProfile,
    Settings,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async HTTP client for the Ollama inference API.

    Uses a persistent httpx.AsyncClient (connection pool) to reduce per-request
    TCP handshake overhead which is critical for TTFT targets.

    Parameters
    ----------
    settings:
        Shared Settings singleton (injected by Orchestrator).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._kv_cache_supported: bool = True
        self._kv_cache_warning: str | None = None
        # Reused client — keep-alive connections to Ollama container.
        self._http_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open the persistent HTTP connection pool."""
        self._http_client = httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=2,
                keepalive_expiry=30.0,
            ),
        )
        logger.info("OllamaClient connected to %s", self._settings.ollama_base_url)

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError("OllamaClient.open() must be called before use.")
        return self._http_client

    # ------------------------------------------------------------------
    # Discovery / health
    # ------------------------------------------------------------------

    async def fetch_tags(self) -> tuple[bool, dict[str, Any] | str]:
        """GET /api/tags — returns (success, payload_or_error_str)."""
        try:
            response = await self._client().get("/api/tags")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
            return False, f"unable to connect ({err})"
        except httpx.HTTPError as err:
            return False, f"HTTP error ({err})"

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"

        try:
            return True, response.json()
        except ValueError as err:
            return False, f"invalid health payload ({err})"

    async def fetch_processes(self) -> tuple[bool, dict[str, Any] | str]:
        """GET /api/ps — returns (success, payload_or_error_str)."""
        try:
            response = await self._client().get("/api/ps")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
            return False, f"unable to connect ({err})"
        except httpx.HTTPError as err:
            return False, f"HTTP error ({err})"

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}"

        try:
            return True, response.json()
        except ValueError as err:
            return False, f"invalid process payload ({err})"

    @staticmethod
    def extract_model_names(payload: dict[str, Any]) -> set[str]:
        models = payload.get("models")
        if not isinstance(models, list):
            return set()
        resolved: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if isinstance(name, str):
                resolved.add(name)
                resolved.add(name.split(":", 1)[0])
        return resolved

    async def pull_model(self, model_name: str) -> tuple[bool, str | None]:
        """POST /api/pull — blocks until the model is downloaded (up to 30 min)."""
        try:
            timeout = httpx.Timeout(1800.0)
            response = await self._client().post(
                "/api/pull",
                json={"name": model_name, "stream": False},
                timeout=timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
            return False, str(err)
        except httpx.HTTPError as err:
            return False, str(err)

        if response.status_code >= 400:
            return False, response.text.strip() or f"HTTP {response.status_code}"

        try:
            payload = response.json()
        except ValueError:
            return True, None

        payload_error = payload.get("error")
        if isinstance(payload_error, str) and payload_error.strip():
            return False, payload_error
        return True, None

    # ------------------------------------------------------------------
    # Inference options builder
    # ------------------------------------------------------------------

    def _build_options(
        self,
        *,
        num_ctx: int,
        num_thread: int,
        num_gpu: int,
        temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_thread": num_thread,
            "num_gpu": num_gpu,
        }
        if self._kv_cache_supported:
            options["kv_cache_type"] = DEFAULT_KV_CACHE_TYPE
        return options

    # ------------------------------------------------------------------
    # Non-streaming inference
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        profile: HardwareProfile,
        *,
        num_gpu: int = 0,
    ) -> str:
        """POST /api/chat (stream=False). Returns completed assistant text."""
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": self._build_options(
                num_ctx=profile.num_ctx,
                num_thread=profile.num_thread,
                num_gpu=num_gpu,
            ),
        }

        try:
            response = await self._client().post("/api/chat", json=request_payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
            raise OllamaUnavailableError(str(err)) from err
        except httpx.HTTPError as err:
            raise OllamaUnavailableError(str(err)) from err

        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            # Retry without KV cache if that specific option is unsupported
            if self._is_kv_cache_error(detail) and self._kv_cache_supported:
                self._kv_cache_supported = False
                self._kv_cache_warning = (
                    "Ollama runtime does not support 4-bit KV cache. Continuing without."
                )
                logger.warning(self._kv_cache_warning)
                return await self.generate(messages, model_name, profile, num_gpu=num_gpu)
            if self._is_memory_pressure(detail):
                return await self.generate(messages, model_name, FALLBACK_PROFILE, num_gpu=0)
            self._raise_from_detail(response.status_code, detail)

        return self._parse_content(response.json())

    # ------------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------------

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        profile: HardwareProfile,
        on_delta: Callable[[str], Awaitable[None]],
        *,
        num_gpu: int = 0,
        num_ctx_override: int | None = None,
        allow_eof_retry: bool = True,
        allow_model_fallback: bool = True,
        _fallback_profile: bool = False,
    ) -> str:
        """POST /api/chat (stream=True). Calls *on_delta* for each token chunk.

        Returns the full accumulated text.

        Benchmark logging:
            Logs TTFT (Time-to-First-Token) at INFO level on every call.
        """
        effective_num_ctx = num_ctx_override if isinstance(num_ctx_override, int) and num_ctx_override > 0 else profile.num_ctx
        effective_profile = HardwareProfile(
            mode=profile.mode,
            num_ctx=effective_num_ctx,
            num_thread=profile.num_thread,
            num_gpu=num_gpu,
            description=profile.description,
        )

        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": self._build_options(
                num_ctx=effective_profile.num_ctx,
                num_thread=effective_profile.num_thread,
                num_gpu=num_gpu,
            ),
        }

        try:
            t_start = time.monotonic()
            first_token_logged = False
            accumulated = ""

            async with self._client().stream("POST", "/api/chat", json=request_payload) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="ignore").strip()
                    detail = detail or f"HTTP {response.status_code}"

                    if self._is_kv_cache_error(detail) and self._kv_cache_supported:
                        self._kv_cache_supported = False
                        self._kv_cache_warning = "KV cache unsupported; disabled."
                        logger.warning(self._kv_cache_warning)
                        return await self.generate_streaming(
                            messages, model_name, profile, on_delta,
                            num_gpu=num_gpu, num_ctx_override=num_ctx_override,
                            allow_eof_retry=allow_eof_retry,
                            allow_model_fallback=allow_model_fallback,
                            _fallback_profile=_fallback_profile,
                        )

                    if self._is_memory_pressure(detail) and not _fallback_profile:
                        logger.warning("Memory pressure on streaming load; retrying with fallback profile.")
                        return await self.generate_streaming(
                            messages, model_name, FALLBACK_PROFILE, on_delta,
                            num_gpu=0, num_ctx_override=num_ctx_override,
                            allow_eof_retry=allow_eof_retry,
                            allow_model_fallback=allow_model_fallback,
                            _fallback_profile=True,
                        )

                    if self._is_memory_pressure(detail) and _fallback_profile and allow_model_fallback:
                        eco_model = self._settings.resolve_model("eco")
                        if eco_model != model_name:
                            logger.warning(
                                "Memory pressure on fallback profile; downgrading to eco model '%s'.",
                                eco_model,
                            )
                            return await self.generate_streaming(
                                messages, eco_model, FALLBACK_PROFILE, on_delta,
                                num_gpu=0, allow_eof_retry=allow_eof_retry,
                                allow_model_fallback=False,
                                _fallback_profile=True,
                            )

                    self._raise_from_detail(response.status_code, detail)

                try:
                    async for line in response.aiter_lines():
                        if not line or not line.strip():
                            continue

                        try:
                            payload = json.loads(line)
                        except ValueError as err:
                            raise OllamaResponseError(
                                f"invalid streaming JSON ({err})"
                            ) from err

                        payload_error = payload.get("error")
                        if isinstance(payload_error, str) and payload_error.strip():
                            self._raise_from_detail(200, payload_error)

                        msg = payload.get("message")
                        if isinstance(msg, dict):
                            chunk: str = msg.get("content") or ""
                            if chunk:
                                if not first_token_logged:
                                    ttft = time.monotonic() - t_start
                                    logger.info(
                                        "TTFT=%.3fs model='%s' mode='%s'",
                                        ttft, model_name, profile.mode,
                                    )
                                    first_token_logged = True
                                accumulated += chunk
                                await on_delta(chunk)

                except (httpx.ReadError, httpx.RemoteProtocolError, httpx.DecodingError) as err:
                    if allow_eof_retry and self._is_eof_error(str(err)):
                        await asyncio.sleep(DEFAULT_EOF_RETRY_DELAY_SECONDS)
                        logger.warning("EOF/protocol error; retrying with reduced context: %s", err)
                        return await self.generate_streaming(
                            messages, model_name, profile, on_delta,
                            num_gpu=num_gpu,
                            num_ctx_override=FALLBACK_PROFILE.num_ctx,
                            allow_eof_retry=False,
                            allow_model_fallback=allow_model_fallback,
                        )
                    raise OllamaUnavailableError(str(err)) from err

            # Empty response retry
            if not accumulated.strip():
                if allow_eof_retry:
                    await asyncio.sleep(DEFAULT_EOF_RETRY_DELAY_SECONDS)
                    return await self.generate_streaming(
                        messages, model_name, profile, on_delta,
                        num_gpu=num_gpu,
                        num_ctx_override=FALLBACK_PROFILE.num_ctx,
                        allow_eof_retry=False,
                        allow_model_fallback=allow_model_fallback,
                    )
                raise OllamaResponseError("assistant message content is empty")

            return accumulated.strip()

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
            raise OllamaUnavailableError(str(err)) from err
        except httpx.HTTPError as err:
            raise OllamaUnavailableError(str(err)) from err

    # ------------------------------------------------------------------
    # Prewarm
    # ------------------------------------------------------------------

    async def prewarm(self, model_name: str, profile: HardwareProfile, *, num_gpu: int = 0) -> None:
        """Fire a minimal 1-token request to warm model weights into memory."""
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "keep_alive": DEFAULT_OLLAMA_KEEP_ALIVE,
            "options": {
                **self._build_options(
                    num_ctx=profile.num_ctx,
                    num_thread=profile.num_thread,
                    num_gpu=num_gpu,
                    temperature=0,
                ),
                "num_predict": 1,
            },
        }
        try:
            response = await self._client().post("/api/chat", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
            raise OllamaUnavailableError(str(err)) from err
        except httpx.HTTPError as err:
            raise OllamaUnavailableError(str(err)) from err

        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            self._raise_from_detail(response.status_code, detail)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_kv_cache_error(detail: str) -> bool:
        n = detail.lower()
        return ("kv_cache" in n or "kv_type" in n) and (
            "unsupported" in n or "unknown" in n or "invalid option" in n
        )

    @staticmethod
    def _is_memory_pressure(detail: str) -> bool:
        n = detail.lower()
        return (
            "runner process has terminated" in n
            or "out of memory" in n
            or "cuda out of memory" in n
            or "failed to allocate" in n
        )

    @staticmethod
    def _is_eof_error(detail: str) -> bool:
        n = detail.lower()
        return "unexpected eof" in n or "incomplete" in n or "connection reset" in n

    @staticmethod
    def _parse_content(payload: Any) -> str:
        if not isinstance(payload, dict):
            raise OllamaResponseError("response payload is not a dict")
        payload_error = payload.get("error")
        if isinstance(payload_error, str) and payload_error.strip():
            raise OllamaUnavailableError(payload_error)
        msg = payload.get("message")
        if not isinstance(msg, dict):
            raise OllamaResponseError("missing message object")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("assistant message content is empty")
        return content.strip()

    def _raise_from_detail(self, status_code: int, detail: str) -> None:
        """Raise the appropriate exception based on the error detail."""
        lower = detail.lower()
        if status_code in (400, 404) and "model" in lower and (
            "not found" in lower or "missing" in lower
        ):
            raise OllamaModelError(detail)
        raise OllamaUnavailableError(detail)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def kv_cache_supported(self) -> bool:
        return self._kv_cache_supported

    @property
    def kv_cache_warning(self) -> str | None:
        return self._kv_cache_warning
