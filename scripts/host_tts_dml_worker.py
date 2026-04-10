from __future__ import annotations

import argparse
import io
import json
import logging
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import requests
import soundfile as sf
from kokoro_onnx import Kokoro

LOGGER = logging.getLogger("host_tts_dml_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
DEFAULT_MODEL_FILE = "kokoro-v1.0.onnx"
DEFAULT_VOICES_FILE = "voices-v1.0.bin"
DEFAULT_VRAM_LIMIT_MB = 500.0


@dataclass(slots=True)
class LocalTtsEngine:
    kokoro: Kokoro
    provider_name: str
    model_path: Path
    voices_path: Path

    @property
    def estimated_vram_mb(self) -> float:
        total_bytes = self.model_path.stat().st_size + self.voices_path.stat().st_size
        return round(total_bytes / (1024 * 1024), 2)

    def synthesize(self, text: str, voice: str, speed: float, lang: str) -> bytes:
        audio, sample_rate = self.kokoro.create(text=text, voice=voice, speed=speed, lang=lang)
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)

        with io.BytesIO() as output:
            sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
            return output.getvalue()


class WorkerConfig:
    def __init__(
        self,
        fallback_url: str,
        model_dir: Path,
        model_url: str,
        voices_url: str,
        vram_limit_mb: float,
    ) -> None:
        self.fallback_url = fallback_url
        self.model_dir = model_dir
        self.model_url = model_url
        self.voices_url = voices_url
        self.vram_limit_mb = vram_limit_mb
        self.provider_name = "cpu-fallback-proxy"
        self.vram_mb = 0.0
        self.last_status = "idle"
        self.last_warning: str | None = None
        self.model_path, self.voices_path = self._prepare_model_assets()
        self.provider_candidates = self._resolve_providers()
        self.provider_index = 0
        self.engine = self._build_engine()

    def _download_file(self, url: str, target: Path) -> None:
        LOGGER.info("Downloading %s -> %s", url, target)
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    def _prepare_model_assets(self) -> tuple[Path, Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_name = Path(urlparse(self.model_url).path).name or DEFAULT_MODEL_FILE
        voices_name = Path(urlparse(self.voices_url).path).name or DEFAULT_VOICES_FILE
        model_path = self.model_dir / model_name
        voices_path = self.model_dir / voices_name

        if not model_path.exists():
            self._download_file(self.model_url, model_path)
        if not voices_path.exists():
            self._download_file(self.voices_url, voices_path)

        return model_path, voices_path

    def _resolve_providers(self) -> list[str]:
        requested = os.getenv(
            "JARVIS_TTS_PROVIDER_ORDER",
            "DmlExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider",
        )
        requested_list = [item.strip() for item in requested.split(",") if item.strip()]
        available = set(ort.get_available_providers())
        providers = [provider for provider in requested_list if provider in available]

        if not providers:
            providers = ["CPUExecutionProvider"]

        return providers

    def _build_session(self, provider: str) -> ort.InferenceSession:
        provider_options: list[dict[str, Any]] | None = None
        if provider == "DmlExecutionProvider":
            dml_device_id = os.getenv("JARVIS_TTS_DML_DEVICE_ID", "0").strip()
            provider_options = [{"device_id": int(dml_device_id)}]

        if provider_options is not None:
            return ort.InferenceSession(
                str(self.model_path),
                providers=[provider],
                provider_options=provider_options,
            )
        return ort.InferenceSession(str(self.model_path), providers=[provider])

    def _build_engine(self) -> LocalTtsEngine | None:
        try:
            provider = self.provider_candidates[self.provider_index]
            session = self._build_session(provider)

            provider_name = session.get_providers()[0] if session.get_providers() else "CPUExecutionProvider"
            kokoro = Kokoro.from_session(session=session, voices_path=str(self.voices_path))
            engine = LocalTtsEngine(
                kokoro=kokoro,
                provider_name=provider_name,
                model_path=self.model_path,
                voices_path=self.voices_path,
            )

            estimated = engine.estimated_vram_mb
            if estimated > self.vram_limit_mb:
                self.last_warning = (
                    f"Estimated model memory {estimated:.2f}MB exceeds limit {self.vram_limit_mb:.2f}MB. "
                    "Using remote fallback mode."
                )
                self.provider_name = "cpu-fallback-proxy"
                self.vram_mb = 0.0
                return None

            self.provider_name = provider_name
            self.vram_mb = estimated
            self.last_warning = None
            return engine
        except Exception as error:  # noqa: BLE001
            provider = self.provider_candidates[self.provider_index] if self.provider_candidates else "unknown"
            self.last_warning = f"Local TTS initialization failed on {provider}: {error}. Using remote fallback mode."
            self.provider_name = "cpu-fallback-proxy"
            self.vram_mb = 0.0
            LOGGER.exception("Local TTS initialization failed")
            return None

    def _try_step_down_provider(self) -> bool:
        if self.provider_index + 1 >= len(self.provider_candidates):
            return False

        self.provider_index += 1
        self.engine = self._build_engine()
        return self.engine is not None

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, str]:
        text = str(payload.get("input", "")).strip()
        if not text:
            raise ValueError("Missing or empty 'input' for TTS request.")

        voice = str(payload.get("voice") or "bm_george")
        lang = str(payload.get("lang") or "en-us")
        speed_raw = payload.get("speed", 1.0)

        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            speed = 1.0

        if self.engine is not None:
            self.last_status = "running"
            try:
                audio = self.engine.synthesize(text=text, voice=voice, speed=speed, lang=lang)
                self.provider_name = self.engine.provider_name
                self.vram_mb = self.engine.estimated_vram_mb
                self.last_status = "ready"
                return audio, "audio/wav"
            except Exception as error:  # noqa: BLE001
                current_provider = self.engine.provider_name
                self.last_warning = f"Local inference failed on {current_provider}: {error}."
                LOGGER.exception("Local inference failed")

                if self._try_step_down_provider() and self.engine is not None:
                    self.last_status = "retry-local"
                    try:
                        audio = self.engine.synthesize(text=text, voice=voice, speed=speed, lang=lang)
                        self.provider_name = self.engine.provider_name
                        self.vram_mb = self.engine.estimated_vram_mb
                        self.last_status = "ready"
                        self.last_warning = f"Switched provider from {current_provider} to {self.provider_name} after local failure."
                        return audio, "audio/wav"
                    except Exception as retry_error:  # noqa: BLE001
                        self.last_warning = (
                            f"Local inference failed on {current_provider} and retry provider {self.engine.provider_name}: "
                            f"{retry_error}. Falling back to remote CPU service."
                        )
                        LOGGER.exception("Local retry inference failed")

        self.last_status = "fallback"
        response = requests.post(self.fallback_url, json=payload, timeout=60)
        response.raise_for_status()
        self.provider_name = "cpu-service"
        self.vram_mb = 0.0
        return response.content, response.headers.get("Content-Type", "audio/mpeg")


def run_worker(
    port: int,
    fallback_url: str,
    model_dir: Path,
    model_url: str,
    voices_url: str,
    vram_limit_mb: float,
) -> None:
    config = WorkerConfig(
        fallback_url=fallback_url,
        model_dir=model_dir,
        model_url=model_url,
        voices_url=voices_url,
        vram_limit_mb=vram_limit_mb,
    )

    class Handler(BaseHTTPRequestHandler):
        @staticmethod
        def _sanitize_header_value(value: str, max_len: int = 300) -> str:
            sanitized = value.replace("\r", " ").replace("\n", " ").strip()
            if len(sanitized) > max_len:
                sanitized = sanitized[: max_len - 3] + "..."
            return sanitized

        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _binary_response(self, status: int, payload: bytes, mime_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-TTS-Provider", config.provider_name)
            self.send_header("X-TTS-VRAM-MB", f"{config.vram_mb:.2f}")
            if config.last_warning:
                self.send_header("X-TTS-Warning", self._sanitize_header_value(config.last_warning))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json_response(
                    200,
                    {
                        "status": "ok",
                        "provider": config.provider_name,
                        "vram_mb": config.vram_mb,
                        "mode": "local" if config.engine is not None else "fallback",
                        "last_status": config.last_status,
                        "warning": config.last_warning,
                        "provider_index": config.provider_index,
                        "provider_candidates": config.provider_candidates,
                        "fallback_url": config.fallback_url,
                    },
                )
                return
            self._json_response(404, {"error": "not-found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/audio/speech":
                self._json_response(404, {"error": "not-found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json_response(400, {"error": "invalid-json"})
                return

            try:
                audio, mime_type = config.synthesize(payload)
            except ValueError as error:
                self._json_response(400, {"error": "invalid-request", "detail": str(error)})
                return
            except requests.RequestException as error:
                LOGGER.exception("Fallback TTS call failed: %s", error)
                self._json_response(502, {"error": "fallback-tts-unavailable", "detail": str(error)})
                return
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("TTS synthesis failed: %s", error)
                self._json_response(500, {"error": "tts-synthesis-failed", "detail": str(error)})
                return

            self._binary_response(200, audio, mime_type)

    server = HTTPServer(("127.0.0.1", port), Handler)
    LOGGER.info("Host TTS worker listening on http://127.0.0.1:%s", port)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Host TTS worker for DirectML/OpenVINO offload path.")
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--fallback-url", type=str, default="http://127.0.0.1:8880/v1/audio/speech")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / ".cache" / "kokoro"),
    )
    parser.add_argument("--model-url", type=str, default=DEFAULT_MODEL_URL)
    parser.add_argument("--voices-url", type=str, default=DEFAULT_VOICES_URL)
    parser.add_argument("--vram-limit-mb", type=float, default=DEFAULT_VRAM_LIMIT_MB)
    args = parser.parse_args()
    run_worker(
        port=args.port,
        fallback_url=args.fallback_url,
        model_dir=Path(args.model_dir),
        model_url=args.model_url,
        voices_url=args.voices_url,
        vram_limit_mb=args.vram_limit_mb,
    )


if __name__ == "__main__":
    main()
