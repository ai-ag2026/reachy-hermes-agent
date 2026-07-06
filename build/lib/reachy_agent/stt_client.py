from __future__ import annotations

import json
import mimetypes
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class SttClientConfig:
    """Configuration for an OpenAI-compatible STT endpoint."""

    base_url: str
    model: str = "parakeet"
    timeout_seconds: float = 20.0
    endpoint_path: str = "/v1/audio/transcriptions"
    response_format: str = "verbose_json"


@dataclass(frozen=True)
class SttResult:
    """Transcription result metadata."""

    text: str
    language: str | None = None
    duration_seconds: float | None = None
    status: str = "ok"
    error: str | None = None


UrlOpener = Callable[[urllib.request.Request, float], Any]


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    """Call urllib with an explicit timeout while keeping a testable opener shape."""
    return urllib.request.urlopen(request, timeout=timeout)


class SttClient:
    """Small stdlib STT client for fixture-audio transcription smokes."""

    def __init__(self, config: SttClientConfig, *, opener: UrlOpener | None = None) -> None:
        """Store config and optional opener for tests."""
        self.config = config
        self._opener = opener or _default_opener

    def transcribe(self, audio_path: Path) -> SttResult:
        """POST an audio file to the STT endpoint and parse transcript text."""
        if not audio_path.exists():
            return SttResult(text="", status="error", error="audio_file_not_found")
        body, content_type = _multipart_body(
            audio_path,
            fields={"model": self.config.model, "response_format": self.config.response_format},
        )
        request = urllib.request.Request(self._url(), data=body, method="POST", headers={"Content-Type": content_type})
        try:
            with self._opener(request, self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8", "replace")
                status = getattr(response, "status", 200)
        except Exception as exc:  # noqa: BLE001
            return SttResult(text="", status="error", error=f"transport_error:{type(exc).__name__}:{exc}")
        if status >= 400:
            return SttResult(text="", status="error", error=f"http_status:{status}")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return SttResult(text="", status="error", error=f"invalid_json:{exc.msg}")
        text = str(payload.get("text") or "").strip()
        if not text:
            return SttResult(text="", status="error", error="empty_transcript")
        language = payload.get("language")
        duration = payload.get("duration") or payload.get("duration_seconds")
        return SttResult(
            text=text,
            language=str(language) if language else None,
            duration_seconds=float(duration) if isinstance(duration, int | float) else None,
            status="ok",
        )

    def _url(self) -> str:
        base = self.config.base_url.rstrip("/")
        path = self.config.endpoint_path
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        return f"{base}{path}"


def inspect_audio_file(path: Path) -> dict[str, Any]:
    """Return metadata-only details for an audio fixture."""
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    info.update(
        {
            "bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
            "mime_type": _mime_type(path),
        }
    )
    return info


def _mime_type(path: Path) -> str:
    if path.suffix.lower() == ".wav":
        return "audio/wav"
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _multipart_body(audio_path: Path, *, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----AGENTSTTBoundary7MA4YWxkTrZu0gW"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode("utf-8")
        )
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
            f"Content-Type: {_mime_type(audio_path)}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(audio_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
