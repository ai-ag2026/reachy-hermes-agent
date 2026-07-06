import asyncio
import json
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

VOICE_FAST_SYSTEM_MESSAGE = """Voice fast mode for Reachy/AGENT.
You are producing the text that will be spoken aloud by Reachy.
Answer as AGENT: concise, natural, direct, and useful.
Keep responses short unless the user explicitly asks for detail.
Use the recent voice context to resolve short follow-ups such as "yes", "we can do that", "ok", or "make it faster".
If the user confirms or refers back to the previous turn, continue that thread instead of treating the utterance as a new standalone task.
Do not perform long tool work inline; if more work is needed, say briefly that it will continue in the background.
""".strip()

_EMPTY_TRANSCRIPT_FALLBACK = "Das habe ich akustisch nicht erwischt."
_TIMEOUT_FALLBACK = "Ich prüfe das weiter im Hintergrund."
_ERROR_FALLBACK = "Da hakt gerade die Verbindung zu AGENT."


class AsyncPostClient(Protocol):
    """Minimal async HTTP client protocol used by HermesAgentClient."""

    async def post(self, url: str, **kwargs: Any) -> Any:
        """Send an async POST request."""


@dataclass(frozen=True)
class AgentBridgeConfig:
    """Configuration for the Hermes/AGENT voice bridge."""

    base_url: str = "http://127.0.0.1:8642/v1"
    model: str = "local-agent"
    api_key_env: str = "API_SERVER_KEY"
    timeout_seconds: float = 2.5
    long_task_timeout_seconds: float = 3600.0
    max_response_chars: int = 900
    history_turns: int = 6
    session_id: str = ""
    session_key: str = ""
    session_title: str = ""

    @classmethod
    def from_env(cls) -> "AgentBridgeConfig":
        """Build config from AGENT_* environment variables."""
        defaults = cls()
        return cls(
            base_url=os.getenv("AGENT_BASE_URL", defaults.base_url),
            model=os.getenv("AGENT_MODEL", defaults.model),
            api_key_env=os.getenv("AGENT_API_KEY_ENV", defaults.api_key_env),
            timeout_seconds=_env_float("AGENT_FAST_TIMEOUT_SECONDS", defaults.timeout_seconds),
            long_task_timeout_seconds=_env_float("AGENT_LONG_TASK_TIMEOUT_SECONDS", defaults.long_task_timeout_seconds),
            max_response_chars=_env_int("AGENT_MAX_RESPONSE_CHARS", defaults.max_response_chars),
            history_turns=_env_int("AGENT_VOICE_HISTORY_TURNS", defaults.history_turns),
            session_id=os.getenv("AGENT_SESSION_ID", defaults.session_id),
            session_key=os.getenv("AGENT_SESSION_KEY", defaults.session_key),
            session_title=os.getenv("AGENT_SESSION_TITLE", defaults.session_title),
        )

    def for_new_voice_session(self) -> "AgentBridgeConfig":
        """Return config for one browser/WebRTC voice session Hermes thread."""
        started = datetime.now().astimezone()
        timestamp_id = started.strftime("%Y%m%d-%H%M")
        timestamp_title = started.strftime("%d.%m. %H:%M")
        suffix = uuid.uuid4().hex[:8]
        session_id = self.session_id or f"reachy-voice-{timestamp_id}-{suffix}"
        session_title = self.session_title or f"Reachy Voice - {timestamp_title} - {suffix[:4]}"
        session_key = self.session_key or "reachy-voice"
        return replace(self, session_id=session_id, session_key=session_key, session_title=session_title)


class HermesAgentClient:
    """Small OpenAI-compatible client for asking Hermes/AGENT in voice-fast mode."""

    def __init__(self, config: AgentBridgeConfig | None = None, http_client: AsyncPostClient | None = None) -> None:
        """Initialize the client with injectable HTTP transport for tests."""
        self.config = config or AgentBridgeConfig.from_env()
        self._http_client = http_client
        self._history: list[tuple[str, str]] = []

    async def ask_fast(self, transcript: str) -> str:
        """Ask AGENT for a short spoken response to a completed user transcript."""
        cleaned_transcript = transcript.strip()
        if not cleaned_transcript:
            return _EMPTY_TRANSCRIPT_FALLBACK

        try:
            response = await self._post_chat_completion(cleaned_transcript)
            response.raise_for_status()
            text = self._extract_response_text(response.json())
            answer = self._trim_for_voice(text) if text else _ERROR_FALLBACK
            self._remember_turn(cleaned_transcript, answer)
            return answer
        except TimeoutError:
            logger.info("AGENT voice-fast request timed out")
            return _TIMEOUT_FALLBACK
        except Exception as exc:
            if _is_timeout_error(exc):
                logger.info("AGENT voice-fast request timed out: %s", exc.__class__.__name__)
                return _TIMEOUT_FALLBACK
            logger.warning("AGENT voice-fast request failed: %s", exc.__class__.__name__)
            return _ERROR_FALLBACK

    async def ask_long(self, transcript: str) -> str:
        """Ask AGENT for long-running work without using the voice-fast timeout."""
        cleaned_transcript = transcript.strip()
        if not cleaned_transcript:
            return _EMPTY_TRANSCRIPT_FALLBACK
        response = await self._post_chat_completion(
            cleaned_transcript, timeout_seconds=self.config.long_task_timeout_seconds
        )
        response.raise_for_status()
        text = self._extract_response_text(response.json())
        answer = self._trim_for_voice(text) if text else _ERROR_FALLBACK
        self._remember_turn(cleaned_transcript, answer)
        return answer

    def _remember_turn(self, transcript: str, answer: str) -> None:
        """Store bounded local voice context so short follow-ups are not orphaned."""
        max_turns = max(0, self.config.history_turns)
        if max_turns <= 0:
            self._history.clear()
            return
        self._history.append((transcript, answer))
        del self._history[:-max_turns]

    def _messages_for_transcript(self, transcript: str) -> list[dict[str, str]]:
        """Build chat messages with recent voice context."""
        messages = [{"role": "system", "content": VOICE_FAST_SYSTEM_MESSAGE}]
        for user_text, assistant_text in self._history[-max(0, self.config.history_turns) :]:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": transcript})
        return messages

    async def _post_chat_completion(self, transcript: str, timeout_seconds: float | None = None) -> Any:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        _add_optional_header(headers, "X-Hermes-Session-Id", self.config.session_id)
        _add_optional_header(headers, "X-Hermes-Session-Key", self.config.session_key)
        _add_optional_header(headers, "X-Hermes-Session-Title", self.config.session_title)

        payload = {
            "model": self.config.model,
            "messages": self._messages_for_transcript(transcript),
            "max_tokens": 220,
            "temperature": 0.6,
            "tools": [],
            "tool_choice": "none",
            "enabled_toolsets": [],
        }

        client = self._http_client
        url = self._chat_completions_url()
        request_timeout = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds
        if client is not None:
            return await client.post(url, json=payload, headers=headers, timeout=request_timeout)

        try:
            async with _build_httpx_async_client(request_timeout) as http_client:
                return await http_client.post(url, json=payload, headers=headers)
        except ModuleNotFoundError as exc:
            if exc.name != "httpx":
                raise
            return await asyncio.to_thread(
                _post_json_with_urllib,
                url,
                payload,
                headers,
                request_timeout,
            )

    def _chat_completions_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _extract_response_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        message = first_choice.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content.strip() if isinstance(content, str) else ""

    def _trim_for_voice(self, text: str) -> str:
        stripped = text.strip()
        limit = self.config.max_response_chars
        if limit <= 0 or len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "…"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float value for %s=%r, using default=%s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int value for %s=%r, using default=%s", name, raw, default)
        return default


def _add_optional_header(headers: dict[str, str], name: str, value: str) -> None:
    """Add a safe single-line ASCII HTTP header when the value is non-empty."""
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        return
    single_line = cleaned.replace("\r", " ").replace("\n", " ")
    headers[name] = single_line.encode("ascii", "replace").decode("ascii")


class _UrllibJsonResponse:
    """Tiny response adapter matching the subset of httpx used by HermesAgentClient."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _post_json_with_urllib(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> _UrllibJsonResponse:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        return _UrllibJsonResponse(getattr(response, "status", 200), parsed if isinstance(parsed, dict) else {})


def _build_httpx_async_client(timeout_seconds: float | None) -> Any:
    """Build a real httpx async client lazily so tests with fake transports need no project deps."""
    import httpx

    return httpx.AsyncClient(timeout=timeout_seconds)


def _is_timeout_error(exc: Exception) -> bool:
    """Return true for stdlib/httpx/httpcore timeout exceptions without importing optional deps."""
    name = type(exc).__name__.lower()
    return isinstance(exc, TimeoutError) or "timeout" in name
