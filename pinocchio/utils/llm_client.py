"""LLM Client -- abstraction for calling language model APIs.

Supports OpenAI-compatible APIs (Ollama, DashScope / Qwen, OpenAI, Azure,
vLLM, etc.) with structured retry, token counting, and multimodal
message construction.

Default backend: Qwen3-VL via local Ollama server.

Performance features:
- HTTP connection pooling via httpx (reuses TCP connections)
- Thread-safe: safe to call from multiple parallel workers
- Async support via ``AsyncLLMClient`` for high-throughput pipelines
- Configurable timeout and retry
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import threading
from pathlib import Path
from typing import Any

from collections.abc import Generator

import httpx
import openai

# Regex to strip Qwen3 thinking tags from responses
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_DEFAULT_MODEL = os.getenv("PINOCCHIO_MODEL", "qwen3-vl:4b")
_DEFAULT_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "http://localhost:11434/v1",
)


# ======================================================================
# Token tracker
# ======================================================================

class TokenTracker:
    """Accumulative token usage tracker.

    Thread-safe: each field is updated atomically via simple addition.
    Designed to be shared across all calls on a single LLMClient.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0

    def record(self, usage: Any) -> None:
        """Record token usage from an OpenAI-compatible usage object."""
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0
        self.call_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0


_logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker for LLM API calls.

    States:
      - CLOSED (normal): requests pass through.
      - OPEN: requests are rejected immediately for *cooldown* seconds.
      - HALF-OPEN: after cooldown expires, one probe request is allowed.
        If it succeeds, state → CLOSED.  If it fails, state → OPEN again.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, cooldown: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._opened_at >= self._cooldown:
                    self._state = self.HALF_OPEN
            return self._state

    def allow_request(self) -> bool:
        """Return True if the request should proceed."""
        s = self.state
        return s != self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                _logger.warning(
                    "Circuit breaker OPEN after %d consecutive failures — "
                    "cooling down for %.0fs",
                    self._consecutive_failures,
                    self._cooldown,
                )

    def reset(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = self.CLOSED
            self._opened_at = 0.0


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions API.

    The default configuration targets **qwen3-vl:4b** running on a
    local Ollama server.  Qwen3-VL is a *native* multimodal
    model that supports text, image, and video inputs in a
    single model -- no separate transcription step needed.

    Skills / Capabilities:
      - Send text and multimodal (vision / audio / video) chat requests
      - Support for system / user / assistant message roles
      - JSON-mode structured output extraction
      - Configurable model, temperature, and max tokens
      - Build multimodal messages with image, audio and video content
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        timeout: float = 120.0,
        max_retries: int = 2,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self._is_qwen3 = "qwen3" in model.lower()
        self.token_tracker = TokenTracker()
        self.circuit_breaker = CircuitBreaker()

        resolved_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        resolved_url = base_url or _DEFAULT_BASE_URL

        # Pooled HTTP client — reuses TCP connections across threads
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        self._client = openai.OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            http_client=http_client,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant's text.

        Parameters
        ----------
        messages : list of dicts with ``role`` and ``content`` keys.
        temperature : override default temperature for this call.
        max_tokens : override default max_tokens for this call.
        json_mode : if True, request JSON response format.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            # Pass num_ctx to Ollama to limit KV cache allocation
            "extra_body": {"options": {"num_ctx": self.num_ctx}},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Circuit breaker check
        if not self.circuit_breaker.allow_request():
            raise RuntimeError(
                "LLM API circuit breaker is OPEN — too many consecutive "
                "failures. Try again later."
            )

        # Retry up to 2 times on empty responses
        last_usage = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception:
                self.circuit_breaker.record_failure()
                raise
            self.circuit_breaker.record_success()
            last_usage = getattr(response, "usage", None)
            choice = response.choices[0]
            content = choice.message.content or ""
            # Strip Qwen3 <think>...</think> blocks — they waste output tokens
            content = _THINK_RE.sub("", content)
            self._last_finish_reason = getattr(choice, "finish_reason", None) or "stop"
            if self._last_finish_reason == "length":
                _logger.warning(
                    "LLM response truncated (finish_reason='length'). "
                    "Consider increasing max_tokens (current: %s).",
                    kwargs.get("max_tokens"),
                )
            text = content.strip()
            if text or attempt >= 2:
                # Record token usage only once — on the final accepted attempt
                self.token_tracker.record(last_usage)
                return text
            # Empty response — retry with slightly higher temperature
            kwargs["temperature"] = min(1.0, kwargs["temperature"] + 0.2)

        return ""  # unreachable but satisfies type checker

    @property
    def last_finish_reason(self) -> str:
        """Return the finish_reason from the most recent completion."""
        return getattr(self, "_last_finish_reason", "stop")

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Stream a chat completion, yielding text chunks as they arrive.

        After the generator is exhausted, ``last_finish_reason`` reflects
        the final finish reason from the stream.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "extra_body": {"options": {"num_ctx": self.num_ctx}},
        }
        stream = self._client.chat.completions.create(**kwargs)
        in_think = False
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                text = delta.content
                # Strip <think>…</think> incrementally
                if "<think>" in text:
                    in_think = True
                if in_think:
                    if "</think>" in text:
                        text = text.split("</think>", 1)[1]
                        in_think = False
                    else:
                        continue
                if text:
                    yield text
            finish = chunk.choices[0].finish_reason if chunk.choices else None
            # Extract token usage from the final chunk (OpenAI/Ollama compatible)
            if hasattr(chunk, "usage") and chunk.usage:
                self.token_tracker.record(chunk.usage)
            if finish:
                self._last_finish_reason = finish

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def ask(self, system: str, user: str, **kwargs: Any) -> str:
        """Simple system + user message call."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, **kwargs)

    def ask_json(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the LLM and parse the response as JSON.

        For Qwen3 models, prepends /no_think to skip the internal reasoning
        phase — JSON classification calls don't need chain-of-thought.

        Parameters
        ----------
        schema : optional JSON Schema dict to validate the result against.
            If the returned JSON does not conform, a single auto-repair
            attempt is made: missing required keys are filled with
            defaults and extra keys are stripped.
        """
        # Disable Qwen3 thinking mode for JSON calls — saves significant time
        if self._is_qwen3:
            user = "/no_think\n" + user
        raw = self.ask(system, user, json_mode=True, **kwargs)
        parsed = self._parse_json_response(raw)
        if schema and parsed:
            parsed = self._validate_and_repair(parsed, schema)
        return parsed

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Best-effort JSON extraction from raw LLM output."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            extracted = raw
            if "```json" in raw:
                extracted = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                extracted = raw.split("```")[1].split("```")[0]
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                _logger.warning(
                    "Failed to parse JSON from LLM response (len=%d): %.120s",
                    len(raw), raw,
                )
                return {}

    @staticmethod
    def _validate_and_repair(
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Lightweight JSON Schema validation + repair.

        - Fills missing required keys with type-appropriate defaults.
        - Coerces wrong-typed values when safe.
        - Strips unknown keys if ``additionalProperties`` is False.
        """
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        _TYPE_DEFAULTS: dict[str, Any] = {
            "string": "", "integer": 0, "number": 0.0,
            "boolean": False, "array": [], "object": {},
        }

        # Fill missing required keys
        for key in required:
            if key not in data:
                prop_type = props.get(key, {}).get("type", "string")
                data[key] = props.get(key, {}).get(
                    "default", _TYPE_DEFAULTS.get(prop_type, ""),
                )
                _logger.warning(
                    "JSON repair: filled missing required key '%s' with default %r",
                    key, data[key],
                )

        # Coerce types where safe
        for key, prop_schema in props.items():
            if key not in data:
                continue
            expected = prop_schema.get("type")
            val = data[key]
            if expected == "integer" and isinstance(val, str):
                try:
                    data[key] = int(val)
                except (ValueError, TypeError):
                    pass
            elif expected == "number" and isinstance(val, str):
                try:
                    data[key] = float(val)
                except (ValueError, TypeError):
                    pass
            elif expected == "boolean" and isinstance(val, str):
                data[key] = val.lower() in ("true", "1", "yes")

        # Strip unknown keys
        if schema.get("additionalProperties") is False:
            data = {k: v for k, v in data.items() if k in props}

        return data

    # ------------------------------------------------------------------
    # Multimodal message builders (Qwen3-VL compatible)
    # ------------------------------------------------------------------

    def build_vision_message(self, text: str, image_urls: list[str]) -> dict[str, Any]:
        """Construct a multimodal user message with text + images."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "auto"},
                }
            )
        return {"role": "user", "content": content}

    def build_audio_message(self, text: str, audio_urls: list[str]) -> dict[str, Any]:
        """Construct a multimodal user message with text + audio.

        Qwen3-VL supports ``input_audio`` content parts natively,
        so we can send audio directly without a separate transcription call.
        Local file paths are converted to base64 data URIs.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in audio_urls:
            resolved = self._resolve_audio_url(url)
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": resolved, "format": self._audio_format(url)},
                }
            )
        return {"role": "user", "content": content}

    def build_video_message(
        self, text: str, video_urls: list[str],
    ) -> dict[str, Any]:
        """Construct a multimodal user message with text + video.

        Qwen3-VL accepts ``video`` content parts directly.  For local
        files the caller should ensure the file is accessible; remote URLs
        are passed through.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in video_urls:
            content.append({"type": "video", "video": url})
        return {"role": "user", "content": content}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_audio_url(path_or_url: str) -> str:
        """Return a base64 string if *path_or_url* is a local file, else pass through."""
        if path_or_url.startswith(("http://", "https://", "data:")):
            return path_or_url
        raw = Path(path_or_url).read_bytes()
        return base64.b64encode(raw).decode()

    @staticmethod
    def _audio_format(path_or_url: str) -> str:
        """Guess the audio format from a file path / URL suffix."""
        suffix = Path(path_or_url).suffix.lower().lstrip(".")
        return {"wav": "wav", "mp3": "mp3", "flac": "flac", "ogg": "ogg"}.get(suffix, "wav")


# ======================================================================
# Async variant
# ======================================================================

class AsyncLLMClient:
    """Async wrapper around an OpenAI-compatible chat completions API.

    Provides the same interface as :class:`LLMClient` but uses
    ``openai.AsyncOpenAI`` and async httpx under the hood.  Suitable for
    ``asyncio.gather``-style parallel modality processing.

    Usage
    -----
    >>> client = AsyncLLMClient()
    >>> response = await client.chat([{"role": "user", "content": "hi"}])
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 16384,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        resolved_url = base_url or _DEFAULT_BASE_URL

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        self._client = openai.AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            http_client=http_client,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # Core async completion
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send an async chat completion request."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        content = _THINK_RE.sub("", content)
        return content.strip()

    async def ask(self, system: str, user: str, **kwargs: Any) -> str:
        """Async system + user message call."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return await self.chat(messages, **kwargs)

    async def ask_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
        """Async call returning parsed JSON."""
        raw = await self.ask(system, user, json_mode=True, **kwargs)
        return LLMClient._parse_json_response(raw)

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        await self._client.close()

    # Delegate multimodal builders to LLMClient (they're sync/pure)
    build_vision_message = LLMClient.build_vision_message
    build_audio_message = LLMClient.build_audio_message
    build_video_message = LLMClient.build_video_message
    _resolve_audio_url = LLMClient._resolve_audio_url
    _audio_format = LLMClient._audio_format


# ======================================================================
# Embedding client (vector search support)
# ======================================================================

_DEFAULT_EMBEDDING_MODEL = os.getenv("PINOCCHIO_EMBEDDING_MODEL", "nomic-embed-text")


class EmbeddingClient:
    """Generate text embeddings via an OpenAI-compatible embeddings API.

    Default backend: ``nomic-embed-text`` running on local Ollama server.
    """

    def __init__(
        self,
        model: str = _DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        resolved_key = api_key or os.getenv("OLLAMA_API_KEY", "ollama")
        resolved_url = base_url or _DEFAULT_BASE_URL

        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._client = openai.OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            http_client=http_client,
        )

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single text string."""
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to preserve input order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
