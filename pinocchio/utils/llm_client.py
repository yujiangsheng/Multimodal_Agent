"""LLM Client -- abstraction for calling language model APIs.

Supports OpenAI-compatible APIs (Ollama, DashScope / Qwen, OpenAI, Azure,
vLLM, etc.) with structured retry, token counting, and multimodal
message construction.

Default backend: Qwen2.5-Omni via local Ollama server.

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
import os
from pathlib import Path
from typing import Any

import httpx
import openai

_DEFAULT_MODEL = os.getenv("PINOCCHIO_MODEL", "qwen2.5-omni")
_DEFAULT_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "http://localhost:11434/v1",
)


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions API.

    The default configuration targets **qwen2.5-omni** running on a
    local Ollama server.  Qwen2.5-Omni is a *native* omni-modal
    model that supports text, image, audio, and video inputs in a
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
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()

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

    def ask_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
        """Call the LLM and parse the response as JSON."""
        raw = self.ask(system, user, json_mode=True, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw)

    # ------------------------------------------------------------------
    # Multimodal message builders (Qwen2.5-Omni compatible)
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

        Qwen2.5-Omni supports ``input_audio`` content parts natively,
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

        Qwen2.5-Omni accepts ``video`` content parts directly.  For local
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
        max_tokens: int = 4096,
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
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw)

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        await self._client.close()

    # Delegate multimodal builders to LLMClient (they're sync/pure)
    build_vision_message = LLMClient.build_vision_message
    build_audio_message = LLMClient.build_audio_message
    build_video_message = LLMClient.build_video_message
    _resolve_audio_url = LLMClient._resolve_audio_url
    _audio_format = LLMClient._audio_format
