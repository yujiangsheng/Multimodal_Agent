"""Tests for async LLM client and async orchestrator chat.

Validates the new ``AsyncLLMClient`` and ``Pinocchio.async_chat()`` features.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pinocchio.utils.llm_client import AsyncLLMClient, LLMClient
from pinocchio.models.schemas import AgentMessage


# ── AsyncLLMClient Tests ─────────────────────────────────────────────────

class TestAsyncLLMClient:
    """Test the async LLM client."""

    @pytest.mark.asyncio
    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_chat_basic(self, mock_openai_mod):
        mock_async_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "async response"
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_async_client

        client = AsyncLLMClient(model="test", api_key="k", base_url="http://x")
        result = await client.chat([{"role": "user", "content": "hi"}])
        assert result == "async response"

    @pytest.mark.asyncio
    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_ask(self, mock_openai_mod):
        mock_async_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "async ask response"
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_async_client

        client = AsyncLLMClient(model="test", api_key="k", base_url="http://x")
        result = await client.ask("sys", "user msg")
        assert result == "async ask response"

    @pytest.mark.asyncio
    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_ask_json(self, mock_openai_mod):
        mock_async_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_async_client

        client = AsyncLLMClient(model="test", api_key="k", base_url="http://x")
        result = await client.ask_json("sys", "user msg")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_ask_json_markdown_fence(self, mock_openai_mod):
        mock_async_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"a": 1}\n```'
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_async_client

        client = AsyncLLMClient(model="test", api_key="k", base_url="http://x")
        result = await client.ask_json("sys", "user msg")
        assert result == {"a": 1}

    @pytest.mark.asyncio
    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_chat_none_content(self, mock_openai_mod):
        mock_async_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_async_client

        client = AsyncLLMClient(model="test", api_key="k", base_url="http://x")
        result = await client.chat([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_async_client_delegates_message_builders(self):
        """Multimodal message builders should be shared with sync client."""
        assert AsyncLLMClient.build_vision_message is LLMClient.build_vision_message
        assert AsyncLLMClient.build_audio_message is LLMClient.build_audio_message
        assert AsyncLLMClient.build_video_message is LLMClient.build_video_message


# ── Orchestrator async_chat Tests ────────────────────────────────────────

class TestOrchestratorAsyncChat:
    """Test the Pinocchio.async_chat() method."""

    @pytest.mark.asyncio
    @patch("pinocchio.orchestrator.LLMClient")
    async def test_async_chat_returns_string(self, MockLLM):
        from pinocchio.orchestrator import Pinocchio

        agent = Pinocchio(model="test", api_key="k", base_url="http://x", verbose=False)

        # Stub all cognitive agents
        from pinocchio.models.schemas import (
            PerceptionResult, StrategyResult, EvaluationResult, LearningResult,
        )
        agent.perception.run = MagicMock(return_value=PerceptionResult())
        agent.strategy.run = MagicMock(return_value=StrategyResult())
        agent.execution.run = MagicMock(
            return_value=AgentMessage(content="async result", confidence=0.8))
        agent.evaluation.run = MagicMock(return_value=EvaluationResult())
        agent.learning.run = MagicMock(return_value=LearningResult())
        agent.meta_reflection.should_trigger = MagicMock(return_value=False)

        result = await agent.async_chat("hello async")
        assert result == "async result"

    @pytest.mark.asyncio
    @patch("pinocchio.orchestrator.LLMClient")
    async def test_async_chat_concurrent_gather(self, MockLLM):
        """Multiple async_chat calls via gather should work."""
        from pinocchio.orchestrator import Pinocchio
        from pinocchio.models.schemas import (
            PerceptionResult, StrategyResult, EvaluationResult, LearningResult,
        )

        agent = Pinocchio(model="test", api_key="k", base_url="http://x", verbose=False)
        agent.perception.run = MagicMock(return_value=PerceptionResult())
        agent.strategy.run = MagicMock(return_value=StrategyResult())
        agent.execution.run = MagicMock(
            return_value=AgentMessage(content="ok", confidence=0.8))
        agent.evaluation.run = MagicMock(return_value=EvaluationResult())
        agent.learning.run = MagicMock(return_value=LearningResult())
        agent.meta_reflection.should_trigger = MagicMock(return_value=False)

        results = await asyncio.gather(
            agent.async_chat("msg1"),
            agent.async_chat("msg2"),
            agent.async_chat("msg3"),
        )
        assert len(results) == 3
        assert all(r == "ok" for r in results)
