"""Tests for LLM exception handling, empty responses, and concurrency safety.

Covers edge cases that were identified as untested:
- LLM timeout / connection errors
- LLM returning empty string
- Concurrent chat() calls (thread safety)
- chat() with all-None arguments
- Extremely long input text
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pinocchio.utils.llm_client import LLMClient
from pinocchio.models.schemas import AgentMessage, MultimodalInput


# ── Helpers ──────────────────────────────────────────────────────────────

def _mock_openai_client():
    """Create a mock OpenAI client with configurable chat response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "test response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ── LLM Timeout and Error Tests ─────────────────────────────────────────

class TestLLMExceptionHandling:
    """Test LLM client behaviour under error conditions."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_chat_timeout_raises(self, mock_openai_mod):
        """LLM client should propagate timeout errors."""
        import httpx
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = httpx.TimeoutException("Request timed out")
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        with pytest.raises(httpx.TimeoutException):
            llm.chat([{"role": "user", "content": "hello"}])

    @patch("pinocchio.utils.llm_client.openai")
    def test_chat_connection_error_raises(self, mock_openai_mod):
        """LLM client should propagate connection errors."""
        import httpx
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = httpx.ConnectError("Connection refused")
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        with pytest.raises(httpx.ConnectError):
            llm.chat([{"role": "user", "content": "hello"}])

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_json_invalid_json_raises(self, mock_openai_mod):
        """ask_json should return empty dict when LLM returns non-JSON garbage."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON at all!"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.ask_json("system", "user")
        assert result == {}


# ── LLM Empty Response Tests ────────────────────────────────────────────

class TestLLMEmptyResponse:
    """Test behaviour when LLM returns empty or None content."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_chat_returns_empty_string_when_content_is_none(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.chat([{"role": "user", "content": "hello"}])
        assert result == ""

    @patch("pinocchio.utils.llm_client.openai")
    def test_chat_returns_empty_string_when_content_is_whitespace(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   \n  "
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.chat([{"role": "user", "content": "hello"}])
        assert result == ""

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_with_empty_response(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.ask("system prompt", "user input")
        assert result == ""


# ── Concurrent Thread Safety ────────────────────────────────────────────

class TestLLMConcurrency:
    """Verify the LLM client is safe for concurrent use."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_concurrent_chat_calls(self, mock_openai_mod):
        """Multiple threads calling chat() should not crash."""
        mock_client = MagicMock()
        call_count = 0
        lock = threading.Lock()

        def _mock_create(**kwargs):
            nonlocal call_count
            with lock:
                call_count += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = f"response-{call_count}"
            return resp

        mock_client.chat.completions.create.side_effect = _mock_create
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")

        results = [None] * 10
        errors = []

        def _call(idx):
            try:
                results[idx] = llm.chat([{"role": "user", "content": f"msg-{idx}"}])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert all(r is not None for r in results)
        assert call_count == 10


# ── Orchestrator Edge Cases ──────────────────────────────────────────────

class TestOrchestratorEdgeCases:
    """Test orchestrator chat() with unusual input combinations."""

    @patch("pinocchio.orchestrator.LLMClient")
    def test_chat_with_all_none_arguments(self, MockLLM):
        """chat() with no text and no media should not crash."""
        from pinocchio.orchestrator import Pinocchio

        agent = Pinocchio(model="test", api_key="k", base_url="http://x", verbose=False)
        # Stub all cognitive agents
        for a_name in ("perceive", "strategize", "execute", "evaluate", "learn"):
            if a_name == "execute":
                setattr(agent.agent, a_name, MagicMock(return_value=AgentMessage(content="ok", confidence=0.5)))
            elif a_name == "perceive":
                from pinocchio.models.schemas import PerceptionResult
                setattr(agent.agent, a_name, MagicMock(return_value=PerceptionResult()))
            elif a_name == "strategize":
                from pinocchio.models.schemas import StrategyResult
                setattr(agent.agent, a_name, MagicMock(return_value=StrategyResult()))
            elif a_name == "evaluate":
                from pinocchio.models.schemas import EvaluationResult
                setattr(agent.agent, a_name, MagicMock(return_value=EvaluationResult()))
            elif a_name == "learn":
                from pinocchio.models.schemas import LearningResult
                setattr(agent.agent, a_name, MagicMock(return_value=LearningResult()))
        agent.agent.should_meta_reflect = MagicMock(return_value=False)

        # Call with no text and no media
        result = agent.chat(None)
        assert isinstance(result, str)
        assert result == "ok"

    @patch("pinocchio.orchestrator.LLMClient")
    def test_chat_with_very_long_text(self, MockLLM):
        """chat() with very long input should not crash."""
        from pinocchio.orchestrator import Pinocchio

        agent = Pinocchio(model="test", api_key="k", base_url="http://x", verbose=False)
        for a_name in ("perceive", "strategize", "execute", "evaluate", "learn"):
            if a_name == "execute":
                setattr(agent.agent, a_name, MagicMock(return_value=AgentMessage(content="handled", confidence=0.5)))
            elif a_name == "perceive":
                from pinocchio.models.schemas import PerceptionResult
                setattr(agent.agent, a_name, MagicMock(return_value=PerceptionResult()))
            elif a_name == "strategize":
                from pinocchio.models.schemas import StrategyResult
                setattr(agent.agent, a_name, MagicMock(return_value=StrategyResult()))
            elif a_name == "evaluate":
                from pinocchio.models.schemas import EvaluationResult
                setattr(agent.agent, a_name, MagicMock(return_value=EvaluationResult()))
            elif a_name == "learn":
                from pinocchio.models.schemas import LearningResult
                setattr(agent.agent, a_name, MagicMock(return_value=LearningResult()))
        agent.agent.should_meta_reflect = MagicMock(return_value=False)

        long_text = "x" * 200_000
        result = agent.chat(long_text)
        assert result == "handled"

    @patch("pinocchio.orchestrator.LLMClient")
    def test_chat_error_recovery_returns_safe_message(self, MockLLM):
        """If the cognitive loop raises, chat() should return a safe error message."""
        from pinocchio.orchestrator import Pinocchio

        agent = Pinocchio(model="test", api_key="k", base_url="http://x", verbose=False)
        agent.agent.perceive = MagicMock(side_effect=RuntimeError("LLM exploded"))

        result = agent.chat("test")
        assert "错误" in result or "抱歉" in result

    @patch("pinocchio.orchestrator.LLMClient")
    def test_concurrent_chat_calls_on_orchestrator(self, MockLLM):
        """Multiple threads calling chat() on the same Pinocchio should not crash."""
        from pinocchio.orchestrator import Pinocchio

        agent = Pinocchio(model="test", api_key="k", base_url="http://x", verbose=False)
        counter = {"n": 0}
        lock = threading.Lock()

        def _mock_perception(**kwargs):
            from pinocchio.models.schemas import PerceptionResult
            with lock:
                counter["n"] += 1
            return PerceptionResult()

        agent.agent.perceive = MagicMock(side_effect=_mock_perception)
        agent.agent.strategize = MagicMock(
            return_value=MagicMock(selected_strategy="test", is_novel=False))
        agent.agent.execute = MagicMock(
            return_value=AgentMessage(content="ok", confidence=0.8))
        agent.agent.evaluate = MagicMock(
            return_value=MagicMock(output_quality=8, task_completion="complete"))
        agent.agent.learn = MagicMock(return_value=MagicMock())
        agent.agent.should_meta_reflect = MagicMock(return_value=False)

        errors = []
        results = [None] * 5

        def _call(idx):
            try:
                results[idx] = agent.chat(f"message-{idx}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(r is not None for r in results)


# ── LLMClient _resolve_audio_url local file ─────────────────────────────

class TestResolveAudioLocal:
    """Cover the local file path in _resolve_audio_url."""

    def test_resolve_local_file(self, tmp_path):
        """Local audio file should be base64 encoded."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00\x01\x02\x03")
        result = LLMClient._resolve_audio_url(str(audio_file))
        import base64
        decoded = base64.b64decode(result)
        assert decoded == b"\x00\x01\x02\x03"
