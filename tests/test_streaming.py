"""Tests for LLMClient streaming (chat_stream)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLLMClientStreaming:
    """Tests for LLMClient.chat_stream()."""

    def test_chat_stream_yields_chunks(self):
        from pinocchio.utils.llm_client import LLMClient, TokenTracker

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.7
        client.max_tokens = 100
        client.num_ctx = 2048
        client._is_qwen3 = False
        client.token_tracker = TokenTracker()

        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)]
        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [MagicMock(delta=MagicMock(content=" world"), finish_reason=None)]
        mock_chunk_3 = MagicMock()
        mock_chunk_3.choices = [MagicMock(delta=MagicMock(content=None), finish_reason="stop")]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk_1, mock_chunk_2, mock_chunk_3])
        client._client = mock_client

        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello", " world"]
        assert client.last_finish_reason == "stop"

    def test_chat_stream_strips_think_tags(self):
        from pinocchio.utils.llm_client import LLMClient, TokenTracker

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.7
        client.max_tokens = 100
        client.num_ctx = 2048
        client._is_qwen3 = True
        client.token_tracker = TokenTracker()

        chunks_data = [
            ("some ", None),
            ("<think>", None),
            ("internal thought", None),
            ("</think>answer", None),
            (None, "stop"),
        ]
        mock_chunks = []
        for content, finish in chunks_data:
            c = MagicMock()
            c.choices = [MagicMock(delta=MagicMock(content=content), finish_reason=finish)]
            mock_chunks.append(c)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        client._client = mock_client

        result = list(client.chat_stream([{"role": "user", "content": "test"}]))
        assert "some " in result
        assert "answer" in result
        full = "".join(result)
        assert "internal thought" not in full


# =====================================================================
# Streaming token tracking
# =====================================================================


class TestStreamingTokenTracking:
    """Streaming responses should track token usage from final chunk."""

    def test_stream_records_final_chunk_usage(self):
        from pinocchio.utils.llm_client import LLMClient, TokenTracker

        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(model="test", api_key="k", base_url="http://x")

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)]
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"), finish_reason="stop")]
        chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([chunk1, chunk2]))
        client._client.chat.completions.create = MagicMock(return_value=mock_stream)

        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert client.token_tracker.total_tokens == 15
        assert client.token_tracker.call_count == 1

    def test_stream_no_usage_graceful(self):
        from pinocchio.utils.llm_client import LLMClient, TokenTracker

        with patch("pinocchio.utils.llm_client.openai"):
            client = LLMClient(model="test", api_key="k", base_url="http://x")

        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="Hi"), finish_reason="stop")]
        del chunk.usage

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([chunk]))
        client._client.chat.completions.create = MagicMock(return_value=mock_stream)

        list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert client.token_tracker.total_tokens == 0


# =====================================================================
# chat_stream thread safety
# =====================================================================


class TestChatStreamThreadSafety:
    """chat_stream() should isolate state within the background thread."""

    def test_restore_happens_inside_thread(self):
        """Originals should be restored inside the background thread."""
        import inspect
        from pinocchio.orchestrator import Pinocchio
        src = inspect.getsource(Pinocchio.chat_stream)
        assert "finally:" in src
        assert "# Restore original state" not in src

    def test_stream_phase_events_still_work(self):
        """Complex input streaming should still yield [PHASE] events."""
        with patch("pinocchio.utils.llm_client.openai"):
            agent = __import__("pinocchio").Pinocchio(
                model="test", api_key="k", base_url="http://x", verbose=False,
            )

        def mock_chat(*args, **kwargs):
            agent._emit_progress("perceive", "running", "test")
            agent._emit_progress("perceive", "done")
            return "final response"

        agent.chat = mock_chat
        chunks = list(agent.chat_stream("Complex input", image_paths=["fake.jpg"]))
        phase_events = [c for c in chunks if "[PHASE]" in c]
        assert len(phase_events) >= 1
        assert chunks[-1] == "final response"


# =====================================================================
# chat_stream real-time streaming
# =====================================================================


class TestChatStreamRealtime:
    """chat_stream should provide real-time events for complex inputs."""

    def test_stream_yields_phase_events(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = __import__("pinocchio").Pinocchio(
                model="test", api_key="k", base_url="http://x", verbose=False,
            )

        def mock_chat(*args, **kwargs):
            agent._emit_progress("perceive", "running", "test")
            agent._emit_progress("perceive", "done")
            return "final response"

        agent.chat = mock_chat
        chunks = list(agent.chat_stream("Complex input", image_paths=["fake.jpg"]))
        phase_events = [c for c in chunks if "[PHASE]" in c]
        assert len(phase_events) >= 1
        assert chunks[-1] == "final response"

    def test_stream_simple_input_direct(self):
        """Simple text input should stream directly via fast path."""
        with patch("pinocchio.utils.llm_client.openai"):
            agent = __import__("pinocchio").Pinocchio(
                model="test", api_key="k", base_url="http://x", verbose=False,
            )

        from pinocchio.orchestrator import Pinocchio
        original = Pinocchio.FAST_PATH_MAX_LENGTH
        Pinocchio.FAST_PATH_MAX_LENGTH = 500
        try:
            agent.llm.chat_stream = MagicMock(return_value=iter(["Hello", " world"]))
            agent._context_manager.build_messages = MagicMock(
                return_value=[{"role": "system", "content": "You are helpful."}]
            )
            chunks = list(agent.chat_stream("hi"))
            assert "Hello" in chunks
            assert " world" in chunks
        finally:
            Pinocchio.FAST_PATH_MAX_LENGTH = original
