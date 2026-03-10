"""Tests for LLMClient streaming (chat_stream)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLLMClientStreaming:
    """Tests for LLMClient.chat_stream()."""

    def test_chat_stream_yields_chunks(self):
        from pinocchio.utils.llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.7
        client.max_tokens = 100
        client.num_ctx = 2048
        client._is_qwen3 = False

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
        from pinocchio.utils.llm_client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.model = "test"
        client.temperature = 0.7
        client.max_tokens = 100
        client.num_ctx = 2048
        client._is_qwen3 = True

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
