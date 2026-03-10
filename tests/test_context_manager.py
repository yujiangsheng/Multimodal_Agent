"""Tests for ContextManager — context window management and auto-summarisation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinocchio.utils.context_manager import ContextManager, estimate_tokens, estimate_messages_tokens


# ======================================================================
# Token estimation
# ======================================================================


class TestTokenEstimation:
    """Tests for the token estimation heuristics."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        tokens = estimate_tokens("The quick brown fox jumps over the lazy dog")
        assert 5 < tokens < 20

    def test_chinese_text(self):
        tokens = estimate_tokens("这是一段中文测试文本")
        assert tokens > 0

    def test_long_text(self):
        tokens = estimate_tokens("hello world " * 100)
        assert tokens > 100

    def test_estimate_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        total = estimate_messages_tokens(msgs)
        assert total > 0


# ======================================================================
# ContextManager
# ======================================================================


class TestContextManager:
    """Tests for ContextManager."""

    def _make_llm(self, summary_response="Summary of earlier conversation."):
        llm = MagicMock()
        llm.ask.return_value = summary_response
        return llm

    def test_short_conversation_no_summary(self):
        llm = self._make_llm()
        cm = ContextManager(llm, max_context_tokens=10000)
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages = cm.build_messages("System prompt", conv)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert not cm.has_summary
        llm.ask.assert_not_called()

    def test_long_conversation_triggers_summary(self):
        llm = self._make_llm()
        cm = ContextManager(llm, max_context_tokens=200, keep_recent_turns=2)
        conv = [{"role": "user", "content": f"Message number {i} " * 20} for i in range(20)]
        messages = cm.build_messages("System prompt", conv)
        assert cm.has_summary
        assert cm.summarised_turn_count > 0
        llm.ask.assert_called_once()

    def test_reset_clears_summary(self):
        llm = self._make_llm()
        cm = ContextManager(llm, max_context_tokens=200, keep_recent_turns=2)
        conv = [{"role": "user", "content": f"Message {i} " * 20} for i in range(20)]
        cm.build_messages("System", conv)
        assert cm.has_summary
        cm.reset()
        assert not cm.has_summary
        assert cm.summarised_turn_count == 0

    def test_stats(self):
        llm = self._make_llm()
        cm = ContextManager(llm, max_context_tokens=5000)
        s = cm.stats()
        assert s["max_context_tokens"] == 5000
        assert s["has_summary"] is False
        assert "summarised_turns" in s

    def test_budget_enforcement(self):
        """Messages returned should not wildly exceed the token budget."""
        llm = self._make_llm()
        cm = ContextManager(llm, max_context_tokens=100, keep_recent_turns=2)
        conv = [{"role": "user", "content": "word " * 50} for _ in range(10)]
        messages = cm.build_messages("System", conv)
        # Should have dropped some turns
        assert len(messages) < 12  # system + summary + at most a few recent

    def test_summary_reused_across_builds(self):
        """Once summarised, subsequent builds should reuse the cached summary."""
        llm = self._make_llm()
        cm = ContextManager(llm, max_context_tokens=200, keep_recent_turns=2)
        conv = [{"role": "user", "content": f"Message {i} " * 20} for i in range(20)]
        cm.build_messages("System", conv)
        assert llm.ask.call_count == 1
        # Build again with same conversation — should reuse
        cm.build_messages("System", conv)
        assert llm.ask.call_count == 1  # not called again
