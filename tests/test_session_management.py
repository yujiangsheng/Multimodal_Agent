"""Tests for orchestrator session management, regenerate, edit, and UserModel injection."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.models.schemas import AgentMessage, UserModel
from pinocchio.models.enums import ExpertiseLevel, CommunicationStyle


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture()
def agent(tmp_path):
    """Create a Pinocchio orchestrator with mocked LLM."""
    with patch("pinocchio.orchestrator.LLMClient") as MockLLM, \
         patch("pinocchio.orchestrator.EmbeddingClient"), \
         patch("pinocchio.orchestrator.ResourceMonitor") as MockRM:
        mock_llm = MockLLM.return_value
        mock_llm.model = "test-model"
        mock_llm.temperature = 0.7
        mock_llm.max_tokens = 4096
        mock_llm.ask_json.return_value = {}
        mock_llm.ask.return_value = "mock response"
        mock_llm.chat.return_value = "mock response"

        mock_rm = MockRM.return_value
        mock_snap = MagicMock()
        mock_snap.has_gpu = False
        mock_snap.cpu_count_physical = 4
        mock_snap.ram_total_mb = 16384
        mock_snap.total_vram_mb = 0
        mock_snap.gpus = []
        mock_snap.recommended_workers = 2
        mock_snap.to_dict.return_value = {}
        mock_rm.snapshot.return_value = mock_snap

        from pinocchio.orchestrator import Pinocchio
        p = Pinocchio(
            data_dir=str(tmp_path),
            verbose=False,
        )
        yield p


# ── Session management ───────────────────────────────────

class TestSessionManagement:
    def test_initial_session_created(self, agent):
        assert agent.current_session_id is not None
        sessions = agent.list_sessions()
        assert len(sessions) >= 1

    def test_new_session(self, agent):
        old_id = agent.current_session_id
        result = agent.new_session("Test session")
        assert result["title"] == "Test session"
        assert agent.current_session_id != old_id
        assert agent.conversation_history == []

    def test_list_sessions_after_creating(self, agent):
        agent.new_session("S1")
        # Add a message so S1 is not filtered as empty
        agent._conversation_store.add_message(agent.current_session_id, "user", "hi")
        agent.new_session("S2")
        agent._conversation_store.add_message(agent.current_session_id, "user", "hi")
        sessions = agent.list_sessions()
        titles = [s["title"] for s in sessions]
        assert "S1" in titles
        assert "S2" in titles

    def test_switch_session(self, agent):
        first_id = agent.current_session_id
        agent.new_session("Second")
        second_id = agent.current_session_id
        result = agent.switch_session(first_id)
        assert result is not None
        assert agent.current_session_id == first_id

    def test_switch_nonexistent_session(self, agent):
        result = agent.switch_session("nonexistent")
        assert result is None

    def test_delete_session(self, agent):
        s = agent.new_session("To delete")
        sid = s["id"]
        agent.new_session("Active")
        ok = agent.delete_session(sid)
        assert ok is True

    def test_cannot_delete_active_session(self, agent):
        ok = agent.delete_session(agent.current_session_id)
        assert ok is False

    def test_rename_session(self, agent):
        ok = agent.rename_session(agent.current_session_id, "Renamed")
        assert ok is True
        sessions = agent.list_sessions()
        current = next(s for s in sessions if s["id"] == agent.current_session_id)
        assert current["title"] == "Renamed"

    def test_get_session_messages_empty(self, agent):
        msgs = agent.get_session_messages(agent.current_session_id)
        assert msgs == []


# ── Chat persistence ─────────────────────────────────────

class TestChatPersistence:
    def test_chat_saves_messages(self, agent):
        agent.chat("Hello")
        msgs = agent.get_session_messages(agent.current_session_id)
        assert len(msgs) == 2  # user + assistant
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"

    def test_message_ids_in_history(self, agent):
        agent.chat("Test")
        assert "id" in agent.conversation_history[0]
        assert "id" in agent.conversation_history[1]
        assert isinstance(agent.conversation_history[0]["id"], int)

    def test_auto_title_from_first_message(self, agent):
        agent.chat("What is quantum entanglement?")
        sessions = agent.list_sessions()
        current = next(s for s in sessions if s["id"] == agent.current_session_id)
        assert "quantum" in current["title"].lower() or current["title"].startswith("What is")

    def test_switch_session_loads_history(self, agent):
        agent.chat("Message in session 1")
        first_id = agent.current_session_id
        agent.new_session("Session 2")
        agent.chat("Message in session 2")
        agent.switch_session(first_id)
        assert len(agent.conversation_history) == 2
        assert "session 1" in agent.conversation_history[0]["content"].lower()

    def test_last_message_ids_tracked(self, agent):
        agent.chat("Hello")
        assert agent._last_user_msg_id is not None
        assert agent._last_asst_msg_id is not None


# ── Reset ────────────────────────────────────────────────

class TestReset:
    def test_reset_creates_new_session(self, agent):
        old_id = agent.current_session_id
        agent.chat("Before reset")
        agent.reset()
        assert agent.current_session_id != old_id
        assert agent.conversation_history == []
        assert agent._interaction_count == 0


# ── Regenerate ───────────────────────────────────────────

class TestRegenerate:
    def test_regenerate_returns_new_response(self, agent):
        agent.chat("Hello")
        result = agent.regenerate()
        assert result is not None
        assert isinstance(result, str)

    def test_regenerate_replaces_last_assistant_msg(self, agent):
        agent.chat("Hello")
        old_msgs = len(agent.conversation_history)
        agent.regenerate()
        assert len(agent.conversation_history) == old_msgs

    def test_regenerate_empty_history(self, agent):
        result = agent.regenerate()
        assert result is None


# ── Edit and regenerate ──────────────────────────────────

class TestEditAndRegenerate:
    def test_edit_returns_response(self, agent):
        agent.chat("Hello")
        user_msg_id = agent.conversation_history[0]["id"]
        result = agent.edit_and_regenerate(user_msg_id, "Hi there")
        assert result is not None

    def test_edit_updates_message(self, agent):
        agent.chat("Hello")
        user_msg_id = agent.conversation_history[0]["id"]
        agent.edit_and_regenerate(user_msg_id, "Edited text")
        assert agent.conversation_history[0]["content"] == "Edited text"

    def test_edit_truncates_after(self, agent):
        agent.chat("First")
        agent.chat("Second")
        first_user_id = agent.conversation_history[0]["id"]
        agent.edit_and_regenerate(first_user_id, "Edited first")
        # Should have: edited user + new assistant = 2 messages
        assert len(agent.conversation_history) == 2

    def test_edit_nonexistent_message(self, agent):
        result = agent.edit_and_regenerate(99999, "Whatever")
        assert result is None


# ── UserModel context ────────────────────────────────────

class TestUserModelContext:
    def test_user_model_context_format(self, agent):
        ctx = agent._user_model_context()
        assert "用户水平" in ctx
        assert "沟通风格" in ctx
        assert "交互次数" in ctx

    def test_user_model_context_includes_domains(self, agent):
        agent.user_model.domains_of_interest = ["物理", "编程"]
        ctx = agent._user_model_context()
        assert "物理" in ctx
        assert "编程" in ctx

    def test_user_model_context_updates_with_interaction(self, agent):
        agent.user_model.interaction_count = 42
        ctx = agent._user_model_context()
        assert "42" in ctx

    def test_status_includes_session_id(self, agent):
        status = agent.status()
        assert "session_id" in status
        assert status["session_id"] == agent.current_session_id


# ========================================================================
# Round 6 — I15: switch_session acquires _lock
# ========================================================================


class TestSwitchSessionLocking:
    """switch_session() must acquire self._lock."""

    def test_switch_acquires_lock(self):
        """Verify switch_session() holds the lock during mutation."""
        from pinocchio.orchestrator import Pinocchio

        p = Pinocchio.__new__(Pinocchio)
        p._lock = threading.Lock()
        p.conversation_history = []
        p._current_session_id = "old"
        p._interaction_count = 0
        p._last_user_msg_id = None
        p._last_asst_msg_id = None

        p._conversation_store = MagicMock()
        p._conversation_store.get_session.return_value = MagicMock(
            id="new-session",
            to_dict=lambda: {"id": "new-session"},
        )
        p._conversation_store.get_messages.return_value = []

        p.user_model = MagicMock()
        p.memory = MagicMock()
        p._context_manager = MagicMock()
        p._response_cache = MagicMock()

        lock_was_held = []
        original_get_messages = p._conversation_store.get_messages

        def check_lock(*args, **kwargs):
            lock_was_held.append(p._lock.locked())
            return original_get_messages(*args, **kwargs)

        p._conversation_store.get_messages = check_lock

        result = p.switch_session("new-session")
        assert result == {"id": "new-session"}
        assert p._current_session_id == "new-session"
        assert lock_was_held == [True]
