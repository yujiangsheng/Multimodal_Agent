"""Tests for ConversationStore — SQLite-backed session & message persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pinocchio.utils.conversation_store import ConversationStore, Session, StoredMessage


@pytest.fixture()
def store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(tmp_path / "test.db")


# ── Session CRUD ─────────────────────────────────────────

class TestSessionCRUD:
    def test_create_session_default_title(self, store: ConversationStore):
        s = store.create_session()
        assert isinstance(s, Session)
        assert s.title == "新对话"
        assert s.message_count == 0
        assert len(s.id) == 12

    def test_create_session_custom_title(self, store: ConversationStore):
        s = store.create_session("量子物理讨论")
        assert s.title == "量子物理讨论"

    def test_list_sessions_ordered_by_updated(self, store: ConversationStore):
        s1 = store.create_session("First")
        store.add_message(s1.id, "user", "hello")
        s2 = store.create_session("Second")
        store.add_message(s2.id, "user", "world")
        sessions = store.list_sessions()
        assert len(sessions) == 2
        # Most recently created should be first
        assert sessions[0].id == s2.id

    def test_get_session(self, store: ConversationStore):
        s = store.create_session("Test")
        found = store.get_session(s.id)
        assert found is not None
        assert found.title == "Test"

    def test_get_session_not_found(self, store: ConversationStore):
        assert store.get_session("nonexistent") is None

    def test_update_session_title(self, store: ConversationStore):
        s = store.create_session("Old")
        ok = store.update_session_title(s.id, "New")
        assert ok is True
        found = store.get_session(s.id)
        assert found.title == "New"

    def test_update_nonexistent_session(self, store: ConversationStore):
        ok = store.update_session_title("nope", "New")
        assert ok is False

    def test_delete_session(self, store: ConversationStore):
        s = store.create_session()
        ok = store.delete_session(s.id)
        assert ok is True
        assert store.get_session(s.id) is None

    def test_delete_nonexistent(self, store: ConversationStore):
        ok = store.delete_session("nope")
        assert ok is False

    def test_delete_cascades_messages(self, store: ConversationStore):
        s = store.create_session()
        store.add_message(s.id, "user", "hello")
        store.add_message(s.id, "assistant", "hi")
        store.delete_session(s.id)
        assert store.get_messages(s.id) == []

    def test_session_to_dict(self, store: ConversationStore):
        s = store.create_session("Test")
        d = s.to_dict()
        assert d["id"] == s.id
        assert d["title"] == "Test"
        assert "created_at" in d
        assert "updated_at" in d
        assert d["message_count"] == 0


# ── Message CRUD ─────────────────────────────────────────

class TestMessageCRUD:
    def test_add_and_get_messages(self, store: ConversationStore):
        s = store.create_session()
        mid1 = store.add_message(s.id, "user", "Hello")
        mid2 = store.add_message(s.id, "assistant", "Hi there!")
        msgs = store.get_messages(s.id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hello"
        assert msgs[0].id == mid1
        assert msgs[1].role == "assistant"
        assert msgs[1].content == "Hi there!"
        assert msgs[1].id == mid2

    def test_add_message_returns_id(self, store: ConversationStore):
        s = store.create_session()
        mid = store.add_message(s.id, "user", "Test")
        assert isinstance(mid, int)
        assert mid > 0

    def test_add_message_with_attachments(self, store: ConversationStore):
        s = store.create_session()
        atts = [{"name": "photo.jpg", "type": "image"}]
        store.add_message(s.id, "user", "Look at this", atts)
        msgs = store.get_messages(s.id)
        msg_dict = msgs[0].to_dict()
        assert msg_dict["attachments"] == atts

    def test_get_message_by_id(self, store: ConversationStore):
        s = store.create_session()
        mid = store.add_message(s.id, "user", "Target")
        msg = store.get_message(mid)
        assert msg is not None
        assert msg.content == "Target"

    def test_get_message_not_found(self, store: ConversationStore):
        assert store.get_message(99999) is None

    def test_update_message(self, store: ConversationStore):
        s = store.create_session()
        mid = store.add_message(s.id, "user", "Old text")
        ok = store.update_message(mid, "New text")
        assert ok is True
        msg = store.get_message(mid)
        assert msg.content == "New text"

    def test_update_nonexistent_message(self, store: ConversationStore):
        ok = store.update_message(99999, "New")
        assert ok is False

    def test_delete_messages_after(self, store: ConversationStore):
        s = store.create_session()
        m1 = store.add_message(s.id, "user", "First")
        m2 = store.add_message(s.id, "assistant", "Second")
        m3 = store.add_message(s.id, "user", "Third")
        m4 = store.add_message(s.id, "assistant", "Fourth")
        deleted = store.delete_messages_after(s.id, m2)
        assert deleted == 2
        remaining = store.get_messages(s.id)
        assert len(remaining) == 2
        assert remaining[-1].id == m2

    def test_message_count_in_session(self, store: ConversationStore):
        s = store.create_session()
        store.add_message(s.id, "user", "A")
        store.add_message(s.id, "assistant", "B")
        found = store.get_session(s.id)
        assert found.message_count == 2

    def test_messages_ordered_by_id(self, store: ConversationStore):
        s = store.create_session()
        store.add_message(s.id, "user", "1")
        store.add_message(s.id, "assistant", "2")
        store.add_message(s.id, "user", "3")
        msgs = store.get_messages(s.id)
        assert [m.content for m in msgs] == ["1", "2", "3"]

    def test_stored_message_to_dict(self, store: ConversationStore):
        s = store.create_session()
        mid = store.add_message(s.id, "user", "Hello")
        msg = store.get_message(mid)
        d = msg.to_dict()
        assert d["id"] == mid
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert isinstance(d["attachments"], list)


# ── Isolation ────────────────────────────────────────────

class TestIsolation:
    def test_messages_scoped_to_session(self, store: ConversationStore):
        s1 = store.create_session()
        s2 = store.create_session()
        store.add_message(s1.id, "user", "In S1")
        store.add_message(s2.id, "user", "In S2")
        assert len(store.get_messages(s1.id)) == 1
        assert len(store.get_messages(s2.id)) == 1
        assert store.get_messages(s1.id)[0].content == "In S1"

    def test_multiple_stores_same_db(self, tmp_path: Path):
        db = tmp_path / "shared.db"
        store1 = ConversationStore(db)
        s = store1.create_session("Shared")
        store1.add_message(s.id, "user", "Visible")
        store2 = ConversationStore(db)
        msgs = store2.get_messages(s.id)
        assert len(msgs) == 1
        assert msgs[0].content == "Visible"
