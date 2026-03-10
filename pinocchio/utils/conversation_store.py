"""Conversation Store — SQLite-backed session and message persistence.

Stores chat sessions and their messages in a local SQLite database
so conversations survive server restarts and browser refreshes.

Tables
------
``sessions``
    id (TEXT PK), title, created_at, updated_at
``messages``
    id (INTEGER PK), session_id (FK), role, content, attachments (JSON),
    created_at

Thread Safety
-------------
All write operations are serialised via a ``threading.Lock``.  The
database uses WAL journal mode for concurrent read performance.

Usage
-----
>>> store = ConversationStore("data/conversations.db")
>>> s = store.create_session("量子力学讨论")
>>> mid = store.add_message(s.id, "user", "什么是量子纠缠？")
>>> msgs = store.get_messages(s.id)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class Session:
    """A conversation session."""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }


@dataclass
class StoredMessage:
    """A persisted chat message."""

    id: int
    session_id: str
    role: str
    content: str
    attachments: str  # JSON string
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "attachments": json.loads(self.attachments) if self.attachments else [],
            "created_at": self.created_at,
        }


class ConversationStore:
    """SQLite-backed conversation persistence.

    Provides CRUD operations for chat sessions and their messages.
    All writes are serialised by an internal lock; reads use WAL mode
    for non-blocking concurrency.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '新对话',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                        content TEXT NOT NULL,
                        attachments TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id)
                            REFERENCES sessions(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_session
                        ON messages(session_id);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, title: str = "新对话") -> Session:
        """Create a new conversation session."""
        sid = uuid.uuid4().hex[:12]
        now = self._now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (sid, title, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        return Session(id=sid, title=title, created_at=now, updated_at=now)

    def list_sessions(self, *, limit: int = 50) -> list[Session]:
        """Return sessions ordered by most recently updated.

        Returns at most *limit* sessions.  Empty sessions are excluded
        unless they are the only ones (so the UI always has at least one).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                HAVING message_count > 0
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            # If no non-empty sessions, return the most recent session
            if not rows:
                rows = conn.execute(
                    """
                    SELECT s.id, s.title, s.created_at, s.updated_at,
                           0 AS message_count
                    FROM sessions s
                    ORDER BY s.updated_at DESC
                    LIMIT 1
                    """
                ).fetchall()
            return [
                Session(
                    id=r["id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a single session by ID."""
        conn = self._connect()
        try:
            r = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.id = ?
                GROUP BY s.id
                """,
                (session_id,),
            ).fetchone()
            if r is None:
                return None
            return Session(
                id=r["id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )
        finally:
            conn.close()

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Rename a session.  Returns True if the session existed."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, self._now(), session_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (CASCADE)."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM sessions WHERE id = ?", (session_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> int:
        """Append a message to a session.  Returns the new message ID."""
        now = self._now()
        att_json = json.dumps(attachments or [], ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO messages "
                    "(session_id, role, content, attachments, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, att_json, now),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )
                conn.commit()
                return cur.lastrowid  # type: ignore[return-value]
            finally:
                conn.close()

    def get_messages(self, session_id: str) -> list[StoredMessage]:
        """Return all messages in a session, ordered by creation."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [
                StoredMessage(
                    id=r["id"],
                    session_id=r["session_id"],
                    role=r["role"],
                    content=r["content"],
                    attachments=r["attachments"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_message(self, message_id: int) -> StoredMessage | None:
        """Retrieve a single message by ID."""
        conn = self._connect()
        try:
            r = conn.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            if r is None:
                return None
            return StoredMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                attachments=r["attachments"],
                created_at=r["created_at"],
            )
        finally:
            conn.close()

    def update_message(self, message_id: int, content: str) -> bool:
        """Update the content of a message.  Returns True if it existed."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE messages SET content = ? WHERE id = ?",
                    (content, message_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def delete_messages_after(self, session_id: str, message_id: int) -> int:
        """Delete all messages in a session with id > *message_id*.

        Returns the number of rows deleted.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM messages WHERE session_id = ? AND id > ?",
                    (session_id, message_id),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def purge_empty_sessions(self, keep_id: str | None = None) -> int:
        """Delete sessions that have zero messages.

        The session identified by *keep_id* is never deleted (it is the
        active session).  Returns the number of sessions removed.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    DELETE FROM sessions
                    WHERE id != ?
                      AND id NOT IN (
                          SELECT DISTINCT session_id FROM messages
                      )
                    """,
                    (keep_id or "",),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
