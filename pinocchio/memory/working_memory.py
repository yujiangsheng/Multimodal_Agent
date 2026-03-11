"""Working Memory — volatile, session-scoped context buffer.

Working memory holds information relevant to the *current* session:
conversation history, active hypotheses, in-progress reasoning state,
and recently accessed long-term memories.  It is purely in-RAM and
is cleared when the session resets.

This is the temporal-axis "working" tier that complements the content-axis
stores (episodic, semantic, procedural).

Design rationale
────────────────
Human working memory has limited capacity (~7±2 items).  We mirror this
with a configurable ``capacity`` that controls how many conversation turns
and context items are retained before the oldest entries are evicted.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class WorkingMemoryItem:
    """A single item held in working memory."""

    item_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str = ""          # "conversation" | "hypothesis" | "context" | "recall"
    content: str = ""
    source: str = ""            # which agent/phase created this
    relevance: float = 1.0      # 0-1, decays over time
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "content": self.content,
            "source": self.source,
            "relevance": self.relevance,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """Volatile, capacity-limited session memory.

    Skills / Capabilities:
      - Store conversation turns and active context items
      - Capacity-limited: oldest items evicted when full (FIFO within category)
      - Retrieve items by category or keyword
      - Provide a formatted context string for downstream agents
      - Track active hypotheses and reasoning state
      - Relevance decay: items lose relevance over turns
      - Fully in-RAM; no disk persistence (session-scoped)
    """

    DEFAULT_CAPACITY = 50  # max items across all categories

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._items: deque[WorkingMemoryItem] = deque(maxlen=capacity)
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Capacity & state
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def turn_count(self) -> int:
        return self._turn_count

    # ------------------------------------------------------------------
    # Add / remove
    # ------------------------------------------------------------------

    def add(self, item: WorkingMemoryItem) -> None:
        """Add an item (oldest evicted if at capacity)."""
        self._items.append(item)

    def add_conversation_turn(
        self, role: str, content: str, **metadata: Any
    ) -> WorkingMemoryItem:
        """Convenience: add a conversation turn."""
        item = WorkingMemoryItem(
            category="conversation",
            content=content,
            source=role,
            metadata=metadata,
        )
        self.add(item)
        self._turn_count += 1
        self._decay_relevance()
        return item

    def add_hypothesis(self, hypothesis: str, source: str = "strategy") -> WorkingMemoryItem:
        """Store an active hypothesis or reasoning fragment."""
        item = WorkingMemoryItem(
            category="hypothesis",
            content=hypothesis,
            source=source,
        )
        self.add(item)
        return item

    def add_context(self, context: str, source: str = "recall") -> WorkingMemoryItem:
        """Store a recalled context item (e.g. from long-term memory)."""
        item = WorkingMemoryItem(
            category="context",
            content=context,
            source=source,
            relevance=0.8,
        )
        self.add(item)
        return item

    def clear(self) -> None:
        """Clear all working memory (session reset)."""
        self._items.clear()
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def all_items(self) -> list[WorkingMemoryItem]:
        return list(self._items)

    def get_by_category(self, category: str) -> list[WorkingMemoryItem]:
        return [it for it in self._items if it.category == category]

    def get_conversation(self) -> list[WorkingMemoryItem]:
        """Return conversation turns in chronological order."""
        return self.get_by_category("conversation")

    def get_hypotheses(self) -> list[WorkingMemoryItem]:
        return self.get_by_category("hypothesis")

    def search(self, keyword: str, limit: int = 10) -> list[WorkingMemoryItem]:
        """Simple keyword search across all items."""
        kw = keyword.lower()
        return [it for it in self._items if kw in it.content.lower()][:limit]

    def get_relevant(self, threshold: float = 0.3) -> list[WorkingMemoryItem]:
        """Return items above a relevance threshold, sorted by relevance."""
        items = [it for it in self._items if it.relevance >= threshold]
        items.sort(key=lambda x: x.relevance, reverse=True)
        return items

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_conversation_context(self, max_turns: int = 10) -> str:
        """Format recent conversation turns as a string for LLM context."""
        turns = self.get_conversation()[-max_turns:]
        if not turns:
            return ""
        lines = []
        for t in turns:
            role_label = "User" if t.source == "user" else "Assistant"
            lines.append(f"{role_label}: {t.content}")
        return "\n".join(lines)

    def format_active_context(self) -> str:
        """Format all non-conversation context as a string."""
        items = [it for it in self._items if it.category != "conversation"]
        if not items:
            return ""
        lines = []
        for it in items:
            lines.append(f"[{it.category}/{it.source}] {it.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Relevance decay
    # ------------------------------------------------------------------

    def _decay_relevance(self, decay_rate: float = 0.02) -> None:
        """Reduce relevance of older items after each turn.

        Uses a gentler default decay rate (0.02 vs original 0.05) so
        important context survives longer.  Conversation items are not
        decayed — their ordering already captures recency.
        """
        for item in self._items:
            if item.category != "conversation":
                item.relevance = max(0.0, item.relevance - decay_rate)

    # ------------------------------------------------------------------
    # Summary (for status/debug)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary of working memory state."""
        categories: dict[str, int] = {}
        for it in self._items:
            categories[it.category] = categories.get(it.category, 0) + 1
        return {
            "total_items": self.count,
            "capacity": self._capacity,
            "turn_count": self._turn_count,
            "categories": categories,
        }
