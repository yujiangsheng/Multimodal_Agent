"""Semantic Memory — distilled, generalizable knowledge.

Represents abstracted knowledge extracted from multiple episodic experiences:
domain heuristics, cross-modal reasoning patterns, user preference models,
and effective strategy templates.

Performance: maintains a domain index for O(k) retrieval.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pinocchio.models.enums import MemoryTier
from pinocchio.models.schemas import SemanticEntry


class SemanticMemory:
    """Persistent semantic memory store backed by a JSON file.

    Skills / Capabilities:
      - Store generalizable knowledge entries with domain tags
      - Retrieve knowledge by domain or keyword
      - Update confidence scores as more evidence accumulates
      - Merge duplicate knowledge entries
      - Trigger knowledge synthesis when episode threshold is reached
      - Persist to disk for cross-session continuity
      - O(1) lookup by entry_id via hash index
      - O(k) search by domain via inverted index
    """

    SYNTHESIS_THRESHOLD = 10  # episodes per domain before synthesis

    def __init__(self, storage_path: str = "data/semantic_memory.json") -> None:
        self._path = Path(storage_path)
        self._entries: list[SemanticEntry] = []
        # ── Indices ──
        self._id_index: dict[str, SemanticEntry] = {}
        self._domain_index: dict[str, list[SemanticEntry]] = defaultdict(list)
        self._load()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_entry(self, entry: SemanticEntry) -> None:
        self._id_index[entry.entry_id] = entry
        self._domain_index[entry.domain.lower()].append(entry)

    def _rebuild_indices(self) -> None:
        self._id_index.clear()
        self._domain_index.clear()
        for e in self._entries:
            self._index_entry(e)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw: list[dict[str, Any]] = json.load(f)
                self._entries = [SemanticEntry.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                import shutil
                backup = self._path.with_suffix(".json.bak")
                shutil.copy2(self._path, backup)
                self._entries = []
        self._rebuild_indices()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: SemanticEntry) -> None:
        self._entries.append(entry)
        self._index_entry(entry)
        self.save()

    def get(self, entry_id: str) -> SemanticEntry | None:
        return self._id_index.get(entry_id)

    def all(self) -> list[SemanticEntry]:
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search_by_domain(self, domain: str, limit: int = 10) -> list[SemanticEntry]:
        """Return entries matching a domain (case-insensitive, substring match).

        Uses the domain index to collect candidates whose domain key contains
        the query as a substring, giving O(D) over distinct domains rather
        than O(N) over all entries when many entries share a domain.
        """
        domain_lower = domain.lower()
        results: list[SemanticEntry] = []
        for key, entries in self._domain_index.items():
            if domain_lower in key:
                results.extend(entries)
                if len(results) >= limit:
                    return results[:limit]
        return results[:limit]

    def search_by_keyword(self, keyword: str, limit: int = 10) -> list[SemanticEntry]:
        keyword_lower = keyword.lower()
        return [
            e
            for e in self._entries
            if keyword_lower in e.knowledge.lower() or keyword_lower in e.domain.lower()
        ][:limit]

    def get_high_confidence(self, threshold: float = 0.7) -> list[SemanticEntry]:
        """Return entries with confidence above a threshold."""
        return [e for e in self._entries if e.confidence >= threshold]

    def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 5,
        threshold: float = 0.3,
    ) -> list[SemanticEntry]:
        """Return entries most similar to *query_embedding* by cosine similarity.

        Only entries that have a stored embedding and whose similarity
        exceeds *threshold* are considered.
        """
        from pinocchio.utils.llm_client import EmbeddingClient

        scored: list[tuple[SemanticEntry, float]] = []
        for entry in self._entries:
            if not entry.embedding:
                continue
            sim = EmbeddingClient.cosine_similarity(query_embedding, entry.embedding)
            if sim >= threshold:
                scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Update & Merge
    # ------------------------------------------------------------------

    def update_confidence(self, entry_id: str, new_confidence: float) -> None:
        entry = self.get(entry_id)
        if entry:
            entry.confidence = max(0.0, min(1.0, new_confidence))
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self.save()

    def add_source_episode(self, entry_id: str, episode_id: str) -> None:
        entry = self.get(entry_id)
        if entry and episode_id not in entry.source_episodes:
            entry.source_episodes.append(episode_id)
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self.save()

    def domain_entry_count(self, domain: str) -> int:
        return len(self.search_by_domain(domain))

    def needs_synthesis(self, domain: str, episode_count: int) -> bool:
        """Check if domain has accumulated enough episodes for knowledge synthesis."""
        return episode_count >= self.SYNTHESIS_THRESHOLD

    # ------------------------------------------------------------------
    # Tier-aware queries (temporal axis)
    # ------------------------------------------------------------------

    def get_by_tier(self, tier: MemoryTier) -> list[SemanticEntry]:
        """Return all entries at a given temporal tier."""
        return [e for e in self._entries if e.memory_tier == tier]

    def get_persistent(self) -> list[SemanticEntry]:
        """Return core knowledge entries that are permanently stored."""
        return self.get_by_tier(MemoryTier.PERSISTENT)

    def promote_to_persistent(self, entry_id: str) -> bool:
        """Promote a knowledge entry to persistent tier."""
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.memory_tier = MemoryTier.PERSISTENT
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()
        return True

    def consolidate_high_confidence(self, threshold: float = 0.85) -> list[SemanticEntry]:
        """Identify long-term entries that should be promoted to persistent.

        High-confidence entries backed by multiple source episodes are
        candidates for permanent retention.
        """
        return [
            e for e in self._entries
            if e.memory_tier == MemoryTier.LONG_TERM
            and e.confidence >= threshold
            and len(e.source_episodes) >= 2
        ]
