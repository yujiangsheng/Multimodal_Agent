"""Episodic Memory — structured log of past interactions.

Each episode captures the full trace of a single interaction: what was asked,
how the agent reasoned, what strategy it used, outcome quality, and lessons
learned.  This memory allows Pinocchio to recall specific past experiences
and apply relevant lessons to new tasks.

Performance: maintains inverted indices by task_type and modality so that
``search_by_task_type``, ``search_by_modality``, and ``find_similar`` run
in O(k) instead of O(n) for large memory stores.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pinocchio.models.enums import MemoryTier, Modality, TaskType
from pinocchio.models.schemas import EpisodicRecord


class EpisodicMemory:
    """Persistent episodic memory store backed by a JSON file.

    Skills / Capabilities:
      - Store and retrieve full interaction episodes
      - Search episodes by task type, modality, or keyword
      - Return the *k* most similar episodes to a given query
      - Compute aggregate statistics (avg score, error frequency)
      - Persist to disk for cross-session continuity
      - O(1) lookup by episode_id via hash index
      - O(k) search by task_type / modality via inverted indices
    """

    def __init__(self, storage_path: str = "data/episodic_memory.json") -> None:
        self._path = Path(storage_path)
        self._episodes: list[EpisodicRecord] = []
        # ── Inverted indices for fast retrieval ──
        self._id_index: dict[str, EpisodicRecord] = {}
        self._task_index: dict[TaskType, list[EpisodicRecord]] = defaultdict(list)
        self._modality_index: dict[Modality, list[EpisodicRecord]] = defaultdict(list)
        self._load()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_episode(self, ep: EpisodicRecord) -> None:
        """Add a single episode to all indices."""
        self._id_index[ep.episode_id] = ep
        self._task_index[ep.task_type].append(ep)
        for mod in ep.modalities:
            self._modality_index[mod].append(ep)

    def _rebuild_indices(self) -> None:
        """Rebuild all indices from scratch (called after _load)."""
        self._id_index.clear()
        self._task_index.clear()
        self._modality_index.clear()
        for ep in self._episodes:
            self._index_episode(ep)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw: list[dict[str, Any]] = json.load(f)
                self._episodes = [EpisodicRecord.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                # Corrupted file — start fresh but preserve the bad file
                import shutil
                backup = self._path.with_suffix(".json.bak")
                shutil.copy2(self._path, backup)
                self._episodes = []
        self._rebuild_indices()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._episodes], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, episode: EpisodicRecord) -> None:
        """Add a new episode to memory, update indices, and persist."""
        self._episodes.append(episode)
        self._index_episode(episode)
        self.save()

    def get(self, episode_id: str) -> EpisodicRecord | None:
        return self._id_index.get(episode_id)

    def all(self) -> list[EpisodicRecord]:
        return list(self._episodes)

    @property
    def count(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------
    # Query / Search
    # ------------------------------------------------------------------

    def search_by_task_type(self, task_type: TaskType, limit: int = 5) -> list[EpisodicRecord]:
        """Return most recent episodes of a given task type (index-accelerated)."""
        matches = self._task_index.get(task_type, [])
        return matches[-limit:]

    def search_by_modality(self, modality: Modality, limit: int = 5) -> list[EpisodicRecord]:
        """Return most recent episodes involving a given modality (index-accelerated)."""
        matches = self._modality_index.get(modality, [])
        return matches[-limit:]

    def search_by_keyword(self, keyword: str, limit: int = 5) -> list[EpisodicRecord]:
        """Simple keyword search across intent, strategy, lessons, and notes."""
        keyword_lower = keyword.lower()
        matches: list[EpisodicRecord] = []
        for ep in self._episodes:
            text_blob = " ".join(
                [ep.user_intent, ep.strategy_used, ep.improvement_notes]
                + ep.lessons
                + ep.error_patterns
            ).lower()
            if keyword_lower in text_blob:
                matches.append(ep)
        return matches[-limit:]

    def find_similar(
        self,
        task_type: TaskType,
        modalities: list[Modality],
        limit: int = 3,
    ) -> list[EpisodicRecord]:
        """Find the most similar past episodes by task type + modality overlap.

        Uses inverted indices to only score *candidate* episodes (those that
        share at least a task_type or one modality), instead of scanning all.

        Similarity heuristic:
          +2 for matching task type
          +1 for each shared modality
        Returns top-k by score, then by recency.
        """
        # Collect candidates from indices (union of task_type + modality hits)
        candidate_ids: set[str] = set()
        for ep in self._task_index.get(task_type, []):
            candidate_ids.add(ep.episode_id)
        for mod in modalities:
            for ep in self._modality_index.get(mod, []):
                candidate_ids.add(ep.episode_id)

        modality_set = set(modalities)

        def _score(ep: EpisodicRecord) -> int:
            s = 0
            if ep.task_type == task_type:
                s += 2
            s += len(set(ep.modalities) & modality_set)
            return s

        scored: list[tuple[EpisodicRecord, int]] = []
        for eid in candidate_ids:
            ep = self._id_index[eid]
            sc = _score(ep)
            if sc > 0:
                scored.append((ep, sc))

        scored.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        return [ep for ep, _ in scored[:limit]]

    def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 5,
        threshold: float = 0.3,
    ) -> list[EpisodicRecord]:
        """Return episodes most similar to *query_embedding* by cosine similarity."""
        from pinocchio.utils.llm_client import EmbeddingClient

        scored: list[tuple[EpisodicRecord, float]] = []
        for ep in self._episodes:
            if not ep.embedding:
                continue
            sim = EmbeddingClient.cosine_similarity(query_embedding, ep.embedding)
            if sim >= threshold:
                scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def average_score(self, last_n: int | None = None) -> float:
        """Average outcome score over the last *n* episodes (or all)."""
        episodes = self._episodes[-last_n:] if last_n else self._episodes
        if not episodes:
            return 0.0
        return sum(e.outcome_score for e in episodes) / len(episodes)

    def error_frequency(self) -> dict[str, int]:
        """Count of each error pattern across all episodes."""
        freq: dict[str, int] = {}
        for ep in self._episodes:
            for err in ep.error_patterns:
                freq[err] = freq.get(err, 0) + 1
        return freq

    def recent_lessons(self, limit: int = 10) -> list[str]:
        """Collect the most recent lessons across episodes."""
        lessons: list[str] = []
        for ep in reversed(self._episodes):
            lessons.extend(ep.lessons)
            if len(lessons) >= limit:
                break
        return lessons[:limit]

    # ------------------------------------------------------------------
    # Tier-aware queries (temporal axis)
    # ------------------------------------------------------------------

    def get_by_tier(self, tier: MemoryTier) -> list[EpisodicRecord]:
        """Return all episodes at a given temporal tier."""
        return [ep for ep in self._episodes if ep.memory_tier == tier]

    def get_persistent(self) -> list[EpisodicRecord]:
        """Return landmark episodes that are permanently stored."""
        return self.get_by_tier(MemoryTier.PERSISTENT)

    def promote_to_persistent(self, episode_id: str) -> bool:
        """Promote an episode to persistent tier (never pruned)."""
        ep = self.get(episode_id)
        if ep is None:
            return False
        ep.memory_tier = MemoryTier.PERSISTENT
        self.save()
        return True

    def consolidate_high_value(self, score_threshold: int = 8) -> list[EpisodicRecord]:
        """Identify long-term episodes that should be promoted to persistent.

        Episodes with high outcome scores and meaningful lessons are
        candidates for permanent retention.
        """
        candidates = [
            ep for ep in self._episodes
            if ep.memory_tier == MemoryTier.LONG_TERM
            and ep.outcome_score >= score_threshold
            and ep.lessons
        ]
        return candidates
