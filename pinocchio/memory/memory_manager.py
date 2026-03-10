"""Memory Manager — unified façade over the dual-axis memory system.

Pinocchio's memory architecture is modelled after human cognition and
has two orthogonal classification axes:

**Content axis** (what is stored):

=============  ======================  ====================================
Store          Human analogy           Stores
=============  ======================  ====================================
Episodic       "I remember that time…" Full interaction traces (intent,
                                        strategy, outcome, lessons)
Semantic       "I know that…"          Distilled, generalisable knowledge
                                        extracted from multiple episodes
Procedural     "I know how to…"        Reusable step-by-step action
                                        templates with success-rate tracking
=============  ======================  ====================================

**Temporal axis** (how long it lives):

===========  ============  ===============================================
Tier         Lifetime       Description
===========  ============  ===============================================
Working      Session        Volatile conversation buffer (in-RAM, FIFO)
Long-term    Cross-session  JSON-persisted, subject to decay / pruning
Persistent   Permanent      High-value entries auto-promoted, never pruned
===========  ============  ===============================================

This manager coordinates reads/writes across all stores and provides:

- Unified ``recall()`` that searches all content + temporal stores at once
- ``store_episode()`` that auto-flags domains for knowledge synthesis
- ``consolidate()`` that promotes high-value entries to the persistent tier
- ``improvement_trend()`` analytics over a sliding window
- ``summary()`` for the status dashboard

Example
-------
>>> mm = MemoryManager(data_dir="data")
>>> context = mm.recall(TaskType.CODE_GENERATION, [Modality.TEXT])
>>> mm.store_episode(episode)
>>> promoted = mm.consolidate()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
from pinocchio.memory.working_memory import WorkingMemory
from pinocchio.models.enums import MemoryTier, Modality, TaskType
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
)

if TYPE_CHECKING:
    from pinocchio.utils.llm_client import EmbeddingClient


class MemoryManager:
    """Unified façade for Pinocchio's dual-axis memory system.

    Provides a single entry point for all memory operations.  The four
    underlying stores are accessible as attributes:

    - ``self.episodic``   — :class:`EpisodicMemory`  (content: episodic)
    - ``self.semantic``   — :class:`SemanticMemory`   (content: semantic)
    - ``self.procedural`` — :class:`ProceduralMemory` (content: procedural)
    - ``self.working``    — :class:`WorkingMemory`    (temporal: working)

    Parameters
    ----------
    data_dir : str
        Directory where the three JSON persistence files are stored.
        Created automatically if it does not exist.
    """

    def __init__(self, data_dir: str = "data") -> None:
        # ── Content-axis stores (persistent on disk) ──
        self.episodic = EpisodicMemory(f"{data_dir}/episodic_memory.json")
        self.semantic = SemanticMemory(f"{data_dir}/semantic_memory.json")
        self.procedural = ProceduralMemory(f"{data_dir}/procedural_memory.json")

        # ── Temporal-axis: working memory (volatile, in-RAM) ──
        self.working = WorkingMemory(capacity=50)

        # Optional embedding client for vector search
        self._embedding_client: EmbeddingClient | None = None

        # Domains pending synthesis — consumed by LearningAgent
        self._pending_synthesis: list[str] = []

    def set_embedding_client(self, client: EmbeddingClient) -> None:
        """Attach an embedding client for vector-based memory recall."""
        self._embedding_client = client

    # ------------------------------------------------------------------
    # Unified recall (both axes)
    # ------------------------------------------------------------------

    def recall(
        self,
        task_type: TaskType,
        modalities: list[Modality],
        keyword: str = "",
    ) -> dict:
        """Retrieve relevant context from all memory stores at once.

        Returns a dict with:
          - similar_episodes: list[EpisodicRecord]
          - relevant_knowledge: list[SemanticEntry]
          - best_procedure: ProceduralEntry | None
          - recent_lessons: list[str]
          - persistent_knowledge: list[SemanticEntry]
          - working_context: str
        """
        similar = self.episodic.find_similar(task_type, modalities, limit=3)
        knowledge: list[SemanticEntry] = []
        if keyword:
            knowledge = self.semantic.search_by_keyword(keyword, limit=5)
        else:
            knowledge = self.semantic.search_by_domain(task_type.value, limit=5)
        procedure = self.procedural.best_procedure(task_type)
        lessons = self.episodic.recent_lessons(limit=5)

        # Vector search enrichment (when embedding client is available)
        if self._embedding_client and keyword:
            try:
                q_emb = self._embedding_client.embed(keyword)
                emb_episodes = self.episodic.search_by_embedding(q_emb, limit=3)
                emb_knowledge = self.semantic.search_by_embedding(q_emb, limit=3)
                # Merge without duplicates
                existing_ep_ids = {ep.episode_id for ep in similar}
                for ep in emb_episodes:
                    if ep.episode_id not in existing_ep_ids:
                        similar.append(ep)
                existing_se_ids = {e.entry_id for e in knowledge}
                for e in emb_knowledge:
                    if e.entry_id not in existing_se_ids:
                        knowledge.append(e)
            except Exception:
                pass  # embedding service unavailable — fall back to keyword

        # Temporal-axis enrichment
        persistent_knowledge = self.semantic.get_persistent()[:5]
        working_context = self.working.format_conversation_context(max_turns=5)

        return {
            "similar_episodes": similar,
            "relevant_knowledge": knowledge,
            "best_procedure": procedure,
            "recent_lessons": lessons,
            "persistent_knowledge": persistent_knowledge,
            "working_context": working_context,
        }

    # ------------------------------------------------------------------
    # Cross-memory operations
    # ------------------------------------------------------------------

    def store_episode(self, episode: EpisodicRecord) -> None:
        """Store an episode and flag the domain for synthesis if threshold is reached."""
        # Auto-embed if embedding client is available
        if self._embedding_client and not episode.embedding:
            try:
                text = f"{episode.user_intent} {episode.strategy_used} {' '.join(episode.lessons)}"
                vec = self._embedding_client.embed(text)
                if isinstance(vec, list) and all(isinstance(v, (int, float)) for v in vec):
                    episode.embedding = vec
            except Exception:
                pass
        self.episodic.add(episode)
        domain = episode.task_type.value
        domain_episodes = self.episodic.search_by_task_type(episode.task_type, limit=100)
        if self.semantic.needs_synthesis(domain, len(domain_episodes)):
            if domain not in self._pending_synthesis:
                self._pending_synthesis.append(domain)

    def pop_pending_synthesis(self) -> list[str]:
        """Return and clear domains that are ready for knowledge synthesis.

        This is consumed by the LearningAgent after each interaction to
        decide whether to trigger cross-episode synthesis for specific domains.
        """
        domains = self._pending_synthesis.copy()
        self._pending_synthesis.clear()
        return domains

    def store_knowledge(self, entry: SemanticEntry) -> None:
        # Auto-embed if embedding client is available
        if self._embedding_client and not entry.embedding:
            try:
                vec = self._embedding_client.embed(entry.knowledge)
                if isinstance(vec, list) and all(isinstance(v, (int, float)) for v in vec):
                    entry.embedding = vec
            except Exception:
                pass
        self.semantic.add(entry)

    def store_procedure(self, entry: ProceduralEntry) -> None:
        self.procedural.add(entry)

    def record_procedure_usage(self, entry_id: str, success: bool) -> None:
        self.procedural.record_usage(entry_id, success)

    # ------------------------------------------------------------------
    # Temporal-axis: consolidation & promotion
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, int]:
        """Run consolidation across all content stores.

        Promotes high-value long-term entries to persistent tier.
        Returns counts of promoted entries per store.
        """
        promoted: dict[str, int] = {"episodic": 0, "semantic": 0, "procedural": 0}

        # Episodic: promote high-score episodes with lessons
        for ep in self.episodic.consolidate_high_value():
            ep.memory_tier = MemoryTier.PERSISTENT
            promoted["episodic"] += 1
        if promoted["episodic"]:
            self.episodic.save()

        # Semantic: promote high-confidence, well-sourced knowledge
        for entry in self.semantic.consolidate_high_confidence():
            entry.memory_tier = MemoryTier.PERSISTENT
            promoted["semantic"] += 1
        if promoted["semantic"]:
            self.semantic.save()

        # Procedural: promote proven procedures
        for entry in self.procedural.consolidate_proven():
            entry.memory_tier = MemoryTier.PERSISTENT
            promoted["procedural"] += 1
        if promoted["procedural"]:
            self.procedural.save()

        return promoted

    def reset_working_memory(self) -> None:
        """Clear the volatile working memory (session reset)."""
        self.working.clear()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def improvement_trend(self, window: int = 10) -> dict:
        """Return improvement metrics over a sliding window."""
        all_eps = self.episodic.all()
        if len(all_eps) < window:
            return {"recent_avg": self.episodic.average_score(), "trend": "insufficient_data"}
        recent = self.episodic.average_score(last_n=window)
        older = self.episodic.average_score(last_n=len(all_eps)) if len(all_eps) > window else recent
        if recent > older + 0.5:
            trend = "improving"
        elif recent < older - 0.5:
            trend = "declining"
        else:
            trend = "stable"
        return {"recent_avg": round(recent, 2), "older_avg": round(older, 2), "trend": trend}

    def summary(self) -> dict:
        """High-level summary of all memory stores (both axes)."""
        return {
            # Content axis
            "episodic_count": self.episodic.count,
            "semantic_count": self.semantic.count,
            "procedural_count": self.procedural.count,
            "avg_score": round(self.episodic.average_score(), 2),
            "error_frequency": self.episodic.error_frequency(),
            "top_procedures": [
                {"name": p.name, "success_rate": round(p.success_rate, 2)}
                for p in self.procedural.top_procedures(limit=3)
            ],
            # Temporal axis
            "working_memory": self.working.summary(),
            "persistent_episodes": len(self.episodic.get_persistent()),
            "persistent_knowledge": len(self.semantic.get_persistent()),
            "persistent_procedures": len(self.procedural.get_persistent()),
        }
