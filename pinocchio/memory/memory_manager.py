"""Memory Manager — unified façade over the three memory subsystems.

Provides a single entry-point for reading and writing episodic, semantic,
and procedural memories, including cross-memory operations such as
knowledge synthesis and experience retrieval.
"""

from __future__ import annotations

from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
from pinocchio.models.enums import Modality, TaskType
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
)


class MemoryManager:
    """Unified façade for Pinocchio's three-part memory system.

    Skills / Capabilities:
      - Coordinate reads/writes across episodic, semantic, and procedural stores
      - Provide a single *recall* method that searches all three stores
      - Trigger knowledge synthesis when a domain reaches the episode threshold
      - Compute cross-memory analytics (improvement trends, strategy rankings)
      - Ensure data consistency across memory stores
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.episodic = EpisodicMemory(f"{data_dir}/episodic_memory.json")
        self.semantic = SemanticMemory(f"{data_dir}/semantic_memory.json")
        self.procedural = ProceduralMemory(f"{data_dir}/procedural_memory.json")
        # Domains pending synthesis — consumed by LearningAgent
        self._pending_synthesis: list[str] = []

    # ------------------------------------------------------------------
    # Unified recall
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
        """
        similar = self.episodic.find_similar(task_type, modalities, limit=3)
        knowledge: list[SemanticEntry] = []
        if keyword:
            knowledge = self.semantic.search_by_keyword(keyword, limit=5)
        else:
            knowledge = self.semantic.search_by_domain(task_type.value, limit=5)
        procedure = self.procedural.best_procedure(task_type)
        lessons = self.episodic.recent_lessons(limit=5)
        return {
            "similar_episodes": similar,
            "relevant_knowledge": knowledge,
            "best_procedure": procedure,
            "recent_lessons": lessons,
        }

    # ------------------------------------------------------------------
    # Cross-memory operations
    # ------------------------------------------------------------------

    def store_episode(self, episode: EpisodicRecord) -> None:
        """Store an episode and flag the domain for synthesis if threshold is reached."""
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
        self.semantic.add(entry)

    def store_procedure(self, entry: ProceduralEntry) -> None:
        self.procedural.add(entry)

    def record_procedure_usage(self, entry_id: str, success: bool) -> None:
        self.procedural.record_usage(entry_id, success)

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
        """High-level summary of all memory stores."""
        return {
            "episodic_count": self.episodic.count,
            "semantic_count": self.semantic.count,
            "procedural_count": self.procedural.count,
            "avg_score": round(self.episodic.average_score(), 2),
            "error_frequency": self.episodic.error_frequency(),
            "top_procedures": [
                {"name": p.name, "success_rate": round(p.success_rate, 2)}
                for p in self.procedural.top_procedures(limit=3)
            ],
        }
