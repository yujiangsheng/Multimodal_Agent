"""Data schemas for the Pinocchio agent system.

Every structured data object flowing through the cognitive loop is defined
here as a ``@dataclass``.  This includes:

* **Memory schemas** — ``EpisodicRecord``, ``SemanticEntry``, ``ProceduralEntry``
* **Cognitive-loop results** — one per phase (Perception → Meta-Reflection)
* **Communication schemas** — ``UserModel``, ``MultimodalInput``, ``AgentMessage``

All memory schemas provide ``to_dict()`` / ``from_dict()`` round-trip
serialisation for JSON persistence.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from pinocchio.models.enums import (
    Modality,
    TaskType,
    Complexity,
    ConfidenceLevel,
    ErrorType,
    FusionStrategy,
    ExpertiseLevel,
    CommunicationStyle,
    MemoryTier,
)


# ---------------------------------------------------------------------------
# Memory Schemas
# ---------------------------------------------------------------------------


@dataclass
class EpisodicRecord:
    """A single episode in episodic memory — one interaction's full trace."""

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task_type: TaskType = TaskType.UNKNOWN
    modalities: list[Modality] = field(default_factory=list)
    user_intent: str = ""
    strategy_used: str = ""
    outcome_score: int = 5            # 1-10
    lessons: list[str] = field(default_factory=list)
    error_patterns: list[str] = field(default_factory=list)
    improvement_notes: str = ""
    memory_tier: MemoryTier = MemoryTier.LONG_TERM  # temporal axis
    embedding: list[float] = field(default_factory=list)  # optional vector

    def to_dict(self) -> dict[str, Any]:
        d = {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "task_type": self.task_type.value,
            "modalities": [m.value for m in self.modalities],
            "user_intent": self.user_intent,
            "strategy_used": self.strategy_used,
            "outcome_score": self.outcome_score,
            "lessons": self.lessons,
            "error_patterns": self.error_patterns,
            "improvement_notes": self.improvement_notes,
            "memory_tier": self.memory_tier.value,
        }
        if self.embedding:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicRecord":
        raw = dict(data)
        if "task_type" in raw:
            raw["task_type"] = TaskType(raw["task_type"])
        if "modalities" in raw:
            raw["modalities"] = [Modality(m) for m in raw["modalities"]]
        if "memory_tier" in raw:
            raw["memory_tier"] = MemoryTier(raw["memory_tier"])
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class SemanticEntry:
    """A distilled knowledge entry in semantic memory."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain: str = ""
    knowledge: str = ""
    source_episodes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = ""
    memory_tier: MemoryTier = MemoryTier.LONG_TERM  # temporal axis
    embedding: list[float] = field(default_factory=list)  # optional vector

    def to_dict(self) -> dict[str, Any]:
        d = {
            "entry_id": self.entry_id,
            "domain": self.domain,
            "knowledge": self.knowledge,
            "source_episodes": self.source_episodes,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "memory_tier": self.memory_tier.value,
        }
        if self.embedding:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticEntry":
        raw = dict(data)
        if "memory_tier" in raw:
            raw["memory_tier"] = MemoryTier(raw["memory_tier"])
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class ProceduralEntry:
    """A reusable procedure / action template in procedural memory."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType = TaskType.UNKNOWN
    name: str = ""
    description: str = ""
    steps: list[str] = field(default_factory=list)
    success_rate: float = 0.0  # 0-1
    usage_count: int = 0
    last_used: str = ""
    memory_tier: MemoryTier = MemoryTier.LONG_TERM  # temporal axis

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "memory_tier": self.memory_tier.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProceduralEntry":
        raw = dict(data)
        if "task_type" in raw:
            raw["task_type"] = TaskType(raw["task_type"])
        if "memory_tier" in raw:
            raw["memory_tier"] = MemoryTier(raw["memory_tier"])
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Cognitive Loop Result Schemas
# ---------------------------------------------------------------------------


@dataclass
class ModalConfidence:
    """Per-modality confidence scores."""

    text: float = 0.0
    image: float = 0.0
    audio: float = 0.0
    video: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "text": self.text,
            "image": self.image,
            "audio": self.audio,
            "video": self.video,
        }


@dataclass
class PerceptionResult:
    """Output of the PERCEIVE phase."""

    modalities: list[Modality] = field(default_factory=list)
    task_type: TaskType = TaskType.UNKNOWN
    complexity: Complexity = Complexity.MODERATE
    similar_episodes: list[str] = field(default_factory=list)
    relevant_lessons: list[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ambiguities: list[str] = field(default_factory=list)
    raw_analysis: str = ""


@dataclass
class StrategyResult:
    """Output of the STRATEGIZE phase."""

    selected_strategy: str = ""
    basis: str = ""
    risk_assessment: str = ""
    fallback_plan: str = ""
    modality_pipeline: str = ""
    fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION
    is_novel: bool = False
    raw_analysis: str = ""


@dataclass
class EvaluationResult:
    """Output of the EVALUATE phase."""

    task_completion: str = "complete"  # complete / partial / failed
    output_quality: int = 5           # 1-10
    strategy_effectiveness: int = 5   # 1-10
    went_well: list[str] = field(default_factory=list)
    went_wrong: list[str] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)
    cross_modal_coherence: int = 5    # 1-10
    is_complete: bool = True           # was the response fully formed?
    incompleteness_details: str = ""   # why the response is incomplete
    user_satisfaction: str = "awaiting"
    raw_analysis: str = ""


@dataclass
class LearningResult:
    """Output of the LEARN phase."""

    new_lessons: list[str] = field(default_factory=list)
    episodic_update: str = ""
    semantic_updates: list[str] = field(default_factory=list)
    procedural_updates: list[str] = field(default_factory=list)
    strategy_refinements: str = ""
    skill_gap: str = ""
    self_improvement_action: str = ""
    raw_analysis: str = ""


@dataclass
class MetaReflectionResult:
    """Output of the META-REFLECT phase."""

    recurring_errors: list[str] = field(default_factory=list)
    strength_domains: list[str] = field(default_factory=list)
    weakness_domains: list[str] = field(default_factory=list)
    strategy_trajectory: str = ""
    bias_check: str = ""
    learning_efficiency: str = ""
    priority_improvements: list[str] = field(default_factory=list)
    experimental_strategies: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    raw_analysis: str = ""


# ---------------------------------------------------------------------------
# Communication & I/O Schemas
# ---------------------------------------------------------------------------


@dataclass
class UserModel:
    """Adaptive model of the current user."""

    expertise: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    style: CommunicationStyle = CommunicationStyle.DETAILED
    domains_of_interest: list[str] = field(default_factory=list)
    feedback_history: list[str] = field(default_factory=list)
    interaction_count: int = 0


@dataclass
class MultimodalInput:
    """A single input message that may contain multiple modalities."""

    text: str | None = None
    image_paths: list[str] = field(default_factory=list)
    audio_paths: list[str] = field(default_factory=list)
    video_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> list[Modality]:
        mods: list[Modality] = []
        if self.text:
            mods.append(Modality.TEXT)
        if self.image_paths:
            mods.append(Modality.IMAGE)
        if self.audio_paths:
            mods.append(Modality.AUDIO)
        if self.video_paths:
            mods.append(Modality.VIDEO)
        return mods


@dataclass
class AgentMessage:
    """A message exchanged between agents or returned to the user."""

    role: str = "assistant"
    content: str = ""
    modality: Modality = Modality.TEXT
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
