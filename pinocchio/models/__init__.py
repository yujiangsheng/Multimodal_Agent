"""Pinocchio data models and enumerations.

This subpackage defines all structured types flowing through the system:

* :mod:`enums`   — taxonomies (Modality, TaskType, AgentRole, …)
* :mod:`schemas` — dataclasses for memory records, cognitive-loop results,
                   and I/O messages
"""

from pinocchio.models.enums import (
    Modality,
    TaskType,
    Complexity,
    ConfidenceLevel,
    ErrorType,
    FusionStrategy,
    AgentRole,
    ExpertiseLevel,
    CommunicationStyle,
)
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
    PerceptionResult,
    StrategyResult,
    EvaluationResult,
    LearningResult,
    MetaReflectionResult,
    UserModel,
    ModalConfidence,
    MultimodalInput,
    AgentMessage,
)

__all__ = [
    "Modality",
    "TaskType",
    "Complexity",
    "ConfidenceLevel",
    "ErrorType",
    "FusionStrategy",
    "AgentRole",
    "ExpertiseLevel",
    "CommunicationStyle",
    "EpisodicRecord",
    "SemanticEntry",
    "ProceduralEntry",
    "PerceptionResult",
    "StrategyResult",
    "EvaluationResult",
    "LearningResult",
    "MetaReflectionResult",
    "UserModel",
    "ModalConfidence",
    "MultimodalInput",
    "AgentMessage",
]
