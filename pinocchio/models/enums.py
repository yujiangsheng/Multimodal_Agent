"""Enumerations for the Pinocchio agent system.

Defines the core taxonomy used throughout the cognitive loop:

* **Modality** — input/output data types (text, image, audio, video)
* **TaskType** — what the user is asking the agent to do
* **Complexity** — how hard the task is (1–5 scale)
* **ConfidenceLevel** — the agent's self-assessed certainty
* **ErrorType** — root-cause categories for failed interactions
* **FusionStrategy** — how to merge information across modalities
* **AgentRole** — which sub-agent is speaking (used for logging)
* **ExpertiseLevel / CommunicationStyle** — adaptive user model axes

All enums inherit from ``str`` (or ``int``) so they serialise
naturally to/from JSON without custom encoders.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Modality & Task Classification
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """Supported input/output modalities.

    Used by :class:`~pinocchio.models.schemas.MultimodalInput` and the
    perception phase to declare which data channels are present.
    """

    TEXT = "text"      # Natural-language text
    IMAGE = "image"    # Static images (JPEG, PNG, WebP, …)
    AUDIO = "audio"    # Audio files (WAV, MP3, FLAC, …)
    VIDEO = "video"    # Video files (MP4, MOV, …)


class TaskType(str, Enum):
    """Coarse-grained classification of user tasks.

    Assigned during the *PERCEIVE* phase and used to look up relevant
    episodic/procedural memory, select strategies, and generate evaluation
    criteria.
    """

    QUESTION_ANSWERING = "question_answering"        # Factual / knowledge Q&A
    CONTENT_GENERATION = "content_generation"        # Generic content creation
    ANALYSIS = "analysis"                            # Data / document analysis
    TRANSLATION = "translation"                      # Language translation
    SUMMARIZATION = "summarization"                  # Condensation / abstractive summary
    CODE_GENERATION = "code_generation"              # Writing or debugging code
    CREATIVE_WRITING = "creative_writing"            # Stories, poetry, copywriting
    MULTIMODAL_REASONING = "multimodal_reasoning"    # Cross-modal inference
    CONVERSATION = "conversation"                    # Casual / chitchat
    TOOL_USE = "tool_use"                            # External tool invocation
    UNKNOWN = "unknown"                              # Fallback when classification fails


# ---------------------------------------------------------------------------
# Difficulty & Confidence
# ---------------------------------------------------------------------------


class Complexity(int, Enum):
    """Task complexity on a 1–5 scale.

    Set during the *PERCEIVE* phase based on the number of reasoning steps,
    domain expertise required, and modality interactions involved.
    """

    TRIVIAL = 1    # Single-step, common-knowledge answer
    SIMPLE = 2     # Straightforward but needs some reasoning
    MODERATE = 3   # Multi-step reasoning or domain knowledge
    COMPLEX = 4    # Deep expertise or cross-modal reasoning
    EXTREME = 5    # Research-level, multi-domain, adversarial input


class ConfidenceLevel(str, Enum):
    """Agent's self-assessed confidence for a given interaction.

    Reported during perception and carried through to evaluation
    to calibrate risk-taking in the strategy phase.
    """

    LOW = "low"       # High uncertainty — may need clarification
    MEDIUM = "medium" # Reasonable understanding, moderate risk
    HIGH = "high"     # Strong certainty — confident in approach


# ---------------------------------------------------------------------------
# Error & Evaluation
# ---------------------------------------------------------------------------


class ErrorType(str, Enum):
    """Root-cause categories for failed or degraded interactions.

    Used during the *EVALUATE* and *LEARN* phases to classify what went
    wrong so the agent can target improvements.
    """

    PERCEPTION_ERROR = "perception_error"        # Misunderstood input/intent
    STRATEGY_ERROR = "strategy_error"            # Chose wrong approach
    EXECUTION_ERROR = "execution_error"          # Correct strategy, poor execution
    KNOWLEDGE_GAP = "knowledge_gap"              # Lacked necessary information
    CROSS_MODAL_ERROR = "cross_modal_error"      # Inconsistency between modalities


# ---------------------------------------------------------------------------
# Multimodal Fusion
# ---------------------------------------------------------------------------


class FusionStrategy(str, Enum):
    """Multimodal fusion strategies for combining information across modalities.

    Selected during the *STRATEGIZE* phase based on task type and the
    modalities present in the input.
    """

    EARLY_FUSION = "early_fusion"    # Combine raw features before reasoning
    LATE_FUSION = "late_fusion"      # Reason per-modality, then integrate
    HYBRID_FUSION = "hybrid_fusion"  # Mix of early and late


# ---------------------------------------------------------------------------
# Agent Identity & User Modelling
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    """Identity tags for each sub-agent in the Pinocchio system.

    Used by :class:`~pinocchio.utils.logger.PinocchioLogger` to colour-code
    and prefix log messages for easy tracing.
    """

    ORCHESTRATOR = "orchestrator"            # Top-level coordinator
    PERCEPTION = "perception"                # Phase 1 — PERCEIVE
    STRATEGY = "strategy"                    # Phase 2 — STRATEGIZE
    EXECUTION = "execution"                  # Phase 3 — EXECUTE
    EVALUATION = "evaluation"                # Phase 4 — EVALUATE
    LEARNING = "learning"                    # Phase 5 — LEARN
    META_REFLECTION = "meta_reflection"      # Phase 6 — META-REFLECT
    TEXT_PROCESSOR = "text_processor"        # Multimodal: text
    VISION_PROCESSOR = "vision_processor"    # Multimodal: image
    AUDIO_PROCESSOR = "audio_processor"      # Multimodal: audio
    VIDEO_PROCESSOR = "video_processor"      # Multimodal: video


class ExpertiseLevel(str, Enum):
    """Estimated expertise of the user for adaptive communication.

    Updated across interactions as the agent learns about the user's
    domain knowledge and the technical depth of their questions.
    """

    BEGINNER = "beginner"           # Needs simplified explanations
    INTERMEDIATE = "intermediate"   # Comfortable with domain terminology
    EXPERT = "expert"               # Expects concise, technical answers


class CommunicationStyle(str, Enum):
    """User's preferred communication style.

    Influences tone, length, and formatting choices in the agent's responses.
    """

    CONCISE = "concise"     # Short, to-the-point answers
    DETAILED = "detailed"   # Thorough explanations with examples
    VISUAL = "visual"       # Prefer diagrams, charts, and structured formats
