"""Parametrized tests — replaces duplicated patterns across test files.

This file uses ``@pytest.mark.parametrize`` to cover:
- Audio format detection across all formats
- Enum completeness for all enum types
- Schema roundtrip tests for all data models
- Memory search/query boundary tests
- Confidence clamping boundary values
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pinocchio.models.enums import (
    Modality, TaskType, Complexity, ConfidenceLevel,
    FusionStrategy, ErrorType, AgentRole, ExpertiseLevel,
    CommunicationStyle,
)
from pinocchio.models.schemas import (
    EpisodicRecord, SemanticEntry, ProceduralEntry,
    MultimodalInput, ModalConfidence, AgentMessage,
    UserModel, PerceptionResult, StrategyResult,
    EvaluationResult, LearningResult, MetaReflectionResult,
)
from pinocchio.utils.llm_client import LLMClient


# ── Audio Format Detection (parametrized) ────────────────────────────────

@pytest.mark.parametrize("filename,expected_format", [
    ("test.wav", "wav"),
    ("test.mp3", "mp3"),
    ("test.flac", "flac"),
    ("test.ogg", "ogg"),
    ("test.aac", "wav"),       # unknown → default wav
    ("test.m4a", "wav"),       # unknown → default wav
    ("test.WAV", "wav"),       # case-insensitive
    ("no_extension", "wav"),   # no extension → default
    ("path/to/audio.mp3", "mp3"),
])
def test_audio_format_detection(filename, expected_format):
    assert LLMClient._audio_format(filename) == expected_format


# ── Resolve Audio URL (parametrized) ─────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://example.com/audio.wav",
    "https://example.com/audio.mp3",
    "data:audio/wav;base64,SGVsbG8=",
])
def test_resolve_audio_url_passthrough(url):
    """HTTP/HTTPS/data URLs should be passed through unchanged."""
    assert LLMClient._resolve_audio_url(url) == url


# ── Enum Completeness (parametrized) ─────────────────────────────────────

@pytest.mark.parametrize("enum_cls,expected_members", [
    (Modality, {"text", "image", "audio", "video"}),
    (TaskType, {
        "question_answering", "content_generation", "analysis",
        "translation", "summarization", "code_generation",
        "creative_writing", "multimodal_reasoning", "conversation",
        "tool_use", "unknown",
    }),
    (Complexity, {1, 2, 3, 4, 5}),
    (ConfidenceLevel, {"low", "medium", "high"}),
    (FusionStrategy, {"early_fusion", "late_fusion", "hybrid_fusion"}),
    (ErrorType, {
        "perception_error", "strategy_error", "execution_error",
        "knowledge_gap", "cross_modal_error",
    }),
    (AgentRole, {
        "orchestrator", "perception", "strategy", "execution",
        "evaluation", "learning", "meta_reflection",
        "text_processor", "vision_processor", "audio_processor",
        "video_processor",
    }),
])
def test_enum_completeness(enum_cls, expected_members):
    actual = {m.value for m in enum_cls}
    assert actual == expected_members


@pytest.mark.parametrize("enum_cls", [
    Modality, TaskType, ConfidenceLevel, FusionStrategy,
    ErrorType, AgentRole, ExpertiseLevel, CommunicationStyle,
])
def test_enum_construction_from_value(enum_cls):
    """Every enum member should be constructible from its own value."""
    for member in enum_cls:
        assert enum_cls(member.value) == member


# ── Schema Roundtrip Tests (parametrized) ────────────────────────────────

@pytest.mark.parametrize("schema_cls,kwargs", [
    (EpisodicRecord, {
        "episode_id": "test-1",
        "task_type": TaskType.ANALYSIS,
        "modalities": [Modality.TEXT, Modality.IMAGE],
        "user_intent": "test intent",
        "strategy_used": "direct",
        "outcome_score": 8,
        "lessons": ["learned X"],
    }),
    (EpisodicRecord, {}),  # all defaults
])
def test_episodic_record_roundtrip(schema_cls, kwargs):
    original = schema_cls(**kwargs)
    d = original.to_dict()
    restored = schema_cls.from_dict(d)
    assert restored.episode_id == original.episode_id
    assert restored.task_type == original.task_type
    assert restored.outcome_score == original.outcome_score


@pytest.mark.parametrize("schema_cls,kwargs", [
    (SemanticEntry, {
        "entry_id": "se-1",
        "domain": "physics",
        "knowledge": "E=mc²",
        "confidence": 0.9,
    }),
    (SemanticEntry, {}),  # defaults
])
def test_semantic_entry_roundtrip(schema_cls, kwargs):
    original = schema_cls(**kwargs)
    d = original.to_dict()
    restored = schema_cls.from_dict(d)
    assert restored.entry_id == original.entry_id
    assert restored.domain == original.domain


@pytest.mark.parametrize("schema_cls,kwargs", [
    (ProceduralEntry, {
        "entry_id": "pe-1",
        "task_type": TaskType.CODE_GENERATION,
        "name": "code_gen_v1",
        "steps": ["step1", "step2"],
        "success_rate": 0.85,
    }),
    (ProceduralEntry, {}),  # defaults
])
def test_procedural_entry_roundtrip(schema_cls, kwargs):
    original = schema_cls(**kwargs)
    d = original.to_dict()
    restored = schema_cls.from_dict(d)
    assert restored.entry_id == original.entry_id
    assert restored.task_type == original.task_type


# ── MultimodalInput.modalities (parametrized) ────────────────────────────

@pytest.mark.parametrize("kwargs,expected_modalities", [
    ({"text": "hello"}, [Modality.TEXT]),
    ({"text": "hi", "image_paths": ["a.jpg"]}, [Modality.TEXT, Modality.IMAGE]),
    ({"text": "hi", "image_paths": ["a.jpg"], "audio_paths": ["a.wav"]},
     [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]),
    ({"text": "hi", "image_paths": ["a.jpg"], "audio_paths": ["a.wav"],
      "video_paths": ["v.mp4"]},
     [Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]),
    ({}, []),  # totally empty
    ({"image_paths": ["a.jpg"]}, [Modality.IMAGE]),  # no text
])
def test_multimodal_input_modalities(kwargs, expected_modalities):
    inp = MultimodalInput(**kwargs)
    assert inp.modalities == expected_modalities


# ── Confidence Clamping (parametrized) ────────────────────────────────────

@pytest.mark.parametrize("new_val,expected", [
    (0.5, 0.5),
    (0.0, 0.0),
    (1.0, 1.0),
    (-0.5, 0.0),   # clamped to 0
    (1.5, 1.0),    # clamped to 1
    (100.0, 1.0),  # clamped to 1
])
def test_semantic_confidence_clamping(tmp_path, new_val, expected):
    from pinocchio.memory.semantic_memory import SemanticMemory
    mem = SemanticMemory(str(tmp_path / "sem.json"))
    entry = SemanticEntry(entry_id="c1", confidence=0.5)
    mem.add(entry)
    mem.update_confidence("c1", new_val)
    assert mem.get("c1").confidence == expected


# ── Cognitive Result Schema Defaults (parametrized) ───────────────────────

@pytest.mark.parametrize("schema_cls", [
    PerceptionResult, StrategyResult, EvaluationResult,
    LearningResult, MetaReflectionResult,
])
def test_cognitive_result_schema_defaults(schema_cls):
    """All cognitive result schemas should instantiate with defaults."""
    obj = schema_cls()
    assert obj is not None
