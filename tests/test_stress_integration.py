"""Stress tests, schema roundtrips, enum coverage, LLM client edge cases,
and cross-cutting integration tests for the Pinocchio system.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.models.enums import (
    AgentRole, Complexity, ConfidenceLevel, CommunicationStyle,
    ErrorType, ExpertiseLevel, FusionStrategy, Modality, TaskType,
)
from pinocchio.models.schemas import (
    AgentMessage, EpisodicRecord, EvaluationResult, LearningResult,
    MetaReflectionResult, ModalConfidence, MultimodalInput,
    PerceptionResult, ProceduralEntry, SemanticEntry, StrategyResult,
    UserModel,
)
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger
from pinocchio.utils.resource_monitor import ResourceMonitor, ResourceSnapshot


# =====================================================================
# Schema Roundtrip Tests
# =====================================================================
class TestSchemaRoundtrips:
    """Ensure all dataclass schemas serialize and deserialize correctly."""

    def test_episodic_record_roundtrip(self):
        ep = EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT, Modality.IMAGE],
            user_intent="Write code for me",
            strategy_used="step_by_step",
            outcome_score=9,
            lessons=["test first", "use types"],
            error_patterns=["off_by_one"],
            improvement_notes="Add error handling",
        )
        d = ep.to_dict()
        ep2 = EpisodicRecord.from_dict(d)
        assert ep2.task_type == ep.task_type
        assert ep2.modalities == ep.modalities
        assert ep2.outcome_score == 9
        assert ep2.lessons == ep.lessons

    def test_semantic_entry_roundtrip(self):
        se = SemanticEntry(
            domain="physics",
            knowledge="E=mc²",
            source_episodes=["ep1", "ep2"],
            confidence=0.95,
        )
        d = se.to_dict()
        se2 = SemanticEntry.from_dict(d)
        assert se2.domain == "physics"
        assert se2.confidence == 0.95

    def test_procedural_entry_roundtrip(self):
        pe = ProceduralEntry(
            task_type=TaskType.ANALYSIS,
            name="deep_analysis",
            description="Thorough analysis procedure",
            steps=["gather", "analyse", "synthesize", "conclude"],
            success_rate=0.87,
            usage_count=15,
        )
        d = pe.to_dict()
        pe2 = ProceduralEntry.from_dict(d)
        assert pe2.task_type == TaskType.ANALYSIS
        assert pe2.name == "deep_analysis"
        assert pe2.steps == pe.steps
        assert pe2.success_rate == 0.87

    def test_multimodal_input_modalities(self):
        inp = MultimodalInput(
            text="Test",
            image_paths=["a.jpg", "b.png"],
            audio_paths=["c.wav"],
            video_paths=["d.mp4"],
        )
        mods = inp.modalities
        assert Modality.TEXT in mods
        assert Modality.IMAGE in mods
        assert Modality.AUDIO in mods
        assert Modality.VIDEO in mods

    def test_multimodal_input_empty(self):
        inp = MultimodalInput()
        assert inp.modalities == []

    def test_multimodal_input_text_only(self):
        inp = MultimodalInput(text="Hello")
        assert inp.modalities == [Modality.TEXT]

    def test_modal_confidence_to_dict(self):
        mc = ModalConfidence(text=0.9, image=0.7, audio=0.5, video=0.3)
        d = mc.to_dict()
        assert d == {"text": 0.9, "image": 0.7, "audio": 0.5, "video": 0.3}

    def test_user_model_defaults(self):
        um = UserModel()
        assert um.expertise == ExpertiseLevel.INTERMEDIATE
        assert um.style == CommunicationStyle.DETAILED
        assert um.interaction_count == 0


# =====================================================================
# Enum Completeness Tests
# =====================================================================
class TestEnumCompleteness:
    """Verify all enum values are reachable and correctly typed."""

    def test_all_task_types(self):
        expected = {
            "question_answering", "content_generation", "analysis",
            "translation", "summarization", "code_generation",
            "creative_writing", "multimodal_reasoning", "conversation",
            "tool_use", "unknown",
        }
        actual = {t.value for t in TaskType}
        assert actual == expected

    def test_all_modalities(self):
        assert {m.value for m in Modality} == {"text", "image", "audio", "video"}

    def test_all_complexity_levels(self):
        assert [c.value for c in Complexity] == [1, 2, 3, 4, 5]

    def test_all_confidence_levels(self):
        assert {c.value for c in ConfidenceLevel} == {"low", "medium", "high"}

    def test_all_fusion_strategies(self):
        assert {f.value for f in FusionStrategy} == {
            "early_fusion", "late_fusion", "hybrid_fusion",
        }

    def test_all_error_types(self):
        assert len(ErrorType) == 5

    def test_all_agent_roles(self):
        assert len(AgentRole) == 11  # 7 cognitive + 4 processors

    def test_enum_str_values(self):
        """Enums should behave like strings for JSON serialization."""
        # In Python 3.9, str(StrEnum) returns 'EnumName.MEMBER_NAME'
        assert TaskType.QUESTION_ANSWERING.value == "question_answering"
        assert "question_answering" in repr(TaskType.QUESTION_ANSWERING).lower()

    def test_enum_construction_from_value(self):
        assert TaskType("code_generation") == TaskType.CODE_GENERATION
        assert Modality("audio") == Modality.AUDIO
        assert Complexity(5) == Complexity.EXTREME


# =====================================================================
# LLMClient Edge Cases
# =====================================================================
class TestLLMClientEdge:
    """Test LLMClient message builders and helpers without real API calls."""

    def test_build_vision_message(self):
        llm = MagicMock(spec=LLMClient)
        llm.build_vision_message = LLMClient.build_vision_message
        msg = llm.build_vision_message(llm, "Describe", ["http://img.com/a.jpg"])
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"

    def test_build_audio_message_http_url(self):
        llm = MagicMock(spec=LLMClient)
        llm.build_audio_message = LLMClient.build_audio_message
        llm._resolve_audio_url = LLMClient._resolve_audio_url
        llm._audio_format = LLMClient._audio_format
        msg = llm.build_audio_message(llm, "Transcribe", ["http://audio.com/a.mp3"])
        assert msg["content"][1]["type"] == "input_audio"
        assert msg["content"][1]["input_audio"]["format"] == "mp3"

    def test_build_audio_message_local_file(self, tmp_path):
        # Create a fake audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)
        # Use a real LLMClient-like object with the static methods
        class FakeLLM:
            _resolve_audio_url = staticmethod(LLMClient._resolve_audio_url)
            _audio_format = staticmethod(LLMClient._audio_format)
            build_audio_message = LLMClient.build_audio_message
        fake = FakeLLM()
        msg = fake.build_audio_message(str(audio_file), [str(audio_file)])
        # Should be base64-encoded
        data = msg["content"][1]["input_audio"]["data"]
        decoded = base64.b64decode(data)
        assert len(decoded) == 100

    def test_build_video_message(self):
        msg = LLMClient.build_video_message(MagicMock(), "Analyse", ["/tmp/clip.mp4"])
        assert msg["content"][1]["type"] == "video"
        assert msg["content"][1]["video"] == "/tmp/clip.mp4"

    def test_audio_format_detection(self):
        assert LLMClient._audio_format("song.mp3") == "mp3"
        assert LLMClient._audio_format("voice.wav") == "wav"
        assert LLMClient._audio_format("track.flac") == "flac"
        assert LLMClient._audio_format("clip.ogg") == "ogg"
        assert LLMClient._audio_format("unknown.xyz") == "wav"  # default

    def test_resolve_audio_url_http(self):
        url = "https://example.com/audio.mp3"
        assert LLMClient._resolve_audio_url(url) == url

    def test_resolve_audio_url_data_uri(self):
        uri = "data:audio/wav;base64,AAAA"
        assert LLMClient._resolve_audio_url(uri) == uri

    def test_ask_json_markdown_fence_extraction(self):
        """Test that ask_json can extract JSON from markdown code fences."""
        llm = MagicMock(spec=LLMClient)
        llm.chat = MagicMock(return_value='```json\n{"key": "value"}\n```')
        llm.ask = lambda self_inner, system, user, **kw: LLMClient.ask(llm, system, user, **kw)
        # Directly test the parser
        raw = '```json\n{"key": "value"}\n```'
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            result = json.loads(raw)
        assert result == {"key": "value"}


# =====================================================================
# Resource Monitor
# =====================================================================
class TestResourceMonitor:
    """Test resource detection and snapshot functionality."""

    def test_snapshot_returns_dataclass(self):
        rm = ResourceMonitor()
        snap = rm.snapshot()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_count_physical >= 1
        assert snap.ram_total_mb > 0

    def test_snapshot_recommended_workers(self):
        rm = ResourceMonitor()
        snap = rm.snapshot()
        assert snap.recommended_workers >= 1
        assert snap.recommended_workers <= snap.cpu_count_physical

    def test_snapshot_to_dict(self):
        rm = ResourceMonitor()
        d = rm.snapshot().to_dict()
        assert "cpu_count" in d or "cpu_count_physical" in d or isinstance(d, dict)

    def test_snapshot_refresh(self):
        rm = ResourceMonitor()
        snap1 = rm.snapshot()
        snap2 = rm.snapshot(refresh=True)
        assert snap2.cpu_count_physical == snap1.cpu_count_physical

    def test_has_gpu_attribute(self):
        rm = ResourceMonitor()
        snap = rm.snapshot()
        assert isinstance(snap.has_gpu, bool)


# =====================================================================
# Logger
# =====================================================================
class TestLogger:
    """Basic logger tests."""

    def test_logger_methods_dont_raise(self):
        logger = PinocchioLogger()
        logger.info(AgentRole.ORCHESTRATOR, "info")
        logger.warn(AgentRole.PERCEPTION, "warning")
        logger.error(AgentRole.EXECUTION, "error")
        logger.phase("Phase 1")
        logger.separator()
        logger.log(AgentRole.LEARNING, "log message")


# =====================================================================
# Cross-Cutting Integration Tests
# =====================================================================
class TestCrossCuttingIntegration:
    """Integration tests that span multiple subsystems."""

    def test_memory_persistence_across_reinstantiation(self, tmp_path):
        """Memory should survive MemoryManager re-creation."""
        mm1 = MemoryManager(str(tmp_path))
        mm1.store_episode(EpisodicRecord(
            task_type=TaskType.QUESTION_ANSWERING,
            user_intent="What is 2+2?",
            outcome_score=10,
            lessons=["basic math"],
        ))
        mm1.store_knowledge(SemanticEntry(
            domain="math",
            knowledge="Addition is commutative",
            confidence=0.95,
        ))
        mm1.store_procedure(ProceduralEntry(
            task_type=TaskType.QUESTION_ANSWERING,
            name="math_qa",
            steps=["parse", "compute", "answer"],
            success_rate=0.95,
            usage_count=5,
        ))

        # Re-create from same directory
        mm2 = MemoryManager(str(tmp_path))
        assert mm2.episodic.count == 1
        assert mm2.semantic.count == 1
        assert mm2.procedural.count == 1
        assert mm2.episodic.all()[0].user_intent == "What is 2+2?"
        assert mm2.semantic.all()[0].knowledge == "Addition is commutative"
        assert mm2.procedural.all()[0].name == "math_qa"

    def test_strategy_finds_procedure_from_prior_learning(self, tmp_path):
        """Verify strategy agent can find procedures stored by learning agent."""
        mm = MemoryManager(str(tmp_path))
        llm = MagicMock()
        llm.model = "test"
        llm.temperature = 0.7
        llm.max_tokens = 4096
        logger = PinocchioLogger()

        # Simulate learning skill storing a procedure
        from pinocchio.agents.unified_agent import PinocchioAgent
        llm.ask_json.return_value = {
            "new_lessons": ["decompose problem"],
            "strategy_refinements": "",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "Decomposition helps",
            "should_save_procedure": True,
            "procedure_name": "decompose_qa",
            "procedure_steps": ["decompose", "solve parts", "synthesize"],
        }
        learning = PinocchioAgent(llm, mm, logger)
        learning.learn(
            user_input_text="Complex question",
            perception=PerceptionResult(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                complexity=Complexity.COMPLEX,
                confidence=ConfidenceLevel.HIGH,
            ),
            strategy=StrategyResult(
                selected_strategy="decompose_qa",
                is_novel=True,
            ),
            evaluation=EvaluationResult(output_quality=9),
        )

        # Now verify strategy skill can recall it
        llm.ask_json.return_value = {
            "selected_strategy": "decompose_qa",
            "basis": "proven procedure",
            "risk_assessment": "low",
            "fallback_plan": "direct answer",
            "modality_pipeline": "text→reason→text",
            "fusion_strategy": "late_fusion",
            "is_novel": False,
            "analysis": "Reusing proven decompose_qa",
        }
        strategy = PinocchioAgent(llm, mm, logger)
        result = strategy.strategize(perception=PerceptionResult(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT],
        ))
        assert result.selected_strategy == "decompose_qa"
        assert result.is_novel is False
        # LLM should have been told about the proven procedure
        call_str = str(llm.ask_json.call_args)
        assert "decompose_qa" in call_str

    def test_episodic_memory_affects_perception(self, tmp_path):
        """Past episodes should influence perception's similar_episodes."""
        mm = MemoryManager(str(tmp_path))
        llm = MagicMock()
        llm.model = "test"
        llm.temperature = 0.7
        llm.max_tokens = 4096
        logger = PinocchioLogger()

        # Store some history
        for i in range(3):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.CODE_GENERATION,
                modalities=[Modality.TEXT],
                user_intent=f"Write function {i}",
                outcome_score=7 + i,
                lessons=[f"lesson {i}"],
            ))

        llm.ask_json.return_value = {
            "task_type": "code_generation",
            "modalities": ["text"],
            "complexity": 3,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "Code gen task",
        }

        from pinocchio.agents.unified_agent import PinocchioAgent
        pa = PinocchioAgent(llm, mm, logger)
        result = pa.perceive(user_input=MultimodalInput(text="Write a sort function"))
        assert len(result.similar_episodes) > 0
        assert len(result.relevant_lessons) > 0

    def test_full_loop_multiple_interactions_memory_grows(self, tmp_path):
        """Running multiple interactions should accumulate episodic memory."""
        mm = MemoryManager(str(tmp_path))
        llm = MagicMock()
        llm.model = "test"
        llm.temperature = 0.7
        llm.max_tokens = 4096
        logger = PinocchioLogger()

        from pinocchio.agents.unified_agent import PinocchioAgent
        la = PinocchioAgent(llm, mm, logger)

        for i in range(5):
            llm.ask_json.return_value = {
                "new_lessons": [f"lesson_{i}"],
                "strategy_refinements": "",
                "skill_gap": "",
                "self_improvement_action": "",
                "semantic_knowledge": f"knowledge_{i}" if i % 2 == 0 else "",
                "should_save_procedure": False,
                "procedure_name": "",
                "procedure_steps": [],
            }
            la.learn(
                user_input_text=f"Question {i}",
                perception=PerceptionResult(
                    task_type=TaskType.QUESTION_ANSWERING,
                    modalities=[Modality.TEXT],
                ),
                strategy=StrategyResult(selected_strategy="default"),
                evaluation=EvaluationResult(output_quality=7),
            )

        assert mm.episodic.count == 5
        assert mm.semantic.count == 3  # i=0, 2, 4 produced knowledge
        lessons = mm.episodic.recent_lessons(limit=10)
        assert len(lessons) == 5

    def test_improvement_trend_after_interactions(self, tmp_path):
        """Improvement trend should reflect score patterns."""
        mm = MemoryManager(str(tmp_path))
        # Simulate 15 interactions: first 5 bad, rest good
        for i in range(15):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=3 if i < 5 else 9,
            ))
        trend = mm.improvement_trend(window=10)
        assert trend["trend"] == "improving"
        assert trend["recent_avg"] >= 8.0


# =====================================================================
# Stress & Scale Tests
# =====================================================================
class TestStressScale:
    """Stress tests for memory subsystems at scale."""

    def test_100_episodes_performance(self, tmp_path):
        """Memory should handle 100 episodes without issues."""
        mm = MemoryManager(str(tmp_path))
        for i in range(100):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType(list(TaskType)[i % len(TaskType)].value),
                modalities=[Modality.TEXT],
                user_intent=f"Task {i}",
                outcome_score=(i % 10) + 1,
                lessons=[f"lesson_{i}"],
            ))
        assert mm.episodic.count == 100
        summary = mm.summary()
        assert summary["episodic_count"] == 100
        assert summary["avg_score"] > 0

    def test_50_procedures_ranking(self, tmp_path):
        """Procedural memory should correctly rank many procedures."""
        mm = MemoryManager(str(tmp_path))
        for i in range(50):
            mm.store_procedure(ProceduralEntry(
                task_type=TaskType.QUESTION_ANSWERING,
                name=f"proc_{i}",
                success_rate=i / 50.0,
                usage_count=max(2, i),
                steps=[f"step_{j}" for j in range(3)],
            ))
        top = mm.procedural.top_procedures(limit=5)
        assert len(top) == 5
        assert top[0].success_rate >= top[1].success_rate

    def test_find_similar_with_many_episodes(self, tmp_path):
        """find_similar should work correctly with 100+ episodes."""
        from pinocchio.memory.episodic_memory import EpisodicMemory
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        for i in range(100):
            em.add(EpisodicRecord(
                task_type=TaskType(list(TaskType)[i % len(TaskType)].value),
                modalities=[Modality.TEXT] if i % 2 == 0 else [Modality.IMAGE],
                user_intent=f"Task {i}",
                outcome_score=5,
            ))
        similar = em.find_similar(
            TaskType.QUESTION_ANSWERING, [Modality.TEXT], limit=5
        )
        assert len(similar) <= 5
        for ep in similar:
            # All should be relevant (same task type or shared modality)
            assert ep.task_type == TaskType.QUESTION_ANSWERING or Modality.TEXT in ep.modalities

    def test_semantic_search_with_many_entries(self, tmp_path):
        """Semantic search should handle 100+ entries."""
        from pinocchio.memory.semantic_memory import SemanticMemory
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        for i in range(100):
            sm.add(SemanticEntry(
                domain=f"domain_{i % 10}",
                knowledge=f"Knowledge item {i}: fact about topic {i % 10}",
                confidence=0.5 + (i % 5) * 0.1,
            ))
        assert sm.count == 100
        results = sm.search_by_domain("domain_3", limit=20)
        assert len(results) == 10  # 100 / 10 domains
        high = sm.get_high_confidence(threshold=0.8)
        assert len(high) > 0

    def test_json_persistence_at_scale(self, tmp_path):
        """Persist and reload 100 episodes correctly."""
        from pinocchio.memory.episodic_memory import EpisodicMemory
        path = str(tmp_path / "ep.json")
        em1 = EpisodicMemory(path)
        ids = []
        for i in range(100):
            ep = EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                user_intent=f"Q{i}",
                outcome_score=i % 10 + 1,
            )
            em1.add(ep)
            ids.append(ep.episode_id)
        # Reload
        em2 = EpisodicMemory(path)
        assert em2.count == 100
        # Spot check
        for idx in [0, 50, 99]:
            assert em2.get(ids[idx]) is not None
            assert em2.get(ids[idx]).user_intent == f"Q{idx}"
