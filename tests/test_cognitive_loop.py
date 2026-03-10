"""Comprehensive tests for all six cognitive-loop agents.

Covers gaps identified in the audit: error handling, edge cases,
multimodal inputs, malformed LLM output, branch coverage for
BaseAgent, PerceptionAgent, StrategyAgent, ExecutionAgent,
EvaluationAgent, LearningAgent, MetaReflectionAgent.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.agents.unified_agent import PinocchioAgent
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import (
    AgentRole, Complexity, ConfidenceLevel, FusionStrategy,
    Modality, TaskType,
)
from pinocchio.models.schemas import (
    AgentMessage, EpisodicRecord, EvaluationResult,
    LearningResult, MetaReflectionResult, PerceptionResult,
    StrategyResult, MultimodalInput,
)
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger


# ── helpers ──────────────────────────────────────────────────────────
def _mock_infra(tmp_path):
    """Return (llm, memory, logger) mocks suitable for agent construction."""
    llm = MagicMock(spec=LLMClient)
    llm.model = "test-model"
    llm.temperature = 0.7
    llm.max_tokens = 4096
    memory = MemoryManager(data_dir=str(tmp_path))
    logger = PinocchioLogger()
    return llm, memory, logger


def _perception(**overrides) -> PerceptionResult:
    defaults = dict(
        task_type=TaskType.QUESTION_ANSWERING,
        modalities=[Modality.TEXT],
        complexity=Complexity.MODERATE,
        confidence=ConfidenceLevel.HIGH,
        ambiguities=[],
        relevant_lessons=[],
        similar_episodes=[],
        raw_analysis="Test perception analysis.",
    )
    defaults.update(overrides)
    return PerceptionResult(**defaults)


def _strategy(**overrides) -> StrategyResult:
    defaults = dict(
        selected_strategy="direct_answer",
        basis="first principles",
        risk_assessment="low",
        fallback_plan="simplify",
        modality_pipeline="text→reasoning→text",
        fusion_strategy=FusionStrategy.LATE_FUSION,
        is_novel=False,
        raw_analysis="Default strategy.",
    )
    defaults.update(overrides)
    return StrategyResult(**defaults)


def _evaluation(**overrides) -> EvaluationResult:
    defaults = dict(
        output_quality=8,
        strategy_effectiveness=7,
        cross_modal_coherence=8,
        went_well=["clear answer"],
        went_wrong=[],
        surprises=[],
        task_completion="complete",
        raw_analysis="Good.",
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# =====================================================================
# BaseAgent
# =====================================================================
class TestBaseAgent:
    """Tests for the abstract BaseAgent class."""

    def test_cannot_instantiate_directly(self):
        llm = MagicMock()
        memory = MagicMock()
        logger = MagicMock()
        with pytest.raises(TypeError):
            BaseAgent(llm, memory, logger)

    def test_subclass_without_run_raises(self):
        with pytest.raises(TypeError):
            class Incomplete(BaseAgent):
                role = AgentRole.PERCEPTION
            Incomplete(MagicMock(), MagicMock(), MagicMock())

    def test_log_methods_delegate_to_logger(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        agent = PinocchioAgent(llm, memory, logger)
        # These should not raise
        agent._log("info message")
        agent._warn("warning")
        agent._error("error")


# =====================================================================
# Perception Skill
# =====================================================================
class TestPerceptionSkillComprehensive:
    """Extended perception tests covering modalities, edge cases, and error handling."""

    def _make_agent(self, tmp_path, json_response):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = json_response
        return PinocchioAgent(llm, memory, logger)

    def test_audio_modality_detected(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_type": "question_answering",
            "modalities": ["audio"],
            "complexity": 3,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "Audio QA",
        })
        inp = MultimodalInput(audio_paths=["test.wav"])
        result = agent.perceive(user_input=inp)
        assert Modality.AUDIO in result.modalities

    def test_video_modality_detected(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_type": "multimodal_reasoning",
            "modalities": ["video", "text"],
            "complexity": 4,
            "confidence": "medium",
            "ambiguities": ["unclear context"],
            "analysis": "Video analysis",
        })
        inp = MultimodalInput(text="Describe", video_paths=["clip.mp4"])
        result = agent.perceive(user_input=inp)
        assert Modality.VIDEO in result.modalities
        assert result.ambiguities == ["unclear context"]

    def test_empty_text_input(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_type": "unknown",
            "modalities": ["image"],
            "complexity": 1,
            "confidence": "low",
            "ambiguities": ["no text"],
            "analysis": "Image only",
        })
        inp = MultimodalInput(image_paths=["img.jpg"])
        result = agent.perceive(user_input=inp)
        assert result.task_type == TaskType.UNKNOWN

    def test_all_four_modalities(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_type": "multimodal_reasoning",
            "modalities": ["text", "image", "audio", "video"],
            "complexity": 5,
            "confidence": "low",
            "ambiguities": [],
            "analysis": "Full multimodal",
        })
        inp = MultimodalInput(
            text="Analyze",
            image_paths=["a.jpg"],
            audio_paths=["b.wav"],
            video_paths=["c.mp4"],
        )
        result = agent.perceive(user_input=inp)
        assert len(result.modalities) == 4
        assert result.complexity == Complexity.EXTREME

    def test_perception_with_similar_episodes(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        # Pre-populate memory
        ep = EpisodicRecord(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT],
            user_intent="old question",
            strategy_used="direct",
            outcome_score=9,
            lessons=["be concise"],
        )
        memory.store_episode(ep)
        llm.ask_json.return_value = {
            "task_type": "question_answering",
            "modalities": ["text"],
            "complexity": 2,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "Similar",
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.perceive(user_input=MultimodalInput(text="new question"))
        assert len(result.similar_episodes) >= 1
        assert len(result.relevant_lessons) >= 1

    def test_perception_with_modality_context(self, tmp_path):
        """Modality context kwargs should not break perception."""
        agent = self._make_agent(tmp_path, {
            "task_type": "question_answering",
            "modalities": ["text"],
            "complexity": 2,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "With context",
        })
        result = agent.perceive(
            user_input=MultimodalInput(text="hello"),
            modality_context={"vision": "A cat on a table"},
        )
        assert result.task_type == TaskType.QUESTION_ANSWERING

    def test_malformed_json_missing_fields(self, tmp_path):
        """Perception should handle missing JSON fields gracefully."""
        agent = self._make_agent(tmp_path, {
            "task_type": "question_answering",
            # missing modalities, complexity, confidence, etc.
        })
        inp = MultimodalInput(text="test")
        result = agent.perceive(user_input=inp)
        # Should still return a PerceptionResult (with defaults)
        assert isinstance(result, PerceptionResult)


# =====================================================================
# Strategy Skill
# =====================================================================
class TestStrategySkillComprehensive:
    """Extended strategy tests: ambiguities, lessons, fallbacks, fusion."""

    def _make_agent(self, tmp_path, json_response):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = json_response
        return PinocchioAgent(llm, memory, logger)

    def test_with_ambiguities_in_perception(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "selected_strategy": "clarify_then_answer",
            "basis": "ambiguities detected",
            "risk_assessment": "medium",
            "fallback_plan": "answer with caveats",
            "modality_pipeline": "text→reasoning→text",
            "fusion_strategy": "late_fusion",
            "is_novel": True,
            "analysis": "Handle ambiguity carefully.",
        })
        p = _perception(ambiguities=["unclear scope", "vague terms"])
        result = agent.strategize(perception=p)
        assert result.selected_strategy == "clarify_then_answer"
        assert result.is_novel is True

    def test_with_relevant_lessons(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "selected_strategy": "lesson_guided",
            "basis": "lessons applied",
            "risk_assessment": "low",
            "fallback_plan": "default",
            "modality_pipeline": "text→reasoning→text",
            "fusion_strategy": "early_fusion",
            "is_novel": False,
            "analysis": "Using lessons.",
        })
        p = _perception(relevant_lessons=["use analogies", "keep it simple"])
        result = agent.strategize(perception=p)
        assert result.fusion_strategy == FusionStrategy.EARLY_FUSION

    def test_hybrid_fusion_strategy(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "selected_strategy": "multimodal",
            "basis": "complex input",
            "risk_assessment": "high",
            "fallback_plan": "text only",
            "modality_pipeline": "image→caption→audio→fuse→reason",
            "fusion_strategy": "hybrid_fusion",
            "is_novel": True,
            "analysis": "Hybrid approach.",
        })
        p = _perception(
            modalities=[Modality.TEXT, Modality.IMAGE, Modality.AUDIO],
            complexity=Complexity.EXTREME,
        )
        result = agent.strategize(perception=p)
        assert result.fusion_strategy == FusionStrategy.HYBRID_FUSION

    def test_default_values_on_partial_json(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "selected_strategy": "partial",
            # everything else missing
        })
        result = agent.strategize(perception=_perception())
        assert result.selected_strategy == "partial"
        assert result.is_novel is True  # default when missing from JSON
        assert result.fallback_plan == ""  # default

    def test_reuses_high_success_procedure(self, tmp_path):
        """When procedural memory has a proven procedure, strategy should reference it."""
        from pinocchio.models.schemas import ProceduralEntry
        llm, memory, logger = _mock_infra(tmp_path)
        proc = ProceduralEntry(
            task_type=TaskType.QUESTION_ANSWERING,
            name="proven_qa",
            description="A tested QA flow",
            steps=["understand", "reason", "answer"],
            success_rate=0.95,
            usage_count=10,
        )
        memory.store_procedure(proc)
        llm.ask_json.return_value = {
            "selected_strategy": "proven_qa",
            "basis": "high success rate procedure",
            "risk_assessment": "low",
            "fallback_plan": "simplify",
            "modality_pipeline": "text→reasoning→text",
            "fusion_strategy": "late_fusion",
            "is_novel": False,
            "analysis": "Reuse proven.",
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.strategize(perception=_perception())
        assert result.selected_strategy == "proven_qa"
        # Verify LLM was called with procedure context
        call_args = llm.ask_json.call_args
        assert "proven_qa" in call_args[1].get("user", "") or "proven_qa" in str(call_args)


# =====================================================================
# Execution Skill
# =====================================================================
class TestExecutionSkillComprehensive:
    """Extended execution tests: audio, video, modality_context, errors."""

    def test_execution_with_modality_context(self, tmp_path):
        """Modality context should be passed but not break execution."""
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask.return_value = "Based on the image, I see a landscape."
        agent = PinocchioAgent(llm, memory, logger)
        inp = MultimodalInput(text="Describe what you see")
        result = agent.execute(
            user_input=inp,
            perception=_perception(),
            strategy=_strategy(),
            modality_context={"vision": "A landscape with mountains"},
        )
        assert isinstance(result, AgentMessage)
        assert len(result.content) > 0

    def test_execution_novel_strategy_lower_confidence(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask.return_value = "Experimental response."
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.execute(
            user_input=MultimodalInput(text="test"),
            perception=_perception(),
            strategy=_strategy(is_novel=True),
        )
        assert result.confidence == 0.8

    def test_execution_proven_strategy_higher_confidence(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask.return_value = "Proven response."
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.execute(
            user_input=MultimodalInput(text="test"),
            perception=_perception(),
            strategy=_strategy(is_novel=False),
        )
        assert result.confidence == 0.9

    def test_execution_with_images_uses_vision_path(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.build_vision_message.return_value = {"role": "user", "content": []}
        llm.chat.return_value = "I see objects in the image."
        agent = PinocchioAgent(llm, memory, logger)
        # Create a real tiny PNG so _encode_image can read it
        img_file = tmp_path / "test.png"
        img_file.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        inp = MultimodalInput(text="Describe", image_paths=[str(img_file)])
        result = agent.execute(
            user_input=inp,
            perception=_perception(modalities=[Modality.TEXT, Modality.IMAGE]),
            strategy=_strategy(),
        )
        llm.build_vision_message.assert_called_once()
        assert "objects" in result.content

    def test_execution_metadata_contains_strategy(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask.return_value = "Response."
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.execute(
            user_input=MultimodalInput(text="test"),
            perception=_perception(),
            strategy=_strategy(selected_strategy="my_strategy"),
        )
        assert result.metadata["strategy"] == "my_strategy"


# =====================================================================
# Evaluation Skill
# =====================================================================
class TestEvaluationSkillComprehensive:
    """Extended evaluation tests: partial completion, multimodal, defaults."""

    def _make_agent(self, tmp_path, json_response):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = json_response
        return PinocchioAgent(llm, memory, logger)

    def test_partial_completion(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_completion": "partial",
            "output_quality": 5,
            "strategy_effectiveness": 4,
            "went_well": ["started well"],
            "went_wrong": ["ran out of context"],
            "surprises": ["unexpected complexity"],
            "cross_modal_coherence": 3,
            "analysis": "Partial.",
        })
        result = agent.evaluate(
            user_input=MultimodalInput(text="complex question"),
            perception=_perception(complexity=Complexity.EXTREME),
            strategy=_strategy(),
            response=AgentMessage(content="Half answer"),
        )
        assert result.task_completion == "partial"
        assert result.output_quality == 5

    def test_failure_detection(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_completion": "failed",
            "output_quality": 2,
            "strategy_effectiveness": 1,
            "went_well": [],
            "went_wrong": ["completely wrong approach", "hallucination"],
            "surprises": [],
            "cross_modal_coherence": 1,
            "analysis": "Failed.",
        })
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=_perception(),
            strategy=_strategy(),
            response=AgentMessage(content="Wrong answer"),
        )
        assert result.output_quality <= 3
        assert len(result.went_wrong) == 2

    def test_high_multimodal_coherence(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "task_completion": "complete",
            "output_quality": 9,
            "strategy_effectiveness": 9,
            "went_well": ["perfect fusion"],
            "went_wrong": [],
            "surprises": [],
            "cross_modal_coherence": 10,
            "analysis": "Excellent multimodal.",
        })
        result = agent.evaluate(
            user_input=MultimodalInput(text="Describe", image_paths=["a.jpg"]),
            perception=_perception(modalities=[Modality.TEXT, Modality.IMAGE]),
            strategy=_strategy(fusion_strategy=FusionStrategy.EARLY_FUSION),
            response=AgentMessage(content="Detailed description"),
        )
        assert result.cross_modal_coherence == 10

    def test_default_values_on_missing_json_keys(self, tmp_path):
        agent = self._make_agent(tmp_path, {
            "output_quality": 6,
            "is_complete": True,
            # everything else missing
        })
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=_perception(),
            strategy=_strategy(),
            response=AgentMessage(content="This is a complete answer to the question asked by the user."),
        )
        assert result.output_quality == 6
        assert isinstance(result.went_well, list)


# =====================================================================
# Learning Skill
# =====================================================================
class TestLearningSkillComprehensive:
    """Extended learning tests: procedure updates, synthesis, edge cases."""

    def test_high_quality_saves_procedure(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = {
            "new_lessons": ["lesson A"],
            "strategy_refinements": "none needed",
            "skill_gap": "",
            "self_improvement_action": "practice more",
            "semantic_knowledge": "QA benefit from analogies",
            "should_save_procedure": True,
            "procedure_name": "analogy_qa",
            "procedure_steps": ["understand", "find analogy", "explain"],
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.learn(
            user_input_text="What is X?",
            perception=_perception(),
            strategy=_strategy(selected_strategy="analogy_qa"),
            evaluation=_evaluation(output_quality=9),
        )
        assert "analogy_qa" in result.procedural_updates
        assert memory.procedural.count >= 1
        assert memory.semantic.count >= 1

    def test_low_quality_skips_procedure(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = {
            "new_lessons": ["approach was wrong"],
            "strategy_refinements": "try different approach",
            "skill_gap": "domain knowledge",
            "self_improvement_action": "study topic",
            "semantic_knowledge": "",
            "should_save_procedure": True,  # wants to save but quality too low
            "procedure_name": "bad_proc",
            "procedure_steps": ["step1"],
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.learn(
            user_input_text="test",
            perception=_perception(),
            strategy=_strategy(),
            evaluation=_evaluation(output_quality=4),
        )
        assert result.procedural_updates == []
        assert result.skill_gap == "domain knowledge"

    def test_existing_procedure_usage_updated(self, tmp_path):
        """When should_save_procedure=False and a procedure exists, update its usage."""
        from pinocchio.models.schemas import ProceduralEntry
        llm, memory, logger = _mock_infra(tmp_path)
        proc = ProceduralEntry(
            task_type=TaskType.QUESTION_ANSWERING,
            name="existing_qa",
            description="Test",
            steps=["s1", "s2"],
            success_rate=0.8,
            usage_count=5,
        )
        memory.store_procedure(proc)
        llm.ask_json.return_value = {
            "new_lessons": [],
            "strategy_refinements": "",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "",
            "should_save_procedure": False,
            "procedure_name": "",
            "procedure_steps": [],
        }
        agent = PinocchioAgent(llm, memory, logger)
        agent.learn(
            user_input_text="test",
            perception=_perception(),
            strategy=_strategy(),
            evaluation=_evaluation(output_quality=8),
        )
        updated = memory.procedural.get(proc.entry_id)
        assert updated.usage_count == 6  # incremented

    def test_empty_lessons_handled(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = {
            "new_lessons": [],
            "strategy_refinements": "",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "",
            "should_save_procedure": False,
            "procedure_name": "",
            "procedure_steps": [],
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.learn(
            user_input_text="test",
            perception=_perception(),
            strategy=_strategy(),
            evaluation=_evaluation(),
        )
        assert result.new_lessons == []
        assert isinstance(result.episodic_update, str)

    def test_semantic_knowledge_stored_with_confidence(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = {
            "new_lessons": ["key insight"],
            "strategy_refinements": "",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "Analogies work well for technical concepts",
            "should_save_procedure": False,
            "procedure_name": "",
            "procedure_steps": [],
        }
        agent = PinocchioAgent(llm, memory, logger)
        agent.learn(
            user_input_text="test",
            perception=_perception(),
            strategy=_strategy(),
            evaluation=_evaluation(output_quality=9),
        )
        entries = memory.semantic.all()
        assert len(entries) == 1
        assert entries[0].confidence == 0.9  # output_quality / 10


# =====================================================================
# Meta-Reflect Skill
# =====================================================================
class TestMetaReflectSkillComprehensive:
    """Extended meta-reflection tests: trigger logic, empty memory, content."""

    def test_trigger_at_zero_episodes_returns_false(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        agent = PinocchioAgent(llm, memory, logger)
        assert agent.should_meta_reflect() is False

    def test_trigger_at_5_episodes(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        for _ in range(5):
            memory.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                user_intent="q",
                strategy_used="s",
                outcome_score=7,
            ))
        agent = PinocchioAgent(llm, memory, logger)
        assert agent.should_meta_reflect() is True

    def test_trigger_at_7_episodes_returns_false(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        for _ in range(7):
            memory.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                user_intent="q",
                strategy_used="s",
                outcome_score=7,
            ))
        agent = PinocchioAgent(llm, memory, logger)
        assert agent.should_meta_reflect() is False

    def test_trigger_at_10_episodes(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        for _ in range(10):
            memory.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                user_intent="q",
                strategy_used="s",
                outcome_score=7,
            ))
        agent = PinocchioAgent(llm, memory, logger)
        assert agent.should_meta_reflect() is True

    def test_run_with_populated_memory(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        for i in range(5):
            memory.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                user_intent=f"question {i}",
                strategy_used="default",
                outcome_score=6 + i,
                lessons=[f"lesson_{i}"],
                error_patterns=["timeout"] if i == 0 else [],
            ))
        llm.ask_json.return_value = {
            "recurring_errors": ["timeout"],
            "strength_domains": ["question_answering"],
            "weakness_domains": ["creative_writing"],
            "strategy_trajectory": "improving",
            "bias_check": "none detected",
            "learning_efficiency": "good",
            "priority_improvements": ["creative_writing", "multimodal"],
            "experimental_strategies": ["try chain-of-thought"],
            "knowledge_gaps": ["video understanding"],
            "analysis": "Steady improvement across QA tasks.",
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.meta_reflect()
        assert isinstance(result, MetaReflectionResult)
        assert "timeout" in result.recurring_errors
        assert "creative_writing" in result.priority_improvements
        assert "chain-of-thought" in result.experimental_strategies[0]

    def test_run_with_empty_memory(self, tmp_path):
        llm, memory, logger = _mock_infra(tmp_path)
        llm.ask_json.return_value = {
            "recurring_errors": [],
            "strength_domains": [],
            "weakness_domains": [],
            "strategy_trajectory": "no data",
            "bias_check": "insufficient data",
            "learning_efficiency": "unknown",
            "priority_improvements": [],
            "experimental_strategies": [],
            "knowledge_gaps": [],
            "analysis": "No data yet.",
        }
        agent = PinocchioAgent(llm, memory, logger)
        result = agent.meta_reflect()
        assert result.recurring_errors == []
        assert result.raw_analysis == "No data yet."
