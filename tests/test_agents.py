"""Tests for all sub-agents with mocked LLM calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinocchio.agents.perception_agent import PerceptionAgent
from pinocchio.agents.strategy_agent import StrategyAgent
from pinocchio.agents.execution_agent import ExecutionAgent
from pinocchio.agents.evaluation_agent import EvaluationAgent
from pinocchio.agents.learning_agent import LearningAgent
from pinocchio.agents.meta_reflection_agent import MetaReflectionAgent, _DEFAULT_META_REFLECT_INTERVAL
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import (
    AgentRole,
    Modality,
    TaskType,
    Complexity,
    ConfidenceLevel,
    FusionStrategy,
)
from pinocchio.models.schemas import (
    MultimodalInput,
    PerceptionResult,
    StrategyResult,
    EvaluationResult,
    AgentMessage,
    EpisodicRecord,
)
from pinocchio.utils.logger import PinocchioLogger


@pytest.fixture
def logger():
    return PinocchioLogger()


@pytest.fixture
def memory(tmp_data_dir):
    return MemoryManager(data_dir=tmp_data_dir)


# ──────────────────────────────────────────────────────────────────
# PerceptionAgent
# ──────────────────────────────────────────────────────────────────


class TestPerceptionAgent:
    def test_role(self, mock_llm, memory, logger):
        agent = PerceptionAgent(mock_llm, memory, logger)
        assert agent.role == AgentRole.PERCEPTION

    def test_run_classifies_task(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "task_type": "code_generation",
            "complexity": 3,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "User wants code.",
        }
        agent = PerceptionAgent(mock_llm, memory, logger)
        user_input = MultimodalInput(text="Write a Python function to sort a list")
        result = agent.run(user_input=user_input)

        assert result.task_type == TaskType.CODE_GENERATION
        assert result.complexity == Complexity.MODERATE
        assert result.confidence == ConfidenceLevel.HIGH
        assert Modality.TEXT in result.modalities
        assert result.raw_analysis == "User wants code."
        mock_llm.ask_json.assert_called_once()

    def test_run_with_image_modality(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "task_type": "multimodal_reasoning",
            "complexity": 4,
            "confidence": "medium",
            "ambiguities": ["unclear which part to focus on"],
            "analysis": "Multimodal image analysis requested.",
        }
        agent = PerceptionAgent(mock_llm, memory, logger)
        user_input = MultimodalInput(text="What's in this photo?", image_paths=["photo.jpg"])
        result = agent.run(user_input=user_input)

        assert Modality.TEXT in result.modalities
        assert Modality.IMAGE in result.modalities
        assert len(result.ambiguities) == 1

    def test_run_retrieves_similar_episodes(self, mock_llm, memory, logger):
        # Seed an episode
        memory.store_episode(EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT],
            lessons=["Use type hints"],
        ))
        mock_llm.ask_json.return_value = {
            "task_type": "code_generation",
            "complexity": 2,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "Simple code request.",
        }
        agent = PerceptionAgent(mock_llm, memory, logger)
        result = agent.run(user_input=MultimodalInput(text="Write hello world"))

        assert len(result.similar_episodes) >= 1
        assert "Use type hints" in result.relevant_lessons


# ──────────────────────────────────────────────────────────────────
# StrategyAgent
# ──────────────────────────────────────────────────────────────────


class TestStrategyAgent:
    def test_role(self, mock_llm, memory, logger):
        agent = StrategyAgent(mock_llm, memory, logger)
        assert agent.role == AgentRole.STRATEGY

    def test_run_produces_strategy(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "selected_strategy": "direct_qa",
            "basis": "first principles",
            "risk_assessment": "low risk",
            "fallback_plan": "simplify explanation",
            "modality_pipeline": "text→reasoning→text",
            "fusion_strategy": "late_fusion",
            "is_novel": True,
            "analysis": "Direct approach is best.",
        }
        agent = StrategyAgent(mock_llm, memory, logger)
        perception = PerceptionResult(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT],
            complexity=Complexity.SIMPLE,
        )
        result = agent.run(perception=perception)

        assert result.selected_strategy == "direct_qa"
        assert result.is_novel is True
        assert result.fusion_strategy == FusionStrategy.LATE_FUSION

    def test_run_reuses_proven_procedure(self, mock_llm, memory, logger):
        from pinocchio.models.schemas import ProceduralEntry
        memory.store_procedure(ProceduralEntry(
            name="qa_proven_v1",
            task_type=TaskType.QUESTION_ANSWERING,
            success_rate=0.95,
            usage_count=20,
            steps=["parse question", "retrieve context", "answer"],
        ))
        mock_llm.ask_json.return_value = {
            "selected_strategy": "qa_proven_v1",
            "basis": "proven procedure from memory",
            "risk_assessment": "very low",
            "fallback_plan": "switch to general reasoning",
            "modality_pipeline": "text→reasoning→text",
            "fusion_strategy": "late_fusion",
            "is_novel": False,
            "analysis": "Reusing proven strategy.",
        }
        agent = StrategyAgent(mock_llm, memory, logger)
        perception = PerceptionResult(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT],
        )
        result = agent.run(perception=perception)

        assert result.is_novel is False
        # The LLM prompt should have included the procedure context
        call_args = mock_llm.ask_json.call_args
        user_prompt = call_args[1].get("user", "") or call_args[0][1] if len(call_args[0]) > 1 else ""
        assert "qa_proven_v1" in str(call_args)


# ──────────────────────────────────────────────────────────────────
# ExecutionAgent
# ──────────────────────────────────────────────────────────────────


class TestExecutionAgent:
    def test_role(self, mock_llm, memory, logger):
        agent = ExecutionAgent(mock_llm, memory, logger)
        assert agent.role == AgentRole.EXECUTION

    def test_run_text_only(self, mock_llm, memory, logger):
        mock_llm.ask.return_value = "The answer is 42."
        agent = ExecutionAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input=MultimodalInput(text="What is the answer?"),
            perception=PerceptionResult(task_type=TaskType.QUESTION_ANSWERING, modalities=[Modality.TEXT]),
            strategy=StrategyResult(selected_strategy="direct", is_novel=False),
        )
        assert isinstance(result, AgentMessage)
        assert result.content == "The answer is 42."
        assert result.confidence == 0.9  # not novel → higher confidence

    def test_run_with_images_uses_vision(self, mock_llm, memory, logger):
        mock_llm.chat.return_value = "I see a cat in the image."
        mock_llm.build_vision_message.return_value = {"role": "user", "content": []}
        agent = ExecutionAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input=MultimodalInput(text="What's in this?", image_paths=["cat.jpg"]),
            perception=PerceptionResult(modalities=[Modality.TEXT, Modality.IMAGE]),
            strategy=StrategyResult(selected_strategy="vision_qa", is_novel=True),
        )
        assert "cat" in result.content
        assert result.confidence == 0.8  # novel → lower confidence
        mock_llm.build_vision_message.assert_called_once()

    def test_run_includes_strategy_metadata(self, mock_llm, memory, logger):
        mock_llm.ask.return_value = "Response"
        agent = ExecutionAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(selected_strategy="test_strat", is_novel=True),
        )
        assert result.metadata["strategy"] == "test_strat"
        assert result.metadata["is_novel_strategy"] is True


# ──────────────────────────────────────────────────────────────────
# EvaluationAgent
# ──────────────────────────────────────────────────────────────────


class TestEvaluationAgent:
    def test_role(self, mock_llm, memory, logger):
        agent = EvaluationAgent(mock_llm, memory, logger)
        assert agent.role == AgentRole.EVALUATION

    def test_run_produces_evaluation(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "task_completion": "complete",
            "output_quality": 8,
            "strategy_effectiveness": 7,
            "went_well": ["accurate answer", "clear explanation"],
            "went_wrong": [],
            "surprises": ["user asked a follow-up"],
            "cross_modal_coherence": 5,
            "analysis": "Good quality output.",
        }
        agent = EvaluationAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(task_type=TaskType.QUESTION_ANSWERING),
            strategy=StrategyResult(selected_strategy="direct"),
            response=AgentMessage(content="The answer..."),
        )

        assert result.task_completion == "complete"
        assert result.output_quality == 8
        assert result.strategy_effectiveness == 7
        assert len(result.went_well) == 2
        assert result.user_satisfaction == "awaiting"

    def test_run_detects_failures(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "task_completion": "failed",
            "output_quality": 2,
            "strategy_effectiveness": 3,
            "went_well": [],
            "went_wrong": ["completely wrong answer", "missed the point"],
            "surprises": [],
            "cross_modal_coherence": 5,
            "analysis": "Failed badly.",
        }
        agent = EvaluationAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(content="wrong"),
        )
        assert result.task_completion == "failed"
        assert result.output_quality == 2
        assert len(result.went_wrong) == 2


# ──────────────────────────────────────────────────────────────────
# LearningAgent
# ──────────────────────────────────────────────────────────────────


class TestLearningAgent:
    def test_role(self, mock_llm, memory, logger):
        agent = LearningAgent(mock_llm, memory, logger)
        assert agent.role == AgentRole.LEARNING

    def test_run_stores_episode(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "new_lessons": ["Always validate input"],
            "strategy_refinements": "Add input validation step",
            "skill_gap": "",
            "self_improvement_action": "Practice error handling",
            "semantic_knowledge": "",
            "should_save_procedure": False,
            "procedure_name": "",
            "procedure_steps": [],
        }
        agent = LearningAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input_text="test question",
            perception=PerceptionResult(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
            ),
            strategy=StrategyResult(selected_strategy="direct"),
            evaluation=EvaluationResult(output_quality=7, went_wrong=[]),
        )

        assert memory.episodic.count == 1
        assert "Always validate input" in result.new_lessons
        assert result.episodic_update  # non-empty

    def test_run_saves_procedure_when_quality_high(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "new_lessons": ["Structured approach works"],
            "strategy_refinements": "Keep using this approach",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "Structured analysis improves accuracy",
            "should_save_procedure": True,
            "procedure_name": "structured_analysis_v1",
            "procedure_steps": ["parse", "analyse", "synthesise"],
        }
        agent = LearningAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input_text="analyse this data",
            perception=PerceptionResult(task_type=TaskType.ANALYSIS, modalities=[Modality.TEXT]),
            strategy=StrategyResult(selected_strategy="structured_analysis"),
            evaluation=EvaluationResult(output_quality=9, went_wrong=[]),
        )

        # Episode stored
        assert memory.episodic.count == 1
        # Semantic knowledge stored
        assert memory.semantic.count == 1
        assert memory.semantic.all()[0].knowledge == "Structured analysis improves accuracy"
        # Procedure stored (quality >= 7)
        assert memory.procedural.count == 1
        assert memory.procedural.all()[0].name == "structured_analysis_v1"
        assert len(result.procedural_updates) == 1

    def test_run_does_not_save_procedure_when_quality_low(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "new_lessons": ["Didn't work"],
            "strategy_refinements": "Try different approach",
            "skill_gap": "domain knowledge",
            "self_improvement_action": "Study the topic",
            "semantic_knowledge": "",
            "should_save_procedure": True,  # LLM says yes but quality too low
            "procedure_name": "bad_proc",
            "procedure_steps": ["step1"],
        }
        agent = LearningAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input_text="test",
            perception=PerceptionResult(task_type=TaskType.ANALYSIS),
            strategy=StrategyResult(selected_strategy="bad"),
            evaluation=EvaluationResult(output_quality=4, went_wrong=["bad answer"]),
        )

        # Procedure should NOT be stored because quality < 7
        assert memory.procedural.count == 0
        assert result.skill_gap == "domain knowledge"

    def test_run_identifies_skill_gap(self, mock_llm, memory, logger):
        mock_llm.ask_json.return_value = {
            "new_lessons": [],
            "strategy_refinements": "",
            "skill_gap": "advanced mathematics",
            "self_improvement_action": "Study calculus",
            "semantic_knowledge": "",
            "should_save_procedure": False,
            "procedure_name": "",
            "procedure_steps": [],
        }
        agent = LearningAgent(mock_llm, memory, logger)
        result = agent.run(
            user_input_text="solve this integral",
            perception=PerceptionResult(task_type=TaskType.ANALYSIS),
            strategy=StrategyResult(),
            evaluation=EvaluationResult(output_quality=3, went_wrong=["wrong answer"]),
        )
        assert result.skill_gap == "advanced mathematics"
        assert result.self_improvement_action == "Study calculus"


# ──────────────────────────────────────────────────────────────────
# MetaReflectionAgent
# ──────────────────────────────────────────────────────────────────


class TestMetaReflectionAgent:
    def test_role(self, mock_llm, memory, logger):
        agent = MetaReflectionAgent(mock_llm, memory, logger)
        assert agent.role == AgentRole.META_REFLECTION

    def test_should_trigger_on_interval(self, mock_llm, memory, logger):
        agent = MetaReflectionAgent(mock_llm, memory, logger)
        assert agent.should_trigger() is False  # 0 episodes
        for i in range(_DEFAULT_META_REFLECT_INTERVAL):
            memory.store_episode(EpisodicRecord(outcome_score=7))
        assert agent.should_trigger() is True

    def test_should_not_trigger_off_interval(self, mock_llm, memory, logger):
        agent = MetaReflectionAgent(mock_llm, memory, logger)
        for _ in range(_DEFAULT_META_REFLECT_INTERVAL + 1):
            memory.store_episode(EpisodicRecord())
        assert agent.should_trigger() is False  # 6 is not divisible by 5

    def test_run_produces_meta_reflection(self, mock_llm, memory, logger):
        # Seed some data
        for _ in range(5):
            memory.store_episode(EpisodicRecord(
                task_type=TaskType.CODE_GENERATION,
                outcome_score=7,
                lessons=["test lesson"],
                error_patterns=["timeout"],
            ))
        mock_llm.ask_json.return_value = {
            "recurring_errors": ["timeout"],
            "strength_domains": ["code_generation"],
            "weakness_domains": ["creative_writing"],
            "strategy_trajectory": "improving steadily",
            "bias_check": "slight over-reliance on direct strategies",
            "learning_efficiency": "good — lessons are specific",
            "priority_improvements": ["diversify strategies", "practice creative tasks"],
            "experimental_strategies": ["chain-of-thought for creative"],
            "knowledge_gaps": ["literary techniques"],
            "analysis": "Overall positive trend with room to grow.",
        }
        agent = MetaReflectionAgent(mock_llm, memory, logger)
        result = agent.run()

        assert "timeout" in result.recurring_errors
        assert "code_generation" in result.strength_domains
        assert "creative_writing" in result.weakness_domains
        assert len(result.priority_improvements) == 2
