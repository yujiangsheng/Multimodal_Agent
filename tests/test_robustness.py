"""Robustness and edge-case tests for the Pinocchio multimodal agent.

Covers:
  - LLM enum/int parsing safety (perception, strategy, evaluation agents)
  - Score clamping and validation in evaluation
  - Sequential preprocessing error isolation in orchestrator
  - modality_context integration in ExecutionAgent
  - Thread-safe orchestrator state
  - AsyncLLMClient & async_chat
  - Memory corruption recovery (corrupted JSON)
  - ask_json double-failure (returns {} on garbage)
  - ask_json with plain ``` fence extraction
  - VideoProcessor temp file cleanup
  - TextProcessor edge cases
  - Logger data dict serialization
  - MemoryManager pending synthesis lifecycle
  - Boundary tests for quality thresholds
  - Custom temperature/max_tokens overrides in LLMClient.chat()
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from pinocchio.agents.unified_agent import PinocchioAgent
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
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
    SemanticEntry,
    ProceduralEntry,
    UserModel,
)
from pinocchio.multimodal.text_processor import TextProcessor
from pinocchio.multimodal.video_processor import VideoProcessor
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.utils.llm_client import LLMClient, AsyncLLMClient
from pinocchio.utils.logger import PinocchioLogger


# =====================================================================
# Section 1: LLM Enum/Int Parsing Safety
# =====================================================================


class TestPerceptionEnumSafety:
    """PerceptionAgent gracefully handles invalid enum values from LLM."""

    def test_invalid_task_type_defaults_to_unknown(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "task_type": "nonsense_type_that_doesnt_exist",
            "complexity": 3,
            "confidence": "medium",
            "ambiguities": [],
            "analysis": "test",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="hello"))
        assert result.task_type == TaskType.UNKNOWN

    def test_invalid_complexity_defaults_to_moderate(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "task_type": "question_answering",
            "complexity": "very_hard",  # invalid — not an int
            "confidence": "high",
            "ambiguities": [],
            "analysis": "test",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="hello"))
        assert result.complexity == Complexity.MODERATE

    def test_invalid_confidence_defaults_to_medium(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "task_type": "analysis",
            "complexity": 2,
            "confidence": "super_confident",  # invalid enum value
            "ambiguities": [],
            "analysis": "test",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="hello"))
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_complexity_out_of_range_defaults(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "task_type": "analysis",
            "complexity": 99,  # out of enum range (1-5)
            "confidence": "low",
            "ambiguities": [],
            "analysis": "test",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="hello"))
        assert result.complexity == Complexity.MODERATE

    def test_none_values_use_defaults(self, mock_llm, memory_manager, mock_logger):
        """LLM returns None for all fields — should fall through to defaults."""
        mock_llm.ask_json.return_value = {}
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="hello"))
        assert result.task_type == TaskType.UNKNOWN
        assert result.complexity == Complexity.MODERATE
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_ambiguities_non_list_handled(self, mock_llm, memory_manager, mock_logger):
        """LLM returns a string instead of list for ambiguities."""
        mock_llm.ask_json.return_value = {
            "task_type": "conversation",
            "complexity": 1,
            "confidence": "high",
            "ambiguities": "This is a string, not a list",
            "analysis": "test",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="hello"))
        assert isinstance(result.ambiguities, list)
        assert result.ambiguities == []


class TestStrategyEnumSafety:
    """StrategyAgent gracefully handles invalid fusion strategy from LLM."""

    def test_invalid_fusion_strategy_defaults_to_late(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "selected_strategy": "test",
            "basis": "",
            "risk_assessment": "",
            "fallback_plan": "",
            "modality_pipeline": "text→text",
            "fusion_strategy": "mega_fusion",  # invalid
            "is_novel": True,
            "analysis": "",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        perception = PerceptionResult(task_type=TaskType.ANALYSIS, modalities=[Modality.TEXT])
        result = agent.strategize(perception=perception)
        assert result.fusion_strategy == FusionStrategy.LATE_FUSION


class TestEvaluationScoreSafety:
    """EvaluationAgent clamps and validates all numeric scores."""

    def test_string_scores_use_default(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "task_completion": "complete",
            "output_quality": "excellent",  # non-integer
            "strategy_effectiveness": "good",
            "went_well": [],
            "went_wrong": [],
            "surprises": [],
            "cross_modal_coherence": "fine",
            "analysis": "test",
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(content="response"),
        )
        assert result.output_quality == 5
        assert result.strategy_effectiveness == 5
        assert result.cross_modal_coherence == 5

    def test_negative_scores_clamped_to_1(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "output_quality": -5,
            "strategy_effectiveness": -10,
            "cross_modal_coherence": -1,
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(content="response"),
        )
        assert result.output_quality == 1
        assert result.strategy_effectiveness == 1
        assert result.cross_modal_coherence == 1

    def test_scores_above_10_clamped(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "output_quality": 99,
            "strategy_effectiveness": 15,
            "cross_modal_coherence": 100,
            "is_complete": True,
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(content="Here is a thorough response that fully addresses your question."),
        )
        assert result.output_quality == 10
        assert result.strategy_effectiveness == 10
        assert result.cross_modal_coherence == 10

    def test_float_scores_converted(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "output_quality": 7.5,
            "strategy_effectiveness": 8.9,
            "cross_modal_coherence": 6.1,
            "is_complete": True,
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(content="Here is a thorough response that fully addresses your question."),
        )
        assert result.output_quality == 7
        assert result.strategy_effectiveness == 8
        assert result.cross_modal_coherence == 6


# =====================================================================
# Section 2: Orchestrator Error Isolation & Thread Safety
# =====================================================================


class TestSequentialPreprocessingErrorIsolation:
    """Sequential modality preprocessing catches per-task errors."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_failing_processor_isolated(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"task_type":"conversation","complexity":1,"confidence":"high","ambiguities":[],"analysis":"ok"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        from pinocchio.orchestrator import Pinocchio
        agent = Pinocchio(
            model="test",
            api_key="test",
            base_url="http://localhost:11434/v1",
            data_dir=tempfile.mkdtemp(),
            verbose=False,
            parallel_modalities=False,  # force sequential
            max_workers=1,
        )

        # Make vision processor raise
        agent.vision_proc.run = MagicMock(side_effect=RuntimeError("GPU OOM"))
        agent.audio_proc.run = MagicMock(return_value="Audio transcription")

        result = agent._preprocess_modalities(
            MultimodalInput(
                text="test",
                image_paths=["img.jpg"],
                audio_paths=["audio.wav"],
            )
        )
        # Vision failed but audio succeeded
        assert "(error processing vision)" in result["vision"]
        assert result["audio"] == "Audio transcription"


class TestOrchestratorThreadSafety:
    """Interaction count and history are protected by a lock."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_concurrent_chats_consistent_count(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"task_type":"conversation","complexity":1,"confidence":"high","ambiguities":[],"analysis":"ok","selected_strategy":"direct","basis":"","risk_assessment":"","fallback_plan":"","modality_pipeline":"text","fusion_strategy":"late_fusion","is_novel":false,"task_completion":"complete","output_quality":8,"strategy_effectiveness":8,"went_well":[],"went_wrong":[],"surprises":[],"cross_modal_coherence":5,"new_lessons":[],"strategy_refinements":"","skill_gap":"","self_improvement_action":"","semantic_knowledge":"","should_save_procedure":false,"procedure_name":"","procedure_steps":[]}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        from pinocchio.orchestrator import Pinocchio
        agent = Pinocchio(
            model="test", api_key="test",
            base_url="http://localhost:11434/v1",
            data_dir=tempfile.mkdtemp(),
            verbose=False,
        )

        n_threads = 10
        barrier = threading.Barrier(n_threads)
        errors = []

        def _chat(i):
            barrier.wait()
            try:
                agent.chat(f"message {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_chat, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent chats: {errors}"
        assert agent._interaction_count == n_threads
        # Every interaction produces an assistant message
        assistant_msgs = [m for m in agent.conversation_history if m["role"] == "assistant"]
        assert len(assistant_msgs) == n_threads


# =====================================================================
# Section 3: Modality Context Integration
# =====================================================================


class TestExecutionAgentModalityContext:
    """ExecutionAgent uses modality_context in its prompt."""

    def test_modality_context_included_in_prompt(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask.return_value = "response with context"
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.execute(
            user_input=MultimodalInput(text="describe what you see"),
            perception=PerceptionResult(
                task_type=TaskType.MULTIMODAL_REASONING,
                modalities=[Modality.TEXT, Modality.IMAGE],
            ),
            strategy=StrategyResult(selected_strategy="vision-text"),
            modality_context={
                "vision": "A cat sitting on a mat",
                "audio": "Purring sound detected",
            },
        )
        # Verify that the LLM was called with the modality context
        call_args = mock_llm.ask.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[1].get("user") or call_args[0][1]
        assert "MULTIMODAL CONTEXT" in user_prompt
        assert "A cat sitting on a mat" in user_prompt
        assert "Purring sound detected" in user_prompt
        assert result.content == "response with context"

    def test_no_modality_context_no_section(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask.return_value = "plain response"
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.execute(
            user_input=MultimodalInput(text="hello"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            modality_context=None,
        )
        call_args = mock_llm.ask.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[1].get("user") or call_args[0][1]
        assert "MULTIMODAL CONTEXT" not in user_prompt

    def test_empty_modality_context_no_section(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask.return_value = "plain response"
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.execute(
            user_input=MultimodalInput(text="hello"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            modality_context={},
        )
        call_args = mock_llm.ask.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[1].get("user") or call_args[0][1]
        assert "MULTIMODAL CONTEXT" not in user_prompt


# =====================================================================
# Section 4: AsyncLLMClient Tests
# =====================================================================


class TestAsyncLLMClient:
    """Tests for the AsyncLLMClient class."""

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_chat(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "async hello world"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = await client.chat([{"role": "user", "content": "hi"}])
        assert result == "async hello world"

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_ask(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "async answer"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = await client.ask("system", "user question")
        assert result == "async answer"

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_ask_json(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": 42}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = await client.ask_json("system", "user")
        assert result == {"answer": 42}

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_ask_json_invalid_returns_empty(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "totally not json"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = await client.ask_json("system", "user")
        assert result == {}

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_close(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        await client.close()
        mock_client.close.assert_awaited_once()

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_chat_json_mode(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "json mode test"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = await client.chat(
            [{"role": "user", "content": "hi"}],
            json_mode=True,
        )
        # Verify json_mode passed through
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_temperature_override(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "temp test"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        client = AsyncLLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        await client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 100


# =====================================================================
# Section 5: LLMClient ask_json Edge Cases
# =====================================================================


class TestAskJsonEdgeCases:
    """Edge cases in JSON parsing from LLM responses."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_json_plain_fence_extraction(self, mock_openai_mod):
        """Extract JSON from plain ``` fences (not ```json)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Here is the result:\n```\n{"key": "value"}\n```\nDone.'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.ask_json("system", "user")
        assert result == {"key": "value"}

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_json_double_failure_returns_empty(self, mock_openai_mod):
        """If both raw and fence-stripped parsing fail, return {}."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```\nstill_not{valid json\n```'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.ask_json("system", "user")
        assert result == {}

    @patch("pinocchio.utils.llm_client.openai")
    def test_ask_json_json_fence_extraction(self, mock_openai_mod):
        """Extract JSON from ```json fences."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Sure:\n```json\n{"a": 1}\n```'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        result = llm.ask_json("system", "user")
        assert result == {"a": 1}

    @patch("pinocchio.utils.llm_client.openai")
    def test_chat_custom_temperature_and_tokens(self, mock_openai_mod):
        """Verify custom temperature and max_tokens override defaults."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hi"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        llm = LLMClient(model="test", api_key="test", base_url="http://localhost:11434/v1")
        llm.chat(
            [{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=256,
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 256


# =====================================================================
# Section 6: Memory Corruption Recovery
# =====================================================================


class TestMemoryCorruptionRecovery:
    """All three memory stores recover from corrupted JSON files."""

    def test_semantic_memory_corrupted(self, tmp_path):
        path = tmp_path / "sem.json"
        path.write_text("{this is broken]")
        sm = SemanticMemory(str(path))
        assert sm.count == 0
        assert (tmp_path / "sem.json.bak").exists()

    def test_procedural_memory_corrupted(self, tmp_path):
        path = tmp_path / "proc.json"
        path.write_text("!!!")
        pm = ProceduralMemory(str(path))
        assert pm.count == 0
        assert (tmp_path / "proc.json.bak").exists()

    def test_episodic_memory_corrupted_keeps_backup(self, tmp_path):
        path = tmp_path / "ep.json"
        bad_content = "corrupt data here"
        path.write_text(bad_content)
        em = EpisodicMemory(str(path))
        assert em.count == 0
        backup = tmp_path / "ep.json.bak"
        assert backup.exists()
        assert backup.read_text() == bad_content

    def test_episodic_memory_type_error_recovery(self, tmp_path):
        """Array with bad element types doesn't crash."""
        path = tmp_path / "ep.json"
        path.write_text('[{"bad_field": true}]')
        # from_dict with missing fields may raise TypeError
        em = EpisodicMemory(str(path))
        # Should either load with defaults or recover
        assert em.count >= 0  # doesn't crash


# =====================================================================
# Section 7: MemoryManager Lifecycle
# =====================================================================


class TestMemoryManagerPendingSynthesis:
    """Exercise MemoryManager.store_episode → pop_pending_synthesis lifecycle."""

    def test_pending_synthesis_triggered(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        # SemanticMemory threshold is 10
        for i in range(10):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.CODE_GENERATION,
                user_intent=f"coding task {i}",
            ))
        domains = mm.pop_pending_synthesis()
        assert "code_generation" in domains

    def test_pending_synthesis_cleared_after_pop(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        for i in range(10):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.ANALYSIS,
                user_intent=f"analysis {i}",
            ))
        mm.pop_pending_synthesis()
        # Second pop returns empty
        assert mm.pop_pending_synthesis() == []

    def test_no_synthesis_below_threshold(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        for i in range(5):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.TRANSLATION,
                user_intent=f"translate {i}",
            ))
        assert mm.pop_pending_synthesis() == []


class TestMemoryManagerImprovement:
    """Test improvement_trend edge cases."""

    def test_improvement_trend_insufficient_data(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        result = mm.improvement_trend(window=10)
        assert result["trend"] == "insufficient_data"

    def test_improvement_trend_improving(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        # Add 20 episodes: first 10 low score, last 10 high score
        for i in range(20):
            score = 3 if i < 10 else 9
            mm.episodic.add(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=score,
            ))
        result = mm.improvement_trend(window=10)
        assert result["trend"] == "improving"

    def test_improvement_trend_declining(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        for i in range(20):
            score = 9 if i < 10 else 3
            mm.episodic.add(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=score,
            ))
        result = mm.improvement_trend(window=10)
        assert result["trend"] == "declining"

    def test_improvement_trend_stable(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        for i in range(20):
            mm.episodic.add(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=5,
            ))
        result = mm.improvement_trend(window=10)
        assert result["trend"] == "stable"


# =====================================================================
# Section 8: LearningAgent Boundary Tests
# =====================================================================


class TestLearningBoundaries:
    """Test quality threshold boundaries in LearningAgent."""

    def _make_agent(self, mock_llm, memory_manager, mock_logger, llm_result):
        mock_llm.ask_json.return_value = llm_result
        return PinocchioAgent(mock_llm, memory_manager, mock_logger)

    def test_quality_6_procedure_success(self, mock_llm, memory_manager, mock_logger):
        """output_quality == 6 should count as success for procedure usage update."""
        # Add a procedure first
        memory_manager.store_procedure(ProceduralEntry(
            task_type=TaskType.CODE_GENERATION,
            name="test_proc",
            success_rate=0.5,
            usage_count=1,
        ))
        agent = self._make_agent(mock_llm, memory_manager, mock_logger, {
            "new_lessons": ["test lesson"],
            "strategy_refinements": "",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "",
            "should_save_procedure": False,
        })
        evaluation = EvaluationResult(output_quality=6, strategy_effectiveness=6)
        agent.learn(
            user_input_text="code task",
            perception=PerceptionResult(task_type=TaskType.CODE_GENERATION),
            strategy=StrategyResult(selected_strategy="test"),
            evaluation=evaluation,
        )
        proc = memory_manager.procedural.best_procedure(TaskType.CODE_GENERATION)
        assert proc.usage_count == 2
        # success (quality >= 6) → success_rate increases
        assert proc.success_rate > 0.5

    def test_quality_5_procedure_failure(self, mock_llm, memory_manager, mock_logger):
        """output_quality == 5 should count as failure for procedure usage update."""
        memory_manager.store_procedure(ProceduralEntry(
            task_type=TaskType.CODE_GENERATION,
            name="test_proc",
            success_rate=1.0,
            usage_count=1,
        ))
        agent = self._make_agent(mock_llm, memory_manager, mock_logger, {
            "new_lessons": [],
            "should_save_procedure": False,
        })
        evaluation = EvaluationResult(output_quality=5)
        agent.learn(
            user_input_text="code task",
            perception=PerceptionResult(task_type=TaskType.CODE_GENERATION),
            strategy=StrategyResult(selected_strategy="test"),
            evaluation=evaluation,
        )
        proc = memory_manager.procedural.best_procedure(TaskType.CODE_GENERATION)
        assert proc.usage_count == 2
        assert proc.success_rate < 1.0

    def test_quality_7_saves_new_procedure(self, mock_llm, memory_manager, mock_logger):
        """output_quality == 7 + should_save_procedure → new procedure stored."""
        agent = self._make_agent(mock_llm, memory_manager, mock_logger, {
            "new_lessons": ["learned something"],
            "should_save_procedure": True,
            "procedure_name": "new_proc",
            "procedure_steps": ["step1", "step2"],
        })
        evaluation = EvaluationResult(output_quality=7)
        agent.learn(
            user_input_text="creative task",
            perception=PerceptionResult(task_type=TaskType.CREATIVE_WRITING),
            strategy=StrategyResult(selected_strategy="creative"),
            evaluation=evaluation,
        )
        proc = memory_manager.procedural.best_procedure(TaskType.CREATIVE_WRITING)
        assert proc is not None
        assert proc.name == "new_proc"

    def test_quality_6_does_not_save_procedure(self, mock_llm, memory_manager, mock_logger):
        """output_quality == 6 + should_save_procedure → NOT saved (threshold is 7)."""
        agent = self._make_agent(mock_llm, memory_manager, mock_logger, {
            "new_lessons": [],
            "should_save_procedure": True,
            "procedure_name": "should_not_save",
            "procedure_steps": ["step1"],
        })
        evaluation = EvaluationResult(output_quality=6)
        agent.learn(
            user_input_text="task",
            perception=PerceptionResult(task_type=TaskType.CREATIVE_WRITING),
            strategy=StrategyResult(selected_strategy="test"),
            evaluation=evaluation,
        )
        proc = memory_manager.procedural.best_procedure(TaskType.CREATIVE_WRITING)
        assert proc is None


# =====================================================================
# Section 9: MetaReflection Custom Interval
# =====================================================================


class TestMetaReflectionInterval:
    """Test custom meta_reflect_interval values."""

    def test_custom_interval_3(self, mock_llm, memory_manager, mock_logger):
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger, meta_reflect_interval=3)
        # No episodes → should not trigger
        assert not agent.should_meta_reflect()
        # Add 3 episodes
        for i in range(3):
            memory_manager.episodic.add(EpisodicRecord(user_intent=f"test {i}"))
        assert agent.should_meta_reflect()

    def test_custom_interval_7(self, mock_llm, memory_manager, mock_logger):
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger, meta_reflect_interval=7)
        for i in range(6):
            memory_manager.episodic.add(EpisodicRecord(user_intent=f"test {i}"))
        assert not agent.should_meta_reflect()
        memory_manager.episodic.add(EpisodicRecord(user_intent="7th"))
        assert agent.should_meta_reflect()


# =====================================================================
# Section 10: TextProcessor Edge Cases
# =====================================================================


class TestTextProcessorEdges:
    """TextProcessor edge cases."""

    def test_empty_text(self, mock_llm, memory_manager, mock_logger):
        proc = TextProcessor(mock_llm, memory_manager, mock_logger)
        mock_llm.ask.return_value = ""
        result = proc.run(task="summarise", text="")
        assert result == ""

    def test_very_long_text(self, mock_llm, memory_manager, mock_logger):
        proc = TextProcessor(mock_llm, memory_manager, mock_logger)
        long_text = "x" * 100_000
        mock_llm.ask.return_value = "summary"
        result = proc.run(task="summarise", text=long_text)
        assert result == "summary"
        # Verify the long text was passed through
        call_args = mock_llm.ask.call_args
        assert long_text in (call_args.kwargs.get("user", "") or call_args[0][1])


# =====================================================================
# Section 11: VisionProcessor Format Coverage
# =====================================================================


class TestVisionProcessorFormats:
    """VisionProcessor._encode_image handles multiple image formats."""

    def test_jpg_mime(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        result = VisionProcessor._encode_image(str(img))
        assert result.startswith("data:image/jpeg;base64,")

    def test_gif_mime(self, tmp_path):
        img = tmp_path / "test.gif"
        img.write_bytes(b"GIF89a")
        result = VisionProcessor._encode_image(str(img))
        assert result.startswith("data:image/gif;base64,")

    def test_webp_mime(self, tmp_path):
        img = tmp_path / "test.webp"
        img.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        result = VisionProcessor._encode_image(str(img))
        assert result.startswith("data:image/webp;base64,")

    def test_unknown_extension_defaults_png(self, tmp_path):
        img = tmp_path / "test.bmp"
        img.write_bytes(b"BM\x00\x00")
        result = VisionProcessor._encode_image(str(img))
        assert result.startswith("data:image/png;base64,")


# =====================================================================
# Section 12: Logger Data Dict & Level
# =====================================================================


class TestLoggerDataDict:
    """Logger handles data dict serialization."""

    def test_log_with_data_dict(self):
        logger = PinocchioLogger()
        # Should not raise
        logger.log(AgentRole.ORCHESTRATOR, "test message", data={"key": "value", "num": 42})

    def test_log_success_method(self):
        logger = PinocchioLogger()
        logger.success(AgentRole.LEARNING, "operation succeeded")

    def test_log_phase_and_separator(self):
        logger = PinocchioLogger()
        logger.phase("Phase 1: TEST")
        logger.separator()


# =====================================================================
# Section 13: Orchestrator async_chat
# =====================================================================


class TestAsyncChat:
    """Test the orchestrator's async_chat method."""

    @patch("pinocchio.utils.llm_client.openai")
    async def test_async_chat_basic(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "task_type": "conversation",
            "complexity": 1,
            "confidence": "high",
            "ambiguities": [],
            "analysis": "ok",
            "selected_strategy": "direct",
            "basis": "",
            "risk_assessment": "",
            "fallback_plan": "",
            "modality_pipeline": "text",
            "fusion_strategy": "late_fusion",
            "is_novel": False,
            "task_completion": "complete",
            "output_quality": 8,
            "strategy_effectiveness": 8,
            "went_well": [],
            "went_wrong": [],
            "surprises": [],
            "cross_modal_coherence": 5,
            "new_lessons": [],
            "strategy_refinements": "",
            "skill_gap": "",
            "self_improvement_action": "",
            "semantic_knowledge": "",
            "should_save_procedure": False,
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        from pinocchio.orchestrator import Pinocchio
        agent = Pinocchio(
            model="test", api_key="test",
            base_url="http://localhost:11434/v1",
            data_dir=tempfile.mkdtemp(),
            verbose=False,
        )
        result = await agent.async_chat("hello async")
        assert isinstance(result, str)
        assert len(result) > 0


# =====================================================================
# Section 14: Evaluation Response Truncation
# =====================================================================


class TestEvaluationTruncation:
    """Verify that response content is truncated at 4000 chars."""

    def test_long_response_truncated(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "output_quality": 7,
            "strategy_effectiveness": 7,
            "cross_modal_coherence": 5,
        }
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        long_response = AgentMessage(content="x" * 10_000)
        agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=long_response,
        )
        # Verify the prompt sent to LLM has truncated content
        call_args = mock_llm.ask_json.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[0][1]
        # The response content should be truncated to at most 4000 chars
        response_section = user_prompt.split("=== RESPONSE ===\n")[1]
        # Extract just the response content (before any warning section)
        response_content = response_section.split("\n\n===")[0]
        assert len(response_content) <= 4000
        # The full 10K chars should NOT be present
        assert "x" * 10_000 not in user_prompt


# =====================================================================
# Section 15: Perception Lessons Truncation
# =====================================================================


class TestPerceptionLessonsTruncation:
    """Verify that lessons are truncated to 5."""

    def test_more_than_5_lessons_truncated(self, mock_llm, memory_manager, mock_logger):
        mock_llm.ask_json.return_value = {
            "task_type": "question_answering",
            "complexity": 3,
            "confidence": "medium",
        }
        # Add episodes with many lessons
        for i in range(3):
            memory_manager.episodic.add(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                lessons=[f"lesson_{i}_a", f"lesson_{i}_b", f"lesson_{i}_c"],
            ))
        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent.perceive(user_input=MultimodalInput(text="test"))
        assert len(result.relevant_lessons) <= 5


# =====================================================================
# Section 16: LLM Audio Format Detection
# =====================================================================


class TestAudioFormatDetection:
    """LLMClient._audio_format correctly detects formats."""

    def test_wav_format(self):
        assert LLMClient._audio_format("recording.wav") == "wav"

    def test_mp3_format(self):
        assert LLMClient._audio_format("song.mp3") == "mp3"

    def test_flac_format(self):
        assert LLMClient._audio_format("lossless.flac") == "flac"

    def test_ogg_format(self):
        assert LLMClient._audio_format("voice.ogg") == "ogg"

    def test_unknown_format_defaults_wav(self):
        assert LLMClient._audio_format("audio.m4a") == "wav"

    def test_url_format(self):
        assert LLMClient._audio_format("https://example.com/audio.mp3") == "mp3"


# =====================================================================
# Section 17: Concurrent Memory Operations
# =====================================================================


class TestConcurrentMemoryOps:
    """Basic thread-safety test for memory writes."""

    def test_concurrent_episodic_adds(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        n = 50
        barrier = threading.Barrier(n)
        errors = []

        def _add(i):
            barrier.wait()
            try:
                em.add(EpisodicRecord(
                    user_intent=f"intent {i}",
                    task_type=TaskType.QUESTION_ANSWERING,
                ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash; count should be consistent
        assert not errors, f"Concurrent add errors: {errors}"
        assert em.count == n


# =====================================================================
# Section 18: VideoProcessor Temp Cleanup
# =====================================================================


class TestVideoTempCleanup:
    """VideoProcessor fallback path cleans up temp files."""

    def test_fallback_cleans_temp_files(self, mock_llm, memory_manager, mock_logger):
        proc = VideoProcessor(mock_llm, memory_manager, mock_logger)
        mock_vision = MagicMock()
        mock_vision.run.return_value = "frame description"
        mock_audio = MagicMock()
        mock_audio.run.return_value = "audio description"
        mock_llm.ask.return_value = "video summary"

        # Create a real temp dir with fake frames and audio
        temp_dir = Path(tempfile.mkdtemp(prefix="pinocchio_frames_"))
        (temp_dir / "frame_0001.jpg").write_bytes(b"fake frame")
        temp_audio = Path(tempfile.mktemp(suffix=".wav", prefix="pinocchio_audio_"))
        temp_audio.write_bytes(b"fake audio")

        with patch.object(proc, "extract_frames", return_value=[str(temp_dir / "frame_0001.jpg")]):
            with patch.object(proc, "extract_audio", return_value=str(temp_audio)):
                proc.run(
                    task="analyse video",
                    video_paths=["test.mp4"],
                    vision_processor=mock_vision,
                    audio_processor=mock_audio,
                    native_video=False,
                )

        # Temp dir and audio file should be cleaned
        assert not temp_dir.exists(), "Temp frame directory was not cleaned up"
        assert not temp_audio.exists(), "Temp audio file was not cleaned up"


# =====================================================================
# Section 19: Full Cognitive Loop with Invalid LLM
# =====================================================================


class TestCognitiveLoopWithGarbageLLM:
    """The cognitive loop survives LLM returning complete garbage."""

    @patch("pinocchio.utils.llm_client.openai")
    def test_loop_survives_garbage(self, mock_openai_mod):
        """Even if LLM returns non-JSON garbage for every call, the loop
        completes and returns a response (all defaults kick in)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I refuse to output JSON"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_mod.OpenAI.return_value = mock_client

        from pinocchio.orchestrator import Pinocchio
        agent = Pinocchio(
            model="test", api_key="test",
            base_url="http://localhost:11434/v1",
            data_dir=tempfile.mkdtemp(),
            verbose=False,
        )
        # Should not crash — returns a response string
        result = agent.chat("What is 2+2?")
        assert isinstance(result, str)
        assert len(result) > 0
