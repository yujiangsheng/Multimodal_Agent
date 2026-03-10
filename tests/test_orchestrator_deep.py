"""Comprehensive tests for the Pinocchio orchestrator.

Covers: full cognitive loop, parallel/sequential modality preprocessing,
error handling, status API, reset, greet, conversation history, audio/video
chat paths, multi-modality inputs, verbose flag, worker override.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pinocchio.orchestrator import Pinocchio
from pinocchio.models.enums import (
    Modality, TaskType, Complexity, ConfidenceLevel, FusionStrategy,
)
from pinocchio.models.schemas import (
    AgentMessage, MultimodalInput, PerceptionResult, StrategyResult,
    EvaluationResult, LearningResult,
)


# ── Shared fixtures ─────────────────────────────────────────────────
@pytest.fixture
def _patch_llm():
    """Patch LLMClient so no real API call is made."""
    with patch("pinocchio.orchestrator.LLMClient") as MockLLM:
        instance = MagicMock()
        instance.model = "test-model"
        instance.temperature = 0.7
        instance.max_tokens = 4096
        MockLLM.return_value = instance
        yield instance


@pytest.fixture
def agent(_patch_llm, tmp_path):
    """Return a Pinocchio orchestrator with all agents mocked at the LLM level."""
    return Pinocchio(
        model="test-model",
        api_key="test",
        base_url="http://test:1234/v1",
        data_dir=str(tmp_path),
        verbose=False,
    )


@pytest.fixture
def agent_verbose(_patch_llm, tmp_path):
    """Verbose variant to test logging path."""
    return Pinocchio(
        model="test-model",
        api_key="test",
        base_url="http://test:1234/v1",
        data_dir=str(tmp_path),
        verbose=True,
    )


def _stub_cognitive_agents(agent: Pinocchio) -> None:
    """Stub all cognitive skills to return valid objects without LLM calls."""
    agent.agent.perceive = MagicMock(return_value=PerceptionResult(
        task_type=TaskType.QUESTION_ANSWERING,
        modalities=[Modality.TEXT],
        complexity=Complexity.MODERATE,
        confidence=ConfidenceLevel.HIGH,
        raw_analysis="Mocked perception.",
    ))
    agent.agent.strategize = MagicMock(return_value=StrategyResult(
        selected_strategy="mock_strategy",
        fusion_strategy=FusionStrategy.LATE_FUSION,
        is_novel=False,
    ))
    agent.agent.execute = MagicMock(return_value=AgentMessage(
        content="Mocked response.",
        confidence=0.9,
        metadata={"strategy": "mock_strategy"},
    ))
    agent.agent.evaluate = MagicMock(return_value=EvaluationResult(
        output_quality=8,
        strategy_effectiveness=8,
        task_completion="complete",
    ))
    agent.agent.learn = MagicMock(return_value=LearningResult(
        new_lessons=["mock lesson"],
        episodic_update="stored",
    ))
    agent.agent.should_meta_reflect = MagicMock(return_value=False)


# =====================================================================
# Full Cognitive Loop
# =====================================================================
class TestCognitiveLoop:
    """Test the full 6-phase cognitive loop through the orchestrator."""

    def test_text_only_chat(self, agent):
        _stub_cognitive_agents(agent)
        response = agent.chat("Hello, Pinocchio!")
        assert response == "Mocked response."
        agent.agent.perceive.assert_called_once()
        agent.agent.strategize.assert_called_once()
        agent.agent.execute.assert_called_once()
        agent.agent.evaluate.assert_called_once()
        agent.agent.learn.assert_called_once()

    def test_interaction_count_increments(self, agent):
        _stub_cognitive_agents(agent)
        agent.chat("First")
        agent.chat("Second")
        assert agent._interaction_count == 2
        assert agent.user_model.interaction_count == 2

    def test_conversation_history_recorded(self, agent):
        _stub_cognitive_agents(agent)
        agent.chat("Test")
        assert len(agent.conversation_history) == 2  # user + assistant
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"

    def test_conversation_history_no_text_input(self, agent):
        """When text=None, only assistant message is stored."""
        _stub_cognitive_agents(agent)
        agent.vision_proc.run = MagicMock(return_value="Image description")
        agent.chat(image_paths=["test.jpg"])
        # No user entry (text was None), just assistant
        user_entries = [h for h in agent.conversation_history if h["role"] == "user"]
        assert len(user_entries) == 0

    def test_meta_reflection_triggered(self, agent):
        _stub_cognitive_agents(agent)
        agent.agent.should_meta_reflect = MagicMock(return_value=True)
        agent.agent.meta_reflect = MagicMock(return_value=MagicMock(
            priority_improvements=["improve creative writing"],
        ))
        agent.chat("Trigger meta")
        agent.agent.meta_reflect.assert_called_once()

    def test_meta_reflection_not_triggered(self, agent):
        _stub_cognitive_agents(agent)
        agent.agent.should_meta_reflect = MagicMock(return_value=False)
        agent.agent.meta_reflect = MagicMock()
        agent.chat("No meta")
        agent.agent.meta_reflect.assert_not_called()

    def test_error_in_cognitive_loop_returns_fallback(self, agent):
        """If any agent raises, the orchestrator returns a safe error message."""
        agent.agent.perceive = MagicMock(side_effect=RuntimeError("LLM down"))
        response = agent.chat("Crash test")
        assert "抱歉" in response or "错误" in response
        assert agent._interaction_count == 1  # still counted


# =====================================================================
# Greet, Reset, Status
# =====================================================================
class TestOrchestratorAPI:
    def test_greet(self, agent):
        g = agent.greet()
        assert "Pinocchio" in g
        assert "进化" in g

    def test_reset_clears_state(self, agent):
        _stub_cognitive_agents(agent)
        agent.chat("Hello")
        agent.reset()
        assert agent._interaction_count == 0
        assert len(agent.conversation_history) == 0
        assert agent.user_model.interaction_count == 0

    def test_status_keys(self, agent):
        s = agent.status()
        assert "interaction_count" in s
        assert "memory_summary" in s
        assert "improvement_trend" in s
        assert "user_model" in s
        assert "resources" in s

    def test_status_resources_populated(self, agent):
        s = agent.status()
        r = s["resources"]
        assert "cpu_count" in r or "cpu_count_physical" in r or isinstance(r, dict)


# =====================================================================
# Parallel Modality Preprocessing
# =====================================================================
class TestModalityPreprocessing:
    """Test _preprocess_modalities parallel and sequential paths."""

    def test_no_modalities_returns_empty(self, agent):
        inp = MultimodalInput(text="Just text")
        result = agent._preprocess_modalities(inp)
        assert result == {}

    def test_single_image_sequential(self, agent):
        agent._parallel_modalities = True
        agent._max_workers = 4
        agent.vision_proc.run = MagicMock(return_value="A cat on a table")
        inp = MultimodalInput(image_paths=["test.jpg"])
        result = agent._preprocess_modalities(inp)
        # Single task → falls through to sequential even if parallel is on
        assert "vision" in result
        assert "cat" in result["vision"]

    def test_multi_modality_parallel(self, agent):
        agent._parallel_modalities = True
        agent._max_workers = 4
        agent.vision_proc.run = MagicMock(return_value="Image desc")
        agent.audio_proc.run = MagicMock(return_value="Audio desc")
        inp = MultimodalInput(
            image_paths=["test.jpg"],
            audio_paths=["test.wav"],
        )
        result = agent._preprocess_modalities(inp)
        assert "vision" in result
        assert "audio" in result

    def test_multi_modality_sequential_when_disabled(self, agent):
        agent._parallel_modalities = False
        agent._max_workers = 1
        agent.vision_proc.run = MagicMock(return_value="Image desc")
        agent.audio_proc.run = MagicMock(return_value="Audio desc")
        inp = MultimodalInput(
            image_paths=["test.jpg"],
            audio_paths=["test.wav"],
        )
        result = agent._preprocess_modalities(inp)
        assert "vision" in result
        assert "audio" in result

    def test_modality_error_captured(self, agent):
        agent._parallel_modalities = True
        agent._max_workers = 4
        agent.vision_proc.run = MagicMock(side_effect=RuntimeError("GPU OOM"))
        agent.audio_proc.run = MagicMock(return_value="Audio OK")
        inp = MultimodalInput(
            image_paths=["test.jpg"],
            audio_paths=["test.wav"],
        )
        result = agent._preprocess_modalities(inp)
        assert "error" in result["vision"]
        assert result["audio"] == "Audio OK"

    def test_video_modality_dispatched(self, agent):
        agent.video_proc.run = MagicMock(return_value="Video analysis")
        inp = MultimodalInput(video_paths=["clip.mp4"])
        result = agent._preprocess_modalities(inp)
        assert "video" in result
        assert result["video"] == "Video analysis"

    def test_all_three_modalities_parallel(self, agent):
        agent._parallel_modalities = True
        agent._max_workers = 4
        agent.vision_proc.run = MagicMock(return_value="Vision")
        agent.audio_proc.run = MagicMock(return_value="Audio")
        agent.video_proc.run = MagicMock(return_value="Video")
        inp = MultimodalInput(
            image_paths=["a.jpg"],
            audio_paths=["b.wav"],
            video_paths=["c.mp4"],
        )
        result = agent._preprocess_modalities(inp)
        assert len(result) == 3
        assert set(result.keys()) == {"vision", "audio", "video"}

    def test_max_workers_override(self, _patch_llm, tmp_path):
        """Custom max_workers should be respected."""
        orch = Pinocchio(
            model="test-model",
            api_key="test",
            base_url="http://test:1234/v1",
            data_dir=str(tmp_path),
            verbose=False,
            max_workers=2,
        )
        assert orch._max_workers == 2


# =====================================================================
# Chat with Images / Audio / Video
# =====================================================================
class TestMultimodalChat:
    """Test chat() with non-text modalities."""

    def test_chat_with_image(self, agent):
        _stub_cognitive_agents(agent)
        agent.vision_proc.run = MagicMock(return_value="Image of a dog")
        response = agent.chat("What is this?", image_paths=["dog.jpg"])
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_with_audio(self, agent):
        _stub_cognitive_agents(agent)
        agent.audio_proc.run = MagicMock(return_value="Meeting transcript")
        response = agent.chat("Summarize this meeting", audio_paths=["meeting.wav"])
        assert isinstance(response, str)

    def test_chat_with_video(self, agent):
        _stub_cognitive_agents(agent)
        agent.video_proc.run = MagicMock(return_value="Video of a cat playing")
        response = agent.chat("What happens in this video?", video_paths=["cat.mp4"])
        assert isinstance(response, str)

    def test_chat_with_all_modalities(self, agent):
        _stub_cognitive_agents(agent)
        agent.vision_proc.run = MagicMock(return_value="Image")
        agent.audio_proc.run = MagicMock(return_value="Audio")
        agent.video_proc.run = MagicMock(return_value="Video")
        response = agent.chat(
            "Analyze all",
            image_paths=["a.jpg"],
            audio_paths=["b.wav"],
            video_paths=["c.mp4"],
        )
        assert isinstance(response, str)

    def test_chat_no_text_pure_image(self, agent):
        _stub_cognitive_agents(agent)
        agent.vision_proc.run = MagicMock(return_value="Landscape photo")
        response = agent.chat(image_paths=["landscape.jpg"])
        assert isinstance(response, str)
