"""Integration test — full cognitive loop with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinocchio.orchestrator import Pinocchio
from pinocchio.models.enums import TaskType


class TestPinocchioIntegration:
    """End-to-end tests for the full cognitive loop via the Pinocchio orchestrator."""

    @pytest.fixture
    def agent(self, tmp_data_dir):
        """Create a Pinocchio instance with mocked LLM."""
        with patch("pinocchio.utils.llm_client.openai") as mock_openai:
            p = Pinocchio(
                model="test-model",
                api_key="test-key",
                data_dir=tmp_data_dir,
            )
            return p

    def _setup_llm_responses(self, agent):
        """Configure the mock LLM to return appropriate responses for each phase."""
        # We need to mock ask_json and ask to return plausible responses
        call_count = {"n": 0}

        def mock_ask_json(system, user, **kwargs):
            call_count["n"] += 1
            n = call_count["n"]
            # Phase 1: Perception
            if n == 1:
                return {
                    "task_type": "question_answering",
                    "complexity": 2,
                    "confidence": "high",
                    "ambiguities": [],
                    "analysis": "Simple question about physics.",
                }
            # Phase 2: Strategy
            elif n == 2:
                return {
                    "selected_strategy": "direct_answer",
                    "basis": "first principles — straightforward QA",
                    "risk_assessment": "low",
                    "fallback_plan": "use analogy",
                    "modality_pipeline": "text→reasoning→text",
                    "fusion_strategy": "late_fusion",
                    "is_novel": True,
                    "analysis": "Direct answer approach.",
                }
            # Phase 4: Evaluation
            elif n == 3:
                return {
                    "task_completion": "complete",
                    "output_quality": 8,
                    "strategy_effectiveness": 8,
                    "went_well": ["clear explanation"],
                    "went_wrong": [],
                    "surprises": [],
                    "cross_modal_coherence": 5,
                    "analysis": "Good output.",
                }
            # Phase 5: Learning
            elif n == 4:
                return {
                    "new_lessons": ["Physics QA works well with direct approach"],
                    "strategy_refinements": "Add real-world examples",
                    "skill_gap": "",
                    "self_improvement_action": "Collect more physics analogies",
                    "semantic_knowledge": "Direct explanation works for simple physics QA",
                    "should_save_procedure": True,
                    "procedure_name": "physics_qa_v1",
                    "procedure_steps": ["classify question", "recall principle", "explain clearly"],
                }
            # Phase 6 or subsequent calls
            else:
                return {
                    "recurring_errors": [],
                    "strength_domains": ["question_answering"],
                    "weakness_domains": [],
                    "strategy_trajectory": "just started",
                    "bias_check": "none detected",
                    "learning_efficiency": "good",
                    "priority_improvements": [],
                    "experimental_strategies": [],
                    "knowledge_gaps": [],
                    "analysis": "Initial reflection.",
                }

        agent.llm.ask_json = MagicMock(side_effect=mock_ask_json)
        agent.llm.ask = MagicMock(return_value="Light travels at approximately 299,792,458 meters per second in a vacuum.")
        agent.llm.chat = MagicMock(return_value="Light speed is about 3×10⁸ m/s.")
        agent.llm.build_vision_message = MagicMock(return_value={"role": "user", "content": []})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chat_and_wait(agent, *args, **kwargs):
        """Call agent.chat() then wait for the background learning thread."""
        result = agent.chat(*args, **kwargs)
        if agent._post_response_thread is not None:
            agent._post_response_thread.join(timeout=5)
        return result

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_greeting(self, agent):
        greeting = agent.greet()
        assert "Pinocchio" in greeting
        assert "进化" in greeting

    def test_single_text_interaction(self, agent):
        self._setup_llm_responses(agent)
        response = self._chat_and_wait(agent, "How fast does light travel?")

        assert response  # non-empty
        assert "299,792,458" in response or "3×10" in response
        assert agent._interaction_count == 1

        # Memory should have stored one episode
        assert agent.memory.episodic.count == 1
        ep = agent.memory.episodic.all()[0]
        assert ep.task_type == TaskType.QUESTION_ANSWERING
        assert ep.outcome_score == 8

    def test_conversation_history_tracking(self, agent):
        self._setup_llm_responses(agent)
        agent.chat("Question 1")

        assert len(agent.conversation_history) == 2  # user + assistant
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"

    def test_multiple_interactions_accumulate_memory(self, agent):
        self._setup_llm_responses(agent)
        self._chat_and_wait(agent, "Question 1")

        # Reset call count for second interaction
        self._setup_llm_responses(agent)
        self._chat_and_wait(agent, "Question 2")

        assert agent._interaction_count == 2
        assert agent.memory.episodic.count == 2

    def test_learning_stores_semantic_and_procedural(self, agent):
        self._setup_llm_responses(agent)
        self._chat_and_wait(agent, "Explain quantum mechanics")

        # Learning phase stores semantic knowledge and procedure
        assert agent.memory.semantic.count >= 1
        assert agent.memory.procedural.count >= 1
        proc = agent.memory.procedural.all()[0]
        assert proc.name == "physics_qa_v1"

    def test_meta_reflection_triggers_at_interval(self, agent):
        """After 5 interactions, meta-reflection should trigger."""
        for i in range(5):
            self._setup_llm_responses(agent)
            self._chat_and_wait(agent, f"Question {i+1}")

        assert agent._interaction_count == 5
        # 5 episodes stored = triggers meta-reflection
        assert agent.memory.episodic.count == 5

    def test_status_returns_valid_summary(self, agent):
        self._setup_llm_responses(agent)
        self._chat_and_wait(agent, "test")

        status = agent.status()
        assert status["interaction_count"] == 1
        assert "memory_summary" in status
        assert "improvement_trend" in status
        assert "user_model" in status
        assert status["memory_summary"]["episodic_count"] == 1

    def test_reset_clears_session_keeps_memory(self, agent):
        self._setup_llm_responses(agent)
        self._chat_and_wait(agent, "test")
        assert agent._interaction_count == 1
        assert len(agent.conversation_history) == 2
        stored_episodes = agent.memory.episodic.count

        agent.reset()

        assert agent._interaction_count == 0
        assert len(agent.conversation_history) == 0
        # Memory persists after reset
        assert agent.memory.episodic.count == stored_episodes

    def test_error_recovery(self, agent):
        """If a sub-agent throws, orchestrator should return a safe response."""
        agent.llm.ask_json = MagicMock(side_effect=RuntimeError("API error"))

        response = agent.chat("This will fail")
        assert "抱歉" in response or "错误" in response
        assert agent._interaction_count == 1


class TestMultimodalInput:
    @pytest.fixture
    def agent(self, tmp_data_dir):
        with patch("pinocchio.utils.llm_client.openai"):
            return Pinocchio(model="test", api_key="test", data_dir=tmp_data_dir)

    def test_image_input_detected(self, agent):
        """Ensure image paths are correctly passed through the cognitive loop."""
        call_count = {"n": 0}

        def mock_ask_json(system, user, **kwargs):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return {
                    "task_type": "multimodal_reasoning",
                    "complexity": 3,
                    "confidence": "medium",
                    "ambiguities": [],
                    "analysis": "Image analysis task.",
                }
            elif n == 2:
                return {
                    "selected_strategy": "vision_qa",
                    "basis": "first principles",
                    "risk_assessment": "medium",
                    "fallback_plan": "describe then reason",
                    "modality_pipeline": "image→caption→reasoning→text",
                    "fusion_strategy": "early_fusion",
                    "is_novel": True,
                    "analysis": "Vision approach.",
                }
            elif n == 3:
                return {
                    "task_completion": "complete",
                    "output_quality": 7,
                    "strategy_effectiveness": 7,
                    "went_well": ["identified objects"],
                    "went_wrong": [],
                    "surprises": [],
                    "cross_modal_coherence": 8,
                    "analysis": "Good vision result.",
                }
            else:
                return {
                    "new_lessons": ["Vision tasks need detail"],
                    "strategy_refinements": "Ask about specific regions",
                    "skill_gap": "",
                    "self_improvement_action": "",
                    "semantic_knowledge": "",
                    "should_save_procedure": False,
                    "procedure_name": "",
                    "procedure_steps": [],
                }

        agent.llm.ask_json = MagicMock(side_effect=mock_ask_json)
        agent.llm.chat = MagicMock(return_value="I see a cat sitting on a table.")
        agent.llm.build_vision_message = MagicMock(return_value={"role": "user", "content": []})
        # Mock the vision processor so modality preprocessing doesn't read the file
        agent.vision_proc.run = MagicMock(return_value="A cat sitting on a table in a room.")

        # Create a real tiny PNG so ExecutionAgent._encode_image can read it
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
                b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            tmp_img = f.name
        try:
            response = agent.chat("What's in this image?", image_paths=[tmp_img])
        finally:
            os.unlink(tmp_img)
        assert "cat" in response
        # Vision processor was invoked during modality preprocessing
        agent.vision_proc.run.assert_called_once()
