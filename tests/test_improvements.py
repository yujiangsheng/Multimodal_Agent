"""Tests for the new P0/P1 improvements: OutputGuard, TokenTracker,
structured output validation, MCP auto-connect, Graph templates,
streaming tool calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# =====================================================================
# OutputGuard
# =====================================================================

class TestOutputGuard:
    """Tests for pinocchio.utils.output_guard.OutputGuard."""

    def test_clean_text(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("The answer is 42.")
        assert result.is_safe is True
        assert result.pii_found == []
        assert result.masked_text == "The answer is 42."

    def test_email_masking(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("Contact john@example.com for info.")
        assert "email" in result.pii_found
        assert "[EMAIL]" in result.masked_text
        assert "john@example.com" not in result.masked_text

    def test_credit_card_masking(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("Card: 4111111111111111")
        assert "credit_card" in result.pii_found
        assert "[CREDIT_CARD]" in result.masked_text

    def test_phone_masking(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("Call +1 555-123-4567 now.")
        assert "phone" in result.pii_found
        assert "[PHONE]" in result.masked_text

    def test_multiple_pii(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("Email alice@test.org, card 5500000000000004")
        assert "email" in result.pii_found
        assert "credit_card" in result.pii_found

    def test_no_mask_when_disabled(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard(mask_pii=False)
        result = guard.check("Email: bob@test.com")
        assert "email" in result.pii_found
        # Original text preserved when masking disabled
        assert "bob@test.com" in result.masked_text

    def test_content_policy_blocks(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard(block_policy=True)
        result = guard.check("You should kill all of them.")
        assert result.is_safe is False
        assert "hate_speech" in result.policy_violations

    def test_content_policy_allows_when_disabled(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard(block_policy=False)
        result = guard.check("You should kill all of them.")
        assert result.is_safe is True  # Still flags, but allows
        assert "hate_speech" in result.policy_violations

    def test_empty_text(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("")
        assert result.is_safe is True
        assert result.masked_text == ""

    def test_summary(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("Safe text.")
        assert result.summary == "clean"

    def test_ip_masking(self):
        from pinocchio.utils.output_guard import OutputGuard
        guard = OutputGuard()
        result = guard.check("Server at 192.168.1.1 is down.")
        assert "ip_address" in result.pii_found
        assert "[IP_ADDRESS]" in result.masked_text


# =====================================================================
# TokenTracker
# =====================================================================

class TestTokenTracker:
    """Tests for pinocchio.utils.llm_client.TokenTracker."""

    def test_initial_state(self):
        from pinocchio.utils.llm_client import TokenTracker
        tt = TokenTracker()
        assert tt.total_tokens == 0
        assert tt.call_count == 0

    def test_record(self):
        from pinocchio.utils.llm_client import TokenTracker
        tt = TokenTracker()
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        tt.record(usage)
        assert tt.prompt_tokens == 100
        assert tt.completion_tokens == 50
        assert tt.total_tokens == 150
        assert tt.call_count == 1

    def test_accumulate(self):
        from pinocchio.utils.llm_client import TokenTracker
        tt = TokenTracker()
        for _ in range(3):
            usage = MagicMock()
            usage.prompt_tokens = 10
            usage.completion_tokens = 5
            usage.total_tokens = 15
            tt.record(usage)
        assert tt.prompt_tokens == 30
        assert tt.call_count == 3

    def test_record_none(self):
        from pinocchio.utils.llm_client import TokenTracker
        tt = TokenTracker()
        tt.record(None)  # Should not crash
        assert tt.call_count == 0

    def test_to_dict(self):
        from pinocchio.utils.llm_client import TokenTracker
        tt = TokenTracker()
        d = tt.to_dict()
        assert "prompt_tokens" in d
        assert "completion_tokens" in d
        assert "total_tokens" in d
        assert "call_count" in d

    def test_reset(self):
        from pinocchio.utils.llm_client import TokenTracker
        tt = TokenTracker()
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        tt.record(usage)
        tt.reset()
        assert tt.total_tokens == 0
        assert tt.call_count == 0


# =====================================================================
# Structured Output (ask_json schema validation)
# =====================================================================

class TestStructuredOutput:
    """Tests for LLMClient._validate_and_repair and ask_json schema kwarg."""

    def test_validate_and_repair_fills_missing(self):
        from pinocchio.utils.llm_client import LLMClient
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice"}
        repaired = LLMClient._validate_and_repair(data, schema)
        assert repaired["age"] == 0  # default integer

    def test_validate_and_repair_coerces_types(self):
        from pinocchio.utils.llm_client import LLMClient
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
            },
        }
        data = {"count": "42", "active": "true"}
        repaired = LLMClient._validate_and_repair(data, schema)
        assert repaired["count"] == 42
        assert repaired["active"] is True

    def test_validate_strips_extra_keys(self):
        from pinocchio.utils.llm_client import LLMClient
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        data = {"x": "hello", "extra": "bad"}
        repaired = LLMClient._validate_and_repair(data, schema)
        assert "extra" not in repaired
        assert repaired["x"] == "hello"

    def test_validate_preserves_when_valid(self):
        from pinocchio.utils.llm_client import LLMClient
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        data = {"a": "ok"}
        repaired = LLMClient._validate_and_repair(data, schema)
        assert repaired == {"a": "ok"}

    def test_parse_json_response_good(self):
        from pinocchio.utils.llm_client import LLMClient
        parsed = LLMClient._parse_json_response('{"key": "val"}')
        assert parsed == {"key": "val"}

    def test_parse_json_response_fence(self):
        from pinocchio.utils.llm_client import LLMClient
        raw = "```json\n{\"a\": 1}\n```"
        parsed = LLMClient._parse_json_response(raw)
        assert parsed == {"a": 1}

    def test_parse_json_response_invalid(self):
        from pinocchio.utils.llm_client import LLMClient
        parsed = LLMClient._parse_json_response("not json at all")
        assert parsed == {}


# =====================================================================
# Graph templates
# =====================================================================

class TestGraphTemplates:
    """Validate that graph templates are registered."""

    def test_templates_registered(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        assert "research" in agent._graph_templates
        assert "code_review" in agent._graph_templates

    def test_list_graph_templates(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        templates = agent.list_graph_templates()
        names = [t["name"] for t in templates]
        assert "research_pipeline" in names
        assert "code_review_pipeline" in names

    def test_run_graph_unknown(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        result = agent.run_graph("nonexistent")
        assert "error" in result


# =====================================================================
# MCP auto-connect
# =====================================================================

class TestMCPAutoConnect:
    """Validate MCP auto-connect from env var."""

    def test_no_env_no_connect(self):
        """No PINOCCHIO_MCP_SERVERS → no connections."""
        with patch.dict("os.environ", {}, clear=False):
            with patch("pinocchio.utils.llm_client.openai"):
                agent = _make_agent()
        assert agent.mcp_bridge.connected_servers == []

    def test_connect_mcp_manual(self):
        """connect_mcp() delegates to bridge."""
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        with patch.object(agent.mcp_bridge, "connect", return_value=["mcp_test"]) as m:
            result = agent.connect_mcp("http://localhost:8080/mcp")
        m.assert_called_once_with("http://localhost:8080/mcp")
        assert result == ["mcp_test"]


# =====================================================================
# Orchestrator output guard integration
# =====================================================================

class TestOrchestratorOutputGuard:
    """Validate output guard is wired into orchestrator."""

    def test_output_guard_exists(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        assert hasattr(agent, "_output_guard")
        from pinocchio.utils.output_guard import OutputGuard
        assert isinstance(agent._output_guard, OutputGuard)

    def test_apply_output_guard_clean(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        result = agent._apply_output_guard("Hello world")
        assert result == "Hello world"

    def test_apply_output_guard_masks_pii(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        result = agent._apply_output_guard("Email me at test@example.com")
        assert "[EMAIL]" in result
        assert "test@example.com" not in result


# =====================================================================
# Token tracker on LLMClient
# =====================================================================

class TestLLMClientTokenTracker:
    """Validate LLMClient has token_tracker attribute."""

    def test_token_tracker_exists(self):
        with patch("pinocchio.utils.llm_client.openai"):
            agent = _make_agent()
        assert hasattr(agent.llm, "token_tracker")
        from pinocchio.utils.llm_client import TokenTracker
        assert isinstance(agent.llm.token_tracker, TokenTracker)


# =====================================================================
# ReAct step_callback
# =====================================================================

class TestReActStepCallback:
    """Test that step_callback is invoked during ReAct execution."""

    def test_step_callback_called(self):
        from pinocchio.planning.react import ReActExecutor, ReActStep
        from pinocchio.tools import ToolExecutor, ToolRegistry

        llm = MagicMock()
        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        # Simulate: first call returns calculator action, second returns FINISH
        responses = [
            json.dumps({
                "thought": "I need to calculate",
                "action": "calculator",
                "action_input": {"expression": "2+2"},
            }),
            json.dumps({
                "thought": "Done",
                "action": "FINISH",
                "action_input": {"answer": "4"},
            }),
        ]
        llm.chat = MagicMock(side_effect=responses)

        react = ReActExecutor(llm, executor, registry)
        steps_received: list[ReActStep] = []

        trace = react.run("What is 2+2?", step_callback=lambda s: steps_received.append(s))

        assert len(steps_received) == 2  # calculator step + FINISH step
        assert steps_received[0].action == "calculator"
        assert steps_received[1].action == "FINISH"
        assert trace.final_answer == "4"


# =====================================================================
# Helpers
# =====================================================================

def _make_agent():
    """Create a Pinocchio agent with mocked LLM for testing."""
    with patch("pinocchio.utils.llm_client.openai"):
        return __import__("pinocchio").Pinocchio(
            model="test", api_key="k", base_url="http://x", verbose=False,
        )
