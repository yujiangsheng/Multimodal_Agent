"""Tests for tool-calling framework: ToolRegistry, ToolExecutor, and PinocchioAgent tool integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        from pinocchio.tools import Tool, ToolRegistry

        registry = ToolRegistry()
        tool = Tool(
            name="echo",
            description="Echo input",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            function=lambda text="": text,
        )
        registry.register(tool)
        assert registry.get("echo") is tool
        assert "echo" in registry.list_names()
        assert registry.count == 1

    def test_register_defaults(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        assert registry.count == 3
        names = registry.list_names()
        assert "calculator" in names
        assert "current_time" in names
        assert "python_eval" in names

    def test_to_prompt_description(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        desc = registry.to_prompt_description()
        assert "calculator" in desc
        assert "current_time" in desc

    def test_to_openai_schema(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        schema = registry.to_openai_schema()
        assert isinstance(schema, list)
        assert len(schema) == 3
        assert schema[0]["type"] == "function"


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def test_execute_calculator(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("calculator", {"expression": "2 + 3"})
        assert result == "5"

    def test_execute_calculator_math_functions(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("calculator", {"expression": "sqrt(16)"})
        assert result == "4.0"

    def test_execute_current_time(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("current_time", {})
        assert len(result) > 10

    def test_execute_unknown_tool(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        result = executor.execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_parse_and_execute(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        raw = '{"tool": "calculator", "arguments": {"expression": "10 * 5"}}'
        result = executor.parse_and_execute(raw)
        assert result == "50"

    def test_parse_and_execute_invalid_json(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        result = executor.parse_and_execute("not valid json")
        assert "Invalid JSON" in result or "Failed to parse" in result

    def test_calculator_rejects_dangerous_expressions(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("calculator", {"expression": "__import__('os').system('ls')"})
        assert "Error" in result


class TestToolCallDetection:
    """Test that PinocchioAgent._process_tool_calls detects and runs tools."""

    def test_process_tool_calls_no_tools(self, mock_llm, memory_manager, mock_logger):
        from pinocchio.agents.unified_agent import PinocchioAgent

        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        result = agent._process_tool_calls("Hello world", "sys", "user")
        assert result == "Hello world"

    def test_process_tool_calls_with_tool(self, mock_llm, memory_manager, mock_logger):
        from pinocchio.agents.unified_agent import PinocchioAgent
        from pinocchio.tools import ToolRegistry, ToolExecutor

        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)
        agent.set_tools(registry, executor)

        response_with_tool = (
            "Let me calculate that for you.\n"
            "```tool_call\n"
            '{"tool": "calculator", "arguments": {"expression": "2+2"}}\n'
            "```\n"
        )
        mock_llm.ask.return_value = "The answer is 4."

        result = agent._process_tool_calls(response_with_tool, "sys prompt", "what is 2+2")
        assert result == "The answer is 4."
        call_args = mock_llm.ask.call_args
        assert "Tool result" in call_args[1]["user"] or "Tool result" in call_args[0][1]

    def test_process_tool_calls_no_match(self, mock_llm, memory_manager, mock_logger):
        from pinocchio.agents.unified_agent import PinocchioAgent
        from pinocchio.tools import ToolRegistry, ToolExecutor

        agent = PinocchioAgent(mock_llm, memory_manager, mock_logger)
        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)
        agent.set_tools(registry, executor)

        result = agent._process_tool_calls("Just a normal response", "sys", "user")
        assert result == "Just a normal response"
