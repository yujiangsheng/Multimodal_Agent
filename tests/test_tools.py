"""Tests for tool-calling framework: ToolRegistry, ToolExecutor, @tool decorator,
enable/disable, unregister, timeout, metrics, and PinocchioAgent tool integration."""

from __future__ import annotations

import asyncio
import time
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
        assert registry.count == 2
        names = registry.list_names()
        assert "calculator" in names
        assert "current_time" in names

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
        assert len(schema) == 2
        assert schema[0]["type"] == "function"

    def test_unregister(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        assert registry.unregister("calculator") is True
        assert registry.get("calculator") is None
        assert registry.count == 1

    def test_unregister_nonexistent(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        assert registry.unregister("nope") is False

    def test_enable_disable(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        assert registry.enabled_count == 2

        assert registry.disable("calculator") is True
        assert registry.enabled_count == 1
        assert "calculator" not in registry.list_enabled()
        # Disabled tools excluded from prompt & schema
        assert "calculator" not in registry.to_prompt_description()
        assert len(registry.to_openai_schema()) == 1

        # Re-enable
        assert registry.enable("calculator") is True
        assert registry.enabled_count == 2

    def test_enable_disable_nonexistent(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        assert registry.enable("no_such") is False
        assert registry.disable("no_such") is False

    def test_register_function(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        t = registry.register_function(
            lambda text="": text.upper(),
            name="upper",
            description="Uppercase",
        )
        assert t.name == "upper"
        assert registry.get("upper") is t


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_decorator_with_registry(self):
        from pinocchio.tools import ToolRegistry, tool

        registry = ToolRegistry()

        @tool(registry, description="Add two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        assert registry.get("add") is not None
        assert registry.get("add").description == "Add two numbers"
        assert add(1, 2) == "3"

    def test_decorator_without_registry(self):
        from pinocchio.tools import tool

        @tool(description="Multiply")
        def multiply(a: int, b: int) -> str:
            return str(a * b)

        assert hasattr(multiply, "_tool")
        assert multiply._tool.name == "multiply"
        assert multiply(3, 4) == "12"

    def test_decorator_auto_schema(self):
        from pinocchio.tools import ToolRegistry, tool

        registry = ToolRegistry()

        @tool(registry, description="Greet")
        def greet(name: str, loud: bool = False) -> str:
            return f"HI {name}" if loud else f"Hi {name}"

        params = registry.get("greet").parameters
        assert "name" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert "name" in params["required"]
        assert "loud" not in params["required"]

    def test_decorator_custom_name(self):
        from pinocchio.tools import ToolRegistry, tool

        registry = ToolRegistry()

        @tool(registry, name="my_tool", description="Custom name")
        def something() -> str:
            return "ok"

        assert registry.get("my_tool") is not None
        assert registry.get("something") is None


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

    def test_execute_disabled_tool(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        registry.disable("calculator")
        executor = ToolExecutor(registry)

        result = executor.execute("calculator", {"expression": "1+1"})
        assert "disabled" in result

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

    def test_timeout(self):
        from pinocchio.tools import Tool, ToolRegistry, ToolExecutor

        def slow(duration: str = "5") -> str:
            time.sleep(float(duration))
            return "done"

        registry = ToolRegistry()
        registry.register(Tool(
            name="slow",
            description="Slow tool",
            parameters={"type": "object", "properties": {"duration": {"type": "string"}}},
            function=slow,
        ))
        executor = ToolExecutor(registry, timeout=0.3)

        result = executor.execute("slow", {"duration": "5"})
        assert "timed out" in result

    def test_metrics_recorded(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        executor.execute("calculator", {"expression": "1+1"})
        executor.execute("current_time", {})
        executor.execute("calculator", {"expression": "2+2"})

        assert len(executor.history) == 3
        stats = executor.stats()
        assert stats["total_calls"] == 3
        assert stats["by_tool"]["calculator"]["calls"] == 2
        assert stats["by_tool"]["calculator"]["success_rate"] == 1.0
        assert stats["by_tool"]["current_time"]["calls"] == 1

    def test_clear_history(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        executor.execute("calculator", {"expression": "1+1"})
        assert len(executor.history) == 1
        executor.clear_history()
        assert len(executor.history) == 0

    def test_stats_empty(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        stats = executor.stats()
        assert stats["total_calls"] == 0


class TestAsyncToolExecutor:
    """Tests for async tool execution."""

    @pytest.mark.asyncio
    async def test_async_execute(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = await executor.async_execute("calculator", {"expression": "3+7"})
        assert result == "10"

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        from pinocchio.tools import Tool, ToolRegistry, ToolExecutor

        def slow() -> str:
            time.sleep(5)
            return "done"

        registry = ToolRegistry()
        registry.register(Tool(
            name="slow", description="Slow", parameters={"type": "object", "properties": {}},
            function=slow,
        ))
        executor = ToolExecutor(registry, timeout=0.3)

        result = await executor.async_execute("slow", {})
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_async_unknown_tool(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        result = await executor.async_execute("nope", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_async_disabled_tool(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        registry.disable("calculator")
        executor = ToolExecutor(registry)

        result = await executor.async_execute("calculator", {"expression": "1"})
        assert "disabled" in result


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
