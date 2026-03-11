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
        assert registry.count == 18
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
        assert len(schema) == 18
        assert schema[0]["type"] == "function"

    def test_unregister(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        assert registry.unregister("calculator") is True
        assert registry.get("calculator") is None
        assert registry.count == 17

    def test_unregister_nonexistent(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        assert registry.unregister("nope") is False

    def test_enable_disable(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        assert registry.enabled_count == 18

        assert registry.disable("calculator") is True
        assert registry.enabled_count == 17
        assert "calculator" not in registry.list_enabled()
        # Disabled tools excluded from prompt & schema
        prompt_desc = registry.to_prompt_description()
        # "calculator(" should not appear (but "hash_calculator" still will)
        assert "- calculator(" not in prompt_desc
        assert len(registry.to_openai_schema()) == 17

        # Re-enable
        assert registry.enable("calculator") is True
        assert registry.enabled_count == 18

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


# =====================================================================
# Tests for new built-in tools
# =====================================================================


class TestRegisterDefaultsCount:
    """Validate that register_defaults registers all 18 tools."""

    def test_default_tool_count(self):
        from pinocchio.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register_defaults()
        assert registry.count == 18
        expected = {
            "calculator", "current_time", "web_fetch", "file_reader",
            "file_writer", "json_formatter", "text_stats", "hash_calculator",
            "uuid_generator", "base64_codec", "unit_converter", "random_number",
            "regex_match", "system_info", "directory_listing", "shell_command",
            "web_search", "python_exec",
        }
        assert set(registry.list_names()) == expected


class TestJsonFormatter:
    """Tests for json_formatter tool."""

    def test_format_valid_json(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("json_formatter", {"text": '{"a":1,"b":2}'})
        assert '"a": 1' in result
        assert '"b": 2' in result

    def test_format_invalid_json(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("json_formatter", {"text": "not json"})
        assert "Error" in result


class TestTextStats:
    """Tests for text_stats tool."""

    def test_basic_stats(self):
        import json as json_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("text_stats", {"text": "Hello world.\nSecond line."})
        stats = json_mod.loads(result)
        assert stats["words"] == 4
        assert stats["lines"] == 2
        assert stats["characters"] > 0


class TestHashCalculator:
    """Tests for hash_calculator tool."""

    def test_sha256(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("hash_calculator", {"text": "hello"})
        assert result.startswith("sha256:")
        assert len(result.split(":")[1]) == 64

    def test_md5(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("hash_calculator", {"text": "hello", "algorithm": "md5"})
        assert result.startswith("md5:")

    def test_unsupported_algo(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("hash_calculator", {"text": "hi", "algorithm": "crc32"})
        assert "Error" in result


class TestUuidGenerator:
    """Tests for uuid_generator tool."""

    def test_uuid4(self):
        import uuid as uuid_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("uuid_generator", {})
        # Validate it's a real UUID
        uuid_mod.UUID(result)

    def test_uuid1(self):
        import uuid as uuid_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("uuid_generator", {"version": 1})
        uuid_mod.UUID(result)

    def test_unsupported_version(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("uuid_generator", {"version": 3})
        assert "Error" in result


class TestBase64Codec:
    """Tests for base64_codec tool."""

    def test_encode(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("base64_codec", {"text": "Hello"})
        assert result == "SGVsbG8="

    def test_decode(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("base64_codec", {"text": "SGVsbG8=", "action": "decode"})
        assert result == "Hello"

    def test_decode_invalid(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("base64_codec", {"text": "\x00\xff\xfe", "action": "decode"})
        assert "Error" in result

    def test_unknown_action(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("base64_codec", {"text": "hi", "action": "compress"})
        assert "Error" in result


class TestUnitConverter:
    """Tests for unit_converter tool."""

    def test_km_to_mi(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("unit_converter", {"value": 10.0, "from_unit": "km", "to_unit": "mi"})
        assert "6.21" in result

    def test_celsius_to_fahrenheit(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("unit_converter", {"value": 100.0, "from_unit": "C", "to_unit": "F"})
        assert "212" in result

    def test_kg_to_lb(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("unit_converter", {"value": 1.0, "from_unit": "kg", "to_unit": "lb"})
        assert "2.20" in result

    def test_incompatible_units(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("unit_converter", {"value": 1.0, "from_unit": "km", "to_unit": "kg"})
        assert "Error" in result


class TestRandomNumber:
    """Tests for random_number tool."""

    def test_single_integer(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("random_number", {"min_val": 1, "max_val": 10})
        val = int(result)
        assert 1 <= val <= 10

    def test_multiple_floats(self):
        import json as json_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("random_number", {
            "min_val": 0, "max_val": 1, "count": 5, "integer": False
        })
        vals = json_mod.loads(result)
        assert len(vals) == 5
        assert all(0 <= v <= 1 for v in vals)


class TestRegexMatch:
    """Tests for regex_match tool."""

    def test_match_found(self):
        import json as json_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("regex_match", {
            "pattern": r"\d+", "text": "abc 123 def"
        })
        data = json_mod.loads(result)
        assert data["matched"] is True
        assert data["match"] == "123"

    def test_find_all(self):
        import json as json_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("regex_match", {
            "pattern": r"\d+", "text": "a1 b22 c333", "find_all": True
        })
        data = json_mod.loads(result)
        assert data["count"] == 3
        assert data["matches"] == ["1", "22", "333"]

    def test_no_match(self):
        import json as json_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("regex_match", {"pattern": r"\d+", "text": "no digits"})
        data = json_mod.loads(result)
        assert data["matched"] is False

    def test_invalid_regex(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("regex_match", {"pattern": "[unclosed", "text": "test"})
        assert "Error" in result


class TestSystemInfo:
    """Tests for system_info tool."""

    def test_returns_json(self):
        import json as json_mod
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("system_info", {})
        data = json_mod.loads(result)
        assert "os" in data
        assert "python_version" in data
        assert "architecture" in data


class TestDirectoryListing:
    """Tests for directory_listing tool."""

    def test_list_current_dir(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("directory_listing", {"path": "."})
        assert len(result) > 0
        assert "Error" not in result

    def test_nonexistent_dir(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("directory_listing", {"path": "/nonexistent_xyz_123"})
        assert "Error" in result


class TestFileReaderWriter:
    """Tests for file_reader and file_writer tools."""

    def test_write_and_read(self, tmp_path):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        fpath = str(tmp_path / "test.txt")
        write_result = executor.execute("file_writer", {"path": fpath, "content": "Hello World"})
        assert "Successfully" in write_result

        read_result = executor.execute("file_reader", {"path": fpath})
        assert read_result == "Hello World"

    def test_append_mode(self, tmp_path):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        fpath = str(tmp_path / "append.txt")
        executor.execute("file_writer", {"path": fpath, "content": "Line 1\n"})
        executor.execute("file_writer", {"path": fpath, "content": "Line 2\n", "mode": "append"})

        result = executor.execute("file_reader", {"path": fpath})
        assert "Line 1" in result
        assert "Line 2" in result

    def test_read_nonexistent(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("file_reader", {"path": "/nonexistent_xyz.txt"})
        assert "Error" in result

    def test_unsupported_extension(self, tmp_path):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("file_reader", {"path": str(tmp_path / "binary.exe")})
        assert "Error" in result


class TestWebFetch:
    """Tests for web_fetch tool (no real HTTP — test validation only)."""

    def test_rejects_private_scheme(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("web_fetch", {"url": "ftp://example.com"})
        assert "Error" in result

    def test_rejects_no_hostname(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("web_fetch", {"url": "http://"})
        assert "Error" in result

    def test_rejects_localhost(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("web_fetch", {"url": "http://127.0.0.1/admin"})
        assert "Error" in result


class TestShellCommand:
    """Tests for shell_command tool."""

    def test_echo(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("shell_command", {"command": "echo hello"})
        assert "hello" in result

    def test_blocked_command(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("shell_command", {"command": "curl http://evil.com"})
        assert "Error" in result

    def test_dangerous_pattern_blocked(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("shell_command", {"command": "echo x | rm -rf /"})
        assert "Error" in result

    def test_empty_command(self):
        from pinocchio.tools import ToolRegistry, ToolExecutor

        registry = ToolRegistry()
        registry.register_defaults()
        executor = ToolExecutor(registry)

        result = executor.execute("shell_command", {"command": ""})
        assert "Error" in result


# =====================================================================
# Tests for provider presets in config
# =====================================================================


class TestProviderPresets:
    """Tests for PinocchioConfig provider preset system."""

    def test_default_config(self):
        from config import PinocchioConfig

        cfg = PinocchioConfig()
        assert cfg.model  # has a model

    def test_from_provider_openai(self):
        from config import PinocchioConfig

        cfg = PinocchioConfig.from_provider("openai")
        assert cfg.model == "gpt-4o"
        assert "openai.com" in cfg.base_url

    def test_from_provider_deepseek(self):
        from config import PinocchioConfig

        cfg = PinocchioConfig.from_provider("deepseek")
        assert cfg.model == "deepseek-chat"
        assert "deepseek.com" in cfg.base_url

    def test_from_provider_with_model_override(self):
        from config import PinocchioConfig

        cfg = PinocchioConfig.from_provider("openai", model="gpt-4o-mini")
        assert cfg.model == "gpt-4o-mini"

    def test_from_provider_unknown_raises(self):
        from config import PinocchioConfig

        with pytest.raises(ValueError, match="Unknown provider"):
            PinocchioConfig.from_provider("nonexistent_provider")

    def test_available_providers(self):
        from config import PinocchioConfig

        providers = PinocchioConfig.available_providers()
        assert "ollama" in providers
        assert "openai" in providers
        assert "deepseek" in providers
        assert len(providers) >= 7
