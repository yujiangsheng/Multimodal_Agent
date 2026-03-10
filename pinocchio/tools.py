"""Tool Framework — registry and executor for callable tools.

Provides a lightweight tool system that allows Pinocchio to invoke
external capabilities (calculators, web search, code execution, etc.)
during the EXECUTE phase of the cognitive loop.

Architecture
------------
1. **Tool** — A dataclass describing a callable: name, description,
   parameters schema, and the Python function to invoke.
2. **ToolRegistry** — Registry that holds all available tools and
   provides lookup / schema export / enable-disable.
3. **ToolExecutor** — Executes a tool call safely with timeout,
   error handling, result formatting, and usage metrics.
4. **@tool** decorator — Convenient way to register functions as tools
   with auto-generated parameter schemas.

Built-in tools (registered by default):
- ``calculator`` — evaluate simple math expressions
- ``current_time`` — return the current date and time

Example
-------
>>> from pinocchio.tools import ToolRegistry, ToolExecutor, tool
>>> registry = ToolRegistry()
>>> registry.register_defaults()
>>>
>>> @tool(registry, description="Greet someone")
... def greet(name: str) -> str:
...     return f"Hello, {name}!"
>>>
>>> executor = ToolExecutor(registry)
>>> result = executor.execute("greet", {"name": "Alice"})
>>> print(result)  # "Hello, Alice!"
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import math
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


# =====================================================================
# Tool dataclass
# =====================================================================

@dataclass
class Tool:
    """A single callable tool available to the agent.

    Attributes
    ----------
    name : Unique identifier for the tool.
    description : Human-readable description shown to the LLM.
    parameters : JSON Schema dict describing accepted parameters.
    function : The Python callable to invoke.
    enabled : Whether the tool is currently active.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., str]
    enabled: bool = True


# =====================================================================
# @tool decorator
# =====================================================================

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


def _build_schema_from_function(func: Callable) -> dict[str, Any]:
    """Auto-generate a JSON Schema 'parameters' block from function signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        json_type = _PYTHON_TYPE_TO_JSON.get(hints.get(name, str), "string")
        prop: dict[str, Any] = {"type": json_type}
        # Use docstring-style annotation if available
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    registry: ToolRegistry | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable:
    """Decorator to register a function as a Tool.

    Usage::

        @tool(registry, description="Add two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

    Or without a registry (register later manually)::

        @tool(description="Multiply")
        def multiply(a: int, b: int) -> str:
            return str(a * b)

        # The Tool is stored as multiply._tool
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        tool_params = parameters or _build_schema_from_function(func)

        t = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            function=func,
        )
        func._tool = t  # type: ignore[attr-defined]

        if registry is not None:
            registry.register(t)

        return func

    # Allow @tool(registry) without keyword — detect if first arg is callable
    if callable(registry):
        func = registry
        registry = None
        return decorator(func)

    return decorator


# =====================================================================
# Usage metric
# =====================================================================

@dataclass
class ToolUsageRecord:
    """A single invocation record for metrics tracking."""
    tool_name: str
    arguments: dict[str, Any]
    result: str
    success: bool
    duration_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


# =====================================================================
# Registry
# =====================================================================

class ToolRegistry:
    """In-memory registry of available tools.

    Provides schema export in the OpenAI ``tools`` format so the LLM
    can decide which tool to call.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- Registration --------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool (overwrites if name already exists)."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def register_function(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        """Convenience: wrap a plain function and register it.

        Returns the created Tool so callers can inspect or modify it.
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
        tool_params = parameters or _build_schema_from_function(func)
        t = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            function=func,
        )
        self.register(t)
        return t

    # -- Lookup --------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_enabled(self) -> list[str]:
        """Return names of tools that are currently enabled."""
        return [n for n, t in self._tools.items() if t.enabled]

    # -- Enable / disable ----------------------------------------------

    def enable(self, name: str) -> bool:
        """Enable a tool by name. Returns False if not found."""
        t = self._tools.get(name)
        if t is None:
            return False
        t.enabled = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a tool by name. Returns False if not found."""
        t = self._tools.get(name)
        if t is None:
            return False
        t.enabled = False
        return True

    # -- Schema export -------------------------------------------------

    def to_openai_schema(self) -> list[dict[str, Any]]:
        """Export enabled tools as an OpenAI-compatible ``tools`` list."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
            if t.enabled
        ]

    def to_prompt_description(self) -> str:
        """Return a plain-text listing of enabled tools for prompt injection."""
        lines: list[str] = []
        for t in self._tools.values():
            if not t.enabled:
                continue
            params = ", ".join(
                f"{k}: {v.get('type', 'string')}"
                for k, v in t.parameters.get("properties", {}).items()
            )
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def register_defaults(self) -> None:
        """Register the built-in tools (copies, so each registry is independent)."""
        import copy
        self.register(copy.copy(_CALCULATOR_TOOL))
        self.register(copy.copy(_CURRENT_TIME_TOOL))

    @property
    def count(self) -> int:
        return len(self._tools)

    @property
    def enabled_count(self) -> int:
        return sum(1 for t in self._tools.values() if t.enabled)


# =====================================================================
# Executor
# =====================================================================

_DEFAULT_TIMEOUT = 30.0  # seconds


class ToolExecutor:
    """Execute tool calls safely with timeout, error handling, and metrics."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_history: int = 200,
    ) -> None:
        self.registry = registry
        self.timeout = timeout
        self._history: list[ToolUsageRecord] = []
        self._max_history = max_history

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        """Invoke a tool by name and return the string result.

        Returns an error message (not an exception) if the tool is
        unknown, disabled, or fails, so the LLM can recover gracefully.
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            return f"[Error] Unknown tool: {tool_name}"
        if not tool.enabled:
            return f"[Error] Tool '{tool_name}' is currently disabled"

        effective_timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()
        success = False
        result = ""

        try:
            result = self._run_with_timeout(tool.function, arguments, effective_timeout)
            success = "[Error]" not in result
        except Exception as exc:
            result = f"[Error] Tool '{tool_name}' failed: {exc}"

        duration_ms = (time.monotonic() - start) * 1000
        self._record(tool_name, arguments, result, success, duration_ms)
        return result

    async def async_execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        """Async version of execute — runs the tool in a thread executor."""
        tool = self.registry.get(tool_name)
        if tool is None:
            return f"[Error] Unknown tool: {tool_name}"
        if not tool.enabled:
            return f"[Error] Tool '{tool_name}' is currently disabled"

        effective_timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()
        success = False
        result = ""

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(tool.function, **arguments),
                timeout=effective_timeout,
            )
            result = str(result)
            success = "[Error]" not in result
        except asyncio.TimeoutError:
            result = f"[Error] Tool '{tool_name}' timed out after {effective_timeout}s"
        except Exception as exc:
            result = f"[Error] Tool '{tool_name}' failed: {exc}"

        duration_ms = (time.monotonic() - start) * 1000
        self._record(tool_name, arguments, result, success, duration_ms)
        return result

    def parse_and_execute(self, raw_json: str) -> str:
        """Parse a JSON tool-call string and execute it.

        Expected format: ``{"tool": "name", "arguments": {...}}``
        """
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            return f"[Error] Invalid JSON: {raw_json[:200]}"
        tool_name = parsed.get("tool", "")
        arguments = parsed.get("arguments", {})
        return self.execute(tool_name, arguments)

    # -- Metrics -------------------------------------------------------

    @property
    def history(self) -> list[ToolUsageRecord]:
        """Return recent tool invocation history."""
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        """Return aggregated usage statistics."""
        if not self._history:
            return {"total_calls": 0, "by_tool": {}}
        by_tool: dict[str, dict[str, Any]] = {}
        for rec in self._history:
            entry = by_tool.setdefault(rec.tool_name, {
                "calls": 0, "successes": 0, "total_ms": 0.0,
            })
            entry["calls"] += 1
            if rec.success:
                entry["successes"] += 1
            entry["total_ms"] += rec.duration_ms
        # Compute averages
        for entry in by_tool.values():
            entry["avg_ms"] = round(entry["total_ms"] / entry["calls"], 1) if entry["calls"] else 0
            entry["success_rate"] = round(entry["successes"] / entry["calls"], 2) if entry["calls"] else 0
        return {"total_calls": len(self._history), "by_tool": by_tool}

    def clear_history(self) -> None:
        self._history.clear()

    # -- Internal ------------------------------------------------------

    def _run_with_timeout(
        self, func: Callable, arguments: dict[str, Any], timeout: float,
    ) -> str:
        """Run a tool function with a thread-based timeout."""
        result_container: list[str | BaseException] = []

        def target() -> None:
            try:
                result_container.append(str(func(**arguments)))
            except Exception as exc:
                result_container.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return f"[Error] Tool timed out after {timeout}s"
        if not result_container:
            return "[Error] Tool returned no result"
        val = result_container[0]
        if isinstance(val, BaseException):
            return f"[Error] Tool failed: {val}"
        return val

    def _record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a usage metric, capping history size."""
        self._history.append(ToolUsageRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result[:500],  # cap stored result length
            success=success,
            duration_ms=round(duration_ms, 2),
        ))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]


# =====================================================================
# Built-in Tools
# =====================================================================

# Allowlist for safe math evaluation
_SAFE_MATH_NAMES: dict[str, Any] = {
    k: getattr(math, k) for k in [
        "sqrt", "sin", "cos", "tan", "log", "log2", "log10",
        "exp", "pow", "ceil", "floor", "pi", "e", "inf",
        "factorial", "gcd",
    ]
}
_SAFE_MATH_NAMES["abs"] = abs
_SAFE_MATH_NAMES["round"] = round
_SAFE_MATH_NAMES["min"] = min
_SAFE_MATH_NAMES["max"] = max


def _calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    # Only allow digits, operators, parentheses, dots, and known function names
    sanitized = expression.strip()
    if not re.match(r'^[\d\s\+\-\*/\.\(\),a-zA-Z_]+$', sanitized):
        return f"[Error] Invalid expression: {sanitized}"
    try:
        result = eval(sanitized, {"__builtins__": {}}, _SAFE_MATH_NAMES)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"[Error] Calculation failed: {exc}"


_CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, log, etc.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. 'sqrt(144) + 2**10'",
            },
        },
        "required": ["expression"],
    },
    function=_calculator,
)


def _current_time() -> str:
    """Return the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


_CURRENT_TIME_TOOL = Tool(
    name="current_time",
    description="Return the current date and time.",
    parameters={
        "type": "object",
        "properties": {},
    },
    function=_current_time,
)
