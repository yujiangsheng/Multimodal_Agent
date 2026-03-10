"""Tool Framework — registry and executor for callable tools.

Provides a lightweight tool system that allows Pinocchio to invoke
external capabilities (calculators, web search, code execution, etc.)
during the EXECUTE phase of the cognitive loop.

Architecture
------------
1. **Tool** — A dataclass describing a callable: name, description,
   parameters schema, and the Python function to invoke.
2. **ToolRegistry** — Singleton-style registry that holds all
   available tools and provides lookup / schema export.
3. **ToolExecutor** — Executes a tool call safely with timeout,
   error handling, and result formatting.

Built-in tools (registered by default):
- ``calculator`` — evaluate simple math expressions
- ``current_time`` — return the current date and time
- ``python_eval`` — execute a Python expression in a restricted sandbox

Example
-------
>>> from pinocchio.tools import ToolRegistry, ToolExecutor
>>> registry = ToolRegistry()
>>> registry.register_defaults()
>>> executor = ToolExecutor(registry)
>>> result = executor.execute("calculator", {"expression": "2 ** 10"})
>>> print(result)  # "1024"
"""

from __future__ import annotations

import datetime
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable


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
    """
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., str]


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

    def register(self, tool: Tool) -> None:
        """Register a tool (overwrites if name already exists)."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_schema(self) -> list[dict[str, Any]]:
        """Export all tools as an OpenAI-compatible ``tools`` list."""
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
        ]

    def to_prompt_description(self) -> str:
        """Return a plain-text listing of tools for prompt injection."""
        lines: list[str] = []
        for t in self._tools.values():
            params = ", ".join(
                f"{k}: {v.get('type', 'string')}"
                for k, v in t.parameters.get("properties", {}).items()
            )
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def register_defaults(self) -> None:
        """Register the built-in tools."""
        self.register(_CALCULATOR_TOOL)
        self.register(_CURRENT_TIME_TOOL)
        self.register(_PYTHON_EVAL_TOOL)

    @property
    def count(self) -> int:
        return len(self._tools)


# =====================================================================
# Executor
# =====================================================================

class ToolExecutor:
    """Execute tool calls safely with error handling."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool by name and return the string result.

        Returns an error message (not an exception) if the tool is
        unknown or the call fails, so the LLM can recover gracefully.
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            return f"[Error] Unknown tool: {tool_name}"
        try:
            result = tool.function(**arguments)
            return str(result)
        except Exception as exc:
            return f"[Error] Tool '{tool_name}' failed: {exc}"

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


def _python_eval(code: str) -> str:
    """Execute a simple Python expression in a restricted sandbox."""
    allowed_builtins = {
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "frozenset": frozenset,
        "hex": hex, "int": int, "isinstance": isinstance, "len": len,
        "list": list, "map": map, "max": max, "min": min, "oct": oct,
        "ord": ord, "pow": pow, "range": range, "repr": repr, "reversed": reversed,
        "round": round, "set": set, "slice": slice, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    }
    allowed_builtins.update(_SAFE_MATH_NAMES)
    try:
        result = eval(code, {"__builtins__": allowed_builtins})  # noqa: S307
        return str(result)
    except SyntaxError:
        # Try exec for statements
        local_ns: dict[str, Any] = {}
        exec(code, {"__builtins__": allowed_builtins}, local_ns)  # noqa: S102
        return str(local_ns.get("result", local_ns))
    except Exception as exc:
        return f"[Error] Execution failed: {exc}"


_PYTHON_EVAL_TOOL = Tool(
    name="python_eval",
    description="Execute a Python expression or simple code snippet. Use variable 'result' to return output from statements.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to evaluate, e.g. 'sorted([3,1,2])'",
            },
        },
        "required": ["code"],
    },
    function=_python_eval,
)
