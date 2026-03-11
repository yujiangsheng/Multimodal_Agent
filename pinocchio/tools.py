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
import base64
import datetime
import hashlib
import inspect
import ipaddress
import json
import math
import os
import platform
import random
import re
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints
from urllib.parse import urlparse


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
        for t in _ALL_DEFAULT_TOOLS:
            self.register(copy.copy(t))

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


# ── web_fetch ────────────────────────────────────────────────────────

def _is_private_ip(hostname: str) -> bool:
    """Check whether *hostname* resolves to a private / loopback address (SSRF guard)."""
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
                return True
    except (socket.gaierror, ValueError):
        return True  # unresolvable → deny
    return False


def _web_fetch(url: str, max_length: int = 8000) -> str:
    """Fetch text content from a URL (with SSRF protection)."""
    try:
        import httpx  # already a project dependency
    except ImportError:
        return "[Error] httpx is not installed"

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"[Error] Unsupported scheme: {parsed.scheme}"
    hostname = parsed.hostname or ""
    if not hostname:
        return "[Error] Invalid URL — no hostname"
    if _is_private_ip(hostname):
        return "[Error] Access to private/internal addresses is not allowed"

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True, max_redirects=5) as client:
            resp = client.get(url, headers={"User-Agent": "Pinocchio-Agent/0.3"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text" in content_type or "json" in content_type or "xml" in content_type:
                text = resp.text[:max_length]
            else:
                text = f"[Binary content: {content_type}, {len(resp.content)} bytes]"
            return text
    except httpx.HTTPStatusError as exc:
        return f"[Error] HTTP {exc.response.status_code}: {exc.response.reason_phrase}"
    except Exception as exc:
        return f"[Error] Fetch failed: {exc}"


_WEB_FETCH_TOOL = Tool(
    name="web_fetch",
    description="Fetch and return text content from a URL. Supports HTTP/HTTPS. Returns up to max_length characters.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch, e.g. 'https://example.com'",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum characters to return (default 8000)",
                "default": 8000,
            },
        },
        "required": ["url"],
    },
    function=_web_fetch,
)


# ── file_reader ──────────────────────────────────────────────────────

# Allowed extensions for safe file reading
_READABLE_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".csv", ".xml", ".html", ".css",
    ".sh", ".bash", ".log", ".rst", ".tex", ".sql", ".r", ".rb",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".swift", ".kt",
}


def _file_reader(path: str, max_length: int = 10000) -> str:
    """Read text content from a file."""
    p = Path(path).resolve()
    if not p.exists():
        return f"[Error] File not found: {path}"
    if not p.is_file():
        return f"[Error] Not a file: {path}"
    if p.suffix.lower() not in _READABLE_EXTENSIONS:
        return f"[Error] Unsupported file type: {p.suffix}. Allowed: {', '.join(sorted(_READABLE_EXTENSIONS))}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_length:
            return text[:max_length] + f"\n\n[... truncated, {len(text)} total chars]"
        return text
    except Exception as exc:
        return f"[Error] Read failed: {exc}"


_FILE_READER_TOOL = Tool(
    name="file_reader",
    description="Read text content from a file. Supports common text/code file types.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum characters to return (default 10000)",
                "default": 10000,
            },
        },
        "required": ["path"],
    },
    function=_file_reader,
)


# ── file_writer ──────────────────────────────────────────────────────

def _file_writer(path: str, content: str, mode: str = "write") -> str:
    """Write or append content to a text file."""
    p = Path(path).resolve()
    if p.suffix.lower() not in _READABLE_EXTENSIONS:
        return f"[Error] Unsupported file type: {p.suffix}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"[Error] Write failed: {exc}"


_FILE_WRITER_TOOL = Tool(
    name="file_writer",
    description="Write or append text content to a file. Supports common text/code file types.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The text content to write",
            },
            "mode": {
                "type": "string",
                "description": "'write' (overwrite) or 'append' (add to end). Default: 'write'",
                "default": "write",
            },
        },
        "required": ["path", "content"],
    },
    function=_file_writer,
)


# ── json_formatter ───────────────────────────────────────────────────

def _json_formatter(text: str, indent: int = 2) -> str:
    """Pretty-print and validate JSON."""
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as exc:
        return f"[Error] Invalid JSON: {exc}"


_JSON_FORMATTER_TOOL = Tool(
    name="json_formatter",
    description="Validate and pretty-print a JSON string.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The JSON string to format",
            },
            "indent": {
                "type": "integer",
                "description": "Indentation spaces (default 2)",
                "default": 2,
            },
        },
        "required": ["text"],
    },
    function=_json_formatter,
)


# ── text_stats ───────────────────────────────────────────────────────

def _text_stats(text: str) -> str:
    """Compute word count, character count, line count, and sentence count."""
    lines = text.split("\n")
    words = text.split()
    sentences = re.split(r'[.!?。！？]+', text)
    sentences = [s for s in sentences if s.strip()]
    return json.dumps({
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
        "sentences": len(sentences),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
    }, ensure_ascii=False)


_TEXT_STATS_TOOL = Tool(
    name="text_stats",
    description="Count characters, words, lines, sentences, and paragraphs in text.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze",
            },
        },
        "required": ["text"],
    },
    function=_text_stats,
)


# ── hash_calculator ──────────────────────────────────────────────────

_HASH_ALGOS = {"md5", "sha1", "sha256", "sha512"}


def _hash_calculator(text: str, algorithm: str = "sha256") -> str:
    """Calculate a cryptographic hash of the input text."""
    algo = algorithm.lower().strip()
    if algo not in _HASH_ALGOS:
        return f"[Error] Unsupported algorithm: {algo}. Supported: {', '.join(sorted(_HASH_ALGOS))}"
    h = hashlib.new(algo, text.encode("utf-8"), usedforsecurity=False)
    return f"{algo}:{h.hexdigest()}"


_HASH_CALCULATOR_TOOL = Tool(
    name="hash_calculator",
    description="Calculate a cryptographic hash (md5, sha1, sha256, sha512) of the input text.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to hash",
            },
            "algorithm": {
                "type": "string",
                "description": "Hash algorithm: md5, sha1, sha256, sha512 (default: sha256)",
                "default": "sha256",
            },
        },
        "required": ["text"],
    },
    function=_hash_calculator,
)


# ── uuid_generator ───────────────────────────────────────────────────

def _uuid_generator(version: int = 4) -> str:
    """Generate a UUID."""
    if version == 4:
        return str(uuid.uuid4())
    if version == 1:
        return str(uuid.uuid1())
    return f"[Error] Unsupported UUID version: {version}. Supported: 1, 4"


_UUID_GENERATOR_TOOL = Tool(
    name="uuid_generator",
    description="Generate a UUID (v1 or v4).",
    parameters={
        "type": "object",
        "properties": {
            "version": {
                "type": "integer",
                "description": "UUID version: 1 (time-based) or 4 (random). Default: 4",
                "default": 4,
            },
        },
    },
    function=_uuid_generator,
)


# ── base64_codec ─────────────────────────────────────────────────────

def _base64_codec(text: str, action: str = "encode") -> str:
    """Encode or decode Base64."""
    if action == "encode":
        return base64.b64encode(text.encode("utf-8")).decode("ascii")
    if action == "decode":
        try:
            return base64.b64decode(text).decode("utf-8")
        except Exception as exc:
            return f"[Error] Base64 decode failed: {exc}"
    return f"[Error] Unknown action: {action}. Use 'encode' or 'decode'."


_BASE64_CODEC_TOOL = Tool(
    name="base64_codec",
    description="Encode text to Base64 or decode Base64 back to text.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to encode, or the Base64 string to decode",
            },
            "action": {
                "type": "string",
                "description": "'encode' or 'decode' (default: 'encode')",
                "default": "encode",
            },
        },
        "required": ["text"],
    },
    function=_base64_codec,
)


# ── unit_converter ───────────────────────────────────────────────────

# Conversion factors to SI base units
_LENGTH_TO_M: dict[str, float] = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
}
_WEIGHT_TO_KG: dict[str, float] = {
    "mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0,
    "oz": 0.0283495, "lb": 0.453592,
}
_VOLUME_TO_L: dict[str, float] = {
    "ml": 0.001, "l": 1.0, "gal": 3.78541, "qt": 0.946353,
    "pt": 0.473176, "cup": 0.236588, "fl_oz": 0.0295735,
}


def _unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units (length, weight, volume, temperature)."""
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    # Temperature special case
    temp_units = {"c", "f", "k", "celsius", "fahrenheit", "kelvin"}
    f_norm = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}.get(f, f)
    t_norm = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}.get(t, t)

    if f_norm in temp_units and t_norm in temp_units:
        # Convert to Celsius first
        if f_norm == "c":
            c = value
        elif f_norm == "f":
            c = (value - 32) * 5 / 9
        elif f_norm == "k":
            c = value - 273.15
        else:
            return f"[Error] Unknown temperature unit: {f}"
        # Convert from Celsius to target
        if t_norm == "c":
            result = c
        elif t_norm == "f":
            result = c * 9 / 5 + 32
        elif t_norm == "k":
            result = c + 273.15
        else:
            return f"[Error] Unknown temperature unit: {t}"
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

    # Linear conversions
    for table, label in [(_LENGTH_TO_M, "length"), (_WEIGHT_TO_KG, "weight"), (_VOLUME_TO_L, "volume")]:
        if f in table and t in table:
            result = value * table[f] / table[t]
            return f"{value} {from_unit} = {round(result, 6)} {to_unit}"

    return f"[Error] Cannot convert '{from_unit}' to '{to_unit}'. Supported: length ({', '.join(_LENGTH_TO_M)}), weight ({', '.join(_WEIGHT_TO_KG)}), volume ({', '.join(_VOLUME_TO_L)}), temperature (C, F, K)"


_UNIT_CONVERTER_TOOL = Tool(
    name="unit_converter",
    description="Convert between common units: length (mm/cm/m/km/in/ft/yd/mi), weight (mg/g/kg/t/oz/lb), volume (ml/l/gal/qt/pt/cup/fl_oz), temperature (C/F/K).",
    parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit, e.g. 'km', 'lb', 'F'",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit, e.g. 'mi', 'kg', 'C'",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    function=_unit_converter,
)


# ── random_number ────────────────────────────────────────────────────

def _random_number(min_val: float = 0.0, max_val: float = 100.0, count: int = 1, integer: bool = True) -> str:
    """Generate random numbers."""
    count = max(1, min(count, 100))
    if integer:
        results = [random.randint(int(min_val), int(max_val)) for _ in range(count)]
    else:
        results = [round(random.uniform(min_val, max_val), 6) for _ in range(count)]
    if count == 1:
        return str(results[0])
    return json.dumps(results)


_RANDOM_NUMBER_TOOL = Tool(
    name="random_number",
    description="Generate one or more random numbers within a range.",
    parameters={
        "type": "object",
        "properties": {
            "min_val": {
                "type": "number",
                "description": "Minimum value (default 0)",
                "default": 0,
            },
            "max_val": {
                "type": "number",
                "description": "Maximum value (default 100)",
                "default": 100,
            },
            "count": {
                "type": "integer",
                "description": "How many numbers to generate (default 1, max 100)",
                "default": 1,
            },
            "integer": {
                "type": "boolean",
                "description": "If true, return integers; if false, return floats (default true)",
                "default": True,
            },
        },
    },
    function=_random_number,
)


# ── regex_match ──────────────────────────────────────────────────────

def _regex_match(pattern: str, text: str, find_all: bool = False) -> str:
    """Test a regex pattern against text."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"[Error] Invalid regex: {exc}"

    if find_all:
        matches = compiled.findall(text)
        return json.dumps({"matches": matches, "count": len(matches)}, ensure_ascii=False)

    m = compiled.search(text)
    if m is None:
        return json.dumps({"matched": False}, ensure_ascii=False)
    return json.dumps({
        "matched": True,
        "match": m.group(),
        "start": m.start(),
        "end": m.end(),
        "groups": list(m.groups()),
    }, ensure_ascii=False)


_REGEX_MATCH_TOOL = Tool(
    name="regex_match",
    description="Test a regular expression pattern against text. Returns match details or all matches.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to test",
            },
            "text": {
                "type": "string",
                "description": "The text to search",
            },
            "find_all": {
                "type": "boolean",
                "description": "If true, find all matches; if false, find first match only (default false)",
                "default": False,
            },
        },
        "required": ["pattern", "text"],
    },
    function=_regex_match,
)


# ── system_info ──────────────────────────────────────────────────────

def _system_info() -> str:
    """Return system information."""
    return json.dumps({
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "processor": platform.processor() or "unknown",
        "cwd": os.getcwd(),
    }, ensure_ascii=False)


_SYSTEM_INFO_TOOL = Tool(
    name="system_info",
    description="Return system information: OS, architecture, Python version, hostname.",
    parameters={
        "type": "object",
        "properties": {},
    },
    function=_system_info,
)


# ── directory_listing ────────────────────────────────────────────────

def _directory_listing(path: str = ".", max_items: int = 100) -> str:
    """List files and directories in a given path."""
    p = Path(path).resolve()
    if not p.exists():
        return f"[Error] Path not found: {path}"
    if not p.is_dir():
        return f"[Error] Not a directory: {path}"
    try:
        items = []
        for i, entry in enumerate(sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))):
            if i >= max_items:
                items.append(f"... and {sum(1 for _ in p.iterdir()) - max_items} more items")
                break
            suffix = "/" if entry.is_dir() else f" ({entry.stat().st_size} bytes)"
            items.append(f"{entry.name}{suffix}")
        return "\n".join(items) if items else "(empty directory)"
    except PermissionError:
        return f"[Error] Permission denied: {path}"
    except Exception as exc:
        return f"[Error] {exc}"


_DIRECTORY_LISTING_TOOL = Tool(
    name="directory_listing",
    description="List files and directories in a given path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current directory)",
                "default": ".",
            },
            "max_items": {
                "type": "integer",
                "description": "Maximum items to return (default 100)",
                "default": 100,
            },
        },
    },
    function=_directory_listing,
)


# ── shell_command ────────────────────────────────────────────────────

# Allowlist of safe commands
_SAFE_COMMANDS = {
    "echo", "date", "whoami", "hostname", "uname", "env",
    "cat", "head", "tail", "wc", "sort", "uniq", "grep",
    "find", "ls", "pwd", "which", "file", "du", "df",
    "python", "python3", "pip", "pip3", "node", "npm",
    "git",
}

# Dangerous patterns to block
_DANGEROUS_PATTERNS = re.compile(
    r"rm\s+-rf|mkfs|dd\s+if=|:\(\)\{|>\s*/dev/|chmod\s+777|curl.*\|\s*sh|wget.*\|\s*sh",
    re.IGNORECASE,
)


def _shell_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command safely (allowlisted commands only)."""
    import subprocess

    command = command.strip()
    if not command:
        return "[Error] Empty command"

    # Extract the base command
    base_cmd = command.split()[0].split("/")[-1]
    if base_cmd not in _SAFE_COMMANDS:
        return f"[Error] Command '{base_cmd}' is not in the allowed list. Allowed: {', '.join(sorted(_SAFE_COMMANDS))}"

    if _DANGEROUS_PATTERNS.search(command):
        return "[Error] Command contains dangerous patterns and was blocked"

    timeout = max(1, min(timeout, 60))
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output[:10000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[Error] Command timed out after {timeout}s"
    except Exception as exc:
        return f"[Error] Command failed: {exc}"


_SHELL_COMMAND_TOOL = Tool(
    name="shell_command",
    description="Execute a safe shell command. Only allowlisted commands (ls, cat, grep, git, python, etc.) are permitted.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30, max 60)",
                "default": 30,
            },
        },
        "required": ["command"],
    },
    function=_shell_command,
)


# =====================================================================
# All default tools — referenced by register_defaults()
# =====================================================================

_ALL_DEFAULT_TOOLS: list[Tool] = [
    _CALCULATOR_TOOL,
    _CURRENT_TIME_TOOL,
    _WEB_FETCH_TOOL,
    _FILE_READER_TOOL,
    _FILE_WRITER_TOOL,
    _JSON_FORMATTER_TOOL,
    _TEXT_STATS_TOOL,
    _HASH_CALCULATOR_TOOL,
    _UUID_GENERATOR_TOOL,
    _BASE64_CODEC_TOOL,
    _UNIT_CONVERTER_TOOL,
    _RANDOM_NUMBER_TOOL,
    _REGEX_MATCH_TOOL,
    _SYSTEM_INFO_TOOL,
    _DIRECTORY_LISTING_TOOL,
    _SHELL_COMMAND_TOOL,
]
