"""Structured tracing — Span/Trace/Tracer for Pinocchio.

Implements a lightweight tracing system inspired by OpenTelemetry concepts:
- **Trace**: A complete end-to-end request/interaction.
- **Span**: A timed operation within a trace (can be nested).
- **Tracer**: Global tracer that manages traces and spans.

Each span records:
- start/end timestamps
- status (OK, ERROR, SKIPPED)
- arbitrary key-value attributes
- events (timestamped log entries)
- parent span ID (for nesting)

Usage::

    tracer = Tracer()

    with tracer.start_trace("chat_request") as trace:
        with trace.span("perceive") as s:
            s.set_attribute("modality", "text")
            result = perceive(...)
            s.set_attribute("task_type", result.task_type)

        with trace.span("execute") as s:
            response = execute(...)
            s.add_event("tool_call", {"tool": "calculator"})

    # Export
    print(tracer.export_json())
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from collections.abc import Generator


class SpanStatus(str, Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class SpanEvent:
    """A timestamped event within a span."""
    name: str = ""
    timestamp: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """A timed operation within a trace.

    Spans can be nested: each span has an optional parent_id.
    Use as a context manager for automatic timing.
    """

    span_id: str = ""
    trace_id: str = ""
    parent_id: str = ""
    name: str = ""
    status: SpanStatus = SpanStatus.OK
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-generate span_id and start_time if not provided."""
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:12]
        if not self.start_time:
            self.start_time = time.time()

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock duration of this span in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key-value attribute to this span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record a timestamped event within this span."""
        self.events.append(SpanEvent(
            name=name, timestamp=time.time(),
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, error: str = "") -> None:
        """Set the span status; optionally attach an error message."""
        self.status = status
        if error:
            self.attributes["error.message"] = error

    def end(self) -> None:
        """Mark the span as finished (records the end timestamp)."""
        if not self.end_time:
            self.end_time = time.time()

    def __enter__(self) -> Span:
        """Context-manager entry — returns the span itself."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context-manager exit — records errors and ends the span."""
        if exc_type:
            self.set_status(SpanStatus.ERROR, str(exc_val))
        self.end()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the span to a JSON-friendly dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
        }


class Trace:
    """A complete end-to-end trace (collection of spans).

    Use ``span()`` as a context manager to create child spans.
    """

    def __init__(self, name: str = "", trace_id: str = "") -> None:
        """Create a new trace.

        Args:
            name: Human-readable label for this trace.
            trace_id: Optional explicit ID; auto-generated if omitted.
        """
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.name = name
        self._spans: list[Span] = []
        self._span_stack: list[Span] = []
        self._start_time = time.time()
        self._end_time = 0.0
        self._status = SpanStatus.OK

    @contextmanager
    def span(self, name: str) -> Generator[Span, None, None]:
        """Create a child span within this trace."""
        parent_id = self._span_stack[-1].span_id if self._span_stack else ""
        s = Span(
            trace_id=self.trace_id,
            parent_id=parent_id,
            name=name,
        )
        self._spans.append(s)
        self._span_stack.append(s)
        try:
            yield s
        except Exception:
            s.set_status(SpanStatus.ERROR)
            self._status = SpanStatus.ERROR
            raise
        finally:
            s.end()
            self._span_stack.pop()

    def add_span(self, span: Span) -> None:
        """Add a pre-built span to this trace."""
        span.trace_id = self.trace_id
        self._spans.append(span)

    @property
    def spans(self) -> list[Span]:
        """Snapshot of all spans in this trace."""
        return list(self._spans)

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock duration of the entire trace in milliseconds."""
        end = self._end_time or time.time()
        return (end - self._start_time) * 1000

    @property
    def status(self) -> SpanStatus:
        """Overall status — ERROR if any span errored."""
        return self._status

    def end(self) -> None:
        """Mark the trace as finished."""
        self._end_time = time.time()

    def __enter__(self) -> Trace:
        """Context-manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context-manager exit — records error status and ends trace."""
        if exc_type:
            self._status = SpanStatus.ERROR
        self.end()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the trace (with all spans) to a JSON-friendly dict."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "status": self._status.value,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "span_count": len(self._spans),
            "spans": [s.to_dict() for s in self._spans],
        }

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"Trace({self.name}) {self._status.value} "
            f"{len(self._spans)} spans "
            f"{self.elapsed_ms:.0f}ms"
        )


class Tracer:
    """Global tracer that manages traces and provides export.

    Thread-safe: traces are append-only and each trace is independent.
    """

    def __init__(self, max_traces: int = 1000, ttl_seconds: float = 3600.0) -> None:
        """Create a tracer that retains up to *max_traces* traces.

        Parameters
        ----------
        max_traces : int
            Maximum number of traces to keep.
        ttl_seconds : float
            Time-to-live in seconds.  Traces older than this are
            automatically evicted on every ``start_trace`` / ``create_trace``
            call.  Defaults to 1 hour.
        """
        self._traces: list[Trace] = []
        self._max_traces = max_traces
        self._ttl_seconds = ttl_seconds

    def _evict(self) -> None:
        """Remove traces that exceed max count or TTL."""
        import time as _time
        cutoff = _time.time() - self._ttl_seconds
        self._traces = [
            t for t in self._traces
            if t._start_time >= cutoff
        ]
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

    @contextmanager
    def start_trace(self, name: str = "interaction") -> Generator[Trace, None, None]:
        """Start a new trace. Use as a context manager."""
        trace = Trace(name=name)
        self._traces.append(trace)
        self._evict()
        try:
            yield trace
        finally:
            trace.end()

    def create_trace(self, name: str = "interaction") -> Trace:
        """Create and register a new trace (non-context-manager variant)."""
        trace = Trace(name=name)
        self._traces.append(trace)
        self._evict()
        return trace

    @property
    def traces(self) -> list[Trace]:
        """Snapshot of all recorded traces."""
        return list(self._traces)

    @property
    def trace_count(self) -> int:
        """Number of traces recorded so far."""
        return len(self._traces)

    def latest(self) -> Trace | None:
        """Return the most recently created trace, or ``None``."""
        return self._traces[-1] if self._traces else None

    def clear(self) -> None:
        """Discard all recorded traces."""
        self._traces.clear()

    def export_json(self, pretty: bool = False) -> str:
        """Export all traces as JSON."""
        data = [t.to_dict() for t in self._traces]
        return json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)

    def export_summary(self) -> str:
        """Export a human-readable summary of all traces."""
        lines = [f"Tracer: {len(self._traces)} traces"]
        for t in self._traces[-10:]:  # last 10
            lines.append(f"  {t.summary()}")
            for s in t.spans:
                indent = "    " if not s.parent_id else "      "
                status = s.status.value
                lines.append(f"{indent}[{status}] {s.name} {s.elapsed_ms:.0f}ms")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all traces."""
        if not self._traces:
            return {"traces": 0}

        total_spans = sum(len(t.spans) for t in self._traces)
        errors = sum(1 for t in self._traces if t.status == SpanStatus.ERROR)
        avg_ms = sum(t.elapsed_ms for t in self._traces) / len(self._traces)

        return {
            "traces": len(self._traces),
            "total_spans": total_spans,
            "error_traces": errors,
            "avg_elapsed_ms": round(avg_ms, 1),
        }
