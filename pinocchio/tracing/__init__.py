"""Structured tracing and observability for Pinocchio.

Provides Span/Trace classes for timing, structured event logging,
and OpenTelemetry-compatible output.

Key concepts:

* **Tracer** — global entry point; manages the lifecycle of multiple traces.
* **Trace** — an end-to-end interaction (e.g. a single ``chat()`` call).
* **Span** — a timed operation within a trace (e.g. ``perceive``, ``execute``).
* **SpanStatus** — OK / ERROR / SKIPPED.

Quick start::

    from pinocchio.tracing import Tracer

    tracer = Tracer()
    with tracer.start_trace("chat_request") as trace:
        with trace.span("perceive") as s:
            s.set_attribute("modality", "text")
            ...
        with trace.span("execute") as s:
            s.add_event("tool_call", {"tool": "calculator"})
            ...
    print(tracer.export_summary())
"""

from pinocchio.tracing.tracer import (
    Span,
    Trace,
    Tracer,
    SpanStatus,
)

__all__ = ["Span", "Trace", "Tracer", "SpanStatus"]
