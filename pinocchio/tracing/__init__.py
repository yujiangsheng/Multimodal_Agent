"""Structured tracing and observability for Pinocchio.

Provides Span/Trace classes for timing, structured event logging,
and OpenTelemetry-compatible output.
"""

from pinocchio.tracing.tracer import (
    Span,
    Trace,
    Tracer,
    SpanStatus,
)

__all__ = ["Span", "Trace", "Tracer", "SpanStatus"]
