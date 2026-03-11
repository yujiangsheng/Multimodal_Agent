"""Tests for Gap 7: Structured tracing (Tracer, Trace, Span)."""

from __future__ import annotations

import json
import time

import pytest

from pinocchio.tracing.tracer import Span, SpanStatus, Trace, Tracer


class TestSpanStatus:
    def test_values(self):
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"
        assert SpanStatus.SKIPPED.value == "skipped"


class TestSpan:
    def test_auto_id(self):
        s = Span(name="test")
        assert s.span_id != ""
        assert len(s.span_id) == 12

    def test_set_attribute(self):
        s = Span(name="test")
        s.set_attribute("key", "value")
        assert s.attributes["key"] == "value"

    def test_add_event(self):
        s = Span(name="test")
        s.add_event("tool_call", {"tool": "calculator"})
        assert len(s.events) == 1
        assert s.events[0].name == "tool_call"
        assert s.events[0].attributes["tool"] == "calculator"

    def test_set_status(self):
        s = Span(name="test")
        s.set_status(SpanStatus.ERROR, "something broke")
        assert s.status == SpanStatus.ERROR
        assert s.attributes["error.message"] == "something broke"

    def test_context_manager(self):
        with Span(name="timed") as s:
            time.sleep(0.01)
        assert s.end_time > s.start_time
        assert s.elapsed_ms > 0

    def test_context_manager_on_error(self):
        with pytest.raises(ValueError):
            with Span(name="error_span") as s:
                raise ValueError("test error")
        assert s.status == SpanStatus.ERROR

    def test_elapsed_ms_before_end(self):
        s = Span(name="running")
        time.sleep(0.01)
        assert s.elapsed_ms > 0  # computed from current time

    def test_to_dict(self):
        s = Span(name="test", trace_id="t1")
        s.set_attribute("x", 1)
        s.end()
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["trace_id"] == "t1"
        assert d["status"] == "ok"
        assert "elapsed_ms" in d

    def test_end_idempotent(self):
        s = Span(name="test")
        s.end()
        t1 = s.end_time
        s.end()  # should not change
        assert s.end_time == t1


class TestTrace:
    def test_create(self):
        t = Trace(name="test_trace")
        assert t.trace_id != ""
        assert t.name == "test_trace"
        assert t.spans == []

    def test_span_context_manager(self):
        t = Trace(name="test")
        with t.span("phase1") as s:
            s.set_attribute("key", "value")
        assert len(t.spans) == 1
        assert t.spans[0].name == "phase1"
        assert t.spans[0].trace_id == t.trace_id

    def test_nested_spans(self):
        t = Trace(name="nested")
        with t.span("outer") as s1:
            with t.span("inner") as s2:
                pass
        assert len(t.spans) == 2
        assert t.spans[1].parent_id == t.spans[0].span_id

    def test_status_on_error(self):
        t = Trace(name="error_test")
        with pytest.raises(ValueError):
            with t.span("bad") as s:
                raise ValueError("boom")
        assert t.status == SpanStatus.ERROR

    def test_context_manager(self):
        with Trace(name="ctx") as t:
            with t.span("s1"):
                pass
        assert t.elapsed_ms > 0

    def test_add_span(self):
        t = Trace(name="manual")
        s = Span(name="added")
        t.add_span(s)
        assert len(t.spans) == 1
        assert t.spans[0].trace_id == t.trace_id

    def test_to_dict(self):
        with Trace(name="test") as t:
            with t.span("s1"):
                pass
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["span_count"] == 1
        assert len(d["spans"]) == 1

    def test_summary(self):
        with Trace(name="summary_test") as t:
            with t.span("phase"):
                pass
        s = t.summary()
        assert "summary_test" in s
        assert "1 spans" in s


class TestTracer:
    def test_start_trace(self):
        tracer = Tracer()
        with tracer.start_trace("test") as t:
            with t.span("s1"):
                pass
        assert tracer.trace_count == 1

    def test_create_trace(self):
        tracer = Tracer()
        t = tracer.create_trace("manual")
        assert tracer.trace_count == 1
        assert t.name == "manual"

    def test_latest(self):
        tracer = Tracer()
        assert tracer.latest() is None
        with tracer.start_trace("first"):
            pass
        with tracer.start_trace("second"):
            pass
        assert tracer.latest().name == "second"

    def test_max_traces(self):
        tracer = Tracer(max_traces=3)
        for i in range(5):
            with tracer.start_trace(f"trace_{i}"):
                pass
        assert tracer.trace_count == 3

    def test_clear(self):
        tracer = Tracer()
        with tracer.start_trace("t"):
            pass
        tracer.clear()
        assert tracer.trace_count == 0

    def test_export_json(self):
        tracer = Tracer()
        with tracer.start_trace("export_test") as t:
            with t.span("s1") as s:
                s.set_attribute("key", "value")

        output = tracer.export_json()
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["name"] == "export_test"

    def test_export_json_pretty(self):
        tracer = Tracer()
        with tracer.start_trace("pretty"):
            pass
        output = tracer.export_json(pretty=True)
        assert "\n" in output

    def test_export_summary(self):
        tracer = Tracer()
        with tracer.start_trace("sum_test") as t:
            with t.span("perceive"):
                pass
            with t.span("execute"):
                pass
        summary = tracer.export_summary()
        assert "sum_test" in summary
        assert "perceive" in summary
        assert "execute" in summary

    def test_stats_empty(self):
        tracer = Tracer()
        s = tracer.stats()
        assert s["traces"] == 0

    def test_stats_with_traces(self):
        tracer = Tracer()
        with tracer.start_trace("t1") as t:
            with t.span("s1"):
                pass
        with tracer.start_trace("t2") as t:
            with t.span("s2"):
                pass
        s = tracer.stats()
        assert s["traces"] == 2
        assert s["total_spans"] == 2
        assert s["error_traces"] == 0
        assert "avg_elapsed_ms" in s

    def test_stats_with_errors(self):
        tracer = Tracer()
        try:
            with tracer.start_trace("error_trace") as t:
                with t.span("bad"):
                    raise RuntimeError("fail")
        except RuntimeError:
            pass
        s = tracer.stats()
        assert s["error_traces"] == 1

    def test_traces_property(self):
        tracer = Tracer()
        with tracer.start_trace("t1"):
            pass
        traces = tracer.traces
        assert len(traces) == 1
        # Should be a copy
        traces.clear()
        assert tracer.trace_count == 1
