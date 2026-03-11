"""结构化追踪与可观测性示例。

演示 Tracer/Trace/Span 的使用方式。
# 离线可运行
"""

from __future__ import annotations

import time

from pinocchio.tracing import Tracer, SpanStatus


def basic_tracing():
    """基本追踪 — 使用上下文管理器。"""
    tracer = Tracer()

    with tracer.start_trace("chat_request") as trace:
        with trace.span("perceive") as s:
            s.set_attribute("modality", "text")
            s.set_attribute("input_length", 42)
            time.sleep(0.01)  # 模拟处理

        with trace.span("strategize") as s:
            s.set_attribute("strategy", "direct_answer")
            time.sleep(0.005)

        with trace.span("execute") as s:
            s.add_event("llm_call", {"model": "qwen3-vl:4b", "tokens": 256})
            time.sleep(0.02)

        with trace.span("evaluate") as s:
            s.set_attribute("quality_score", 8)
            s.set_attribute("complete", True)

    print("=== 基本追踪 ===")
    print(tracer.export_summary())
    print()


def nested_spans():
    """嵌套 Span — 父子关系。"""
    tracer = Tracer()

    with tracer.start_trace("complex_task") as trace:
        with trace.span("execute") as parent:
            parent.set_attribute("strategy", "multi_step")

            with trace.span("step_1_search") as child:
                child.set_attribute("tool", "web_fetch")
                child.add_event("http_request", {"url": "example.com"})
                time.sleep(0.01)

            with trace.span("step_2_process") as child:
                child.set_attribute("tool", "python_eval")
                time.sleep(0.005)

    print("=== 嵌套 Span ===")
    for s in trace.spans:
        indent = "    " if s.parent_id else "  "
        print(f"{indent}{s.name} ({s.elapsed_ms:.1f}ms) parent={s.parent_id or 'root'}")
    print()


def error_handling():
    """错误追踪 — 异常自动记录。"""
    tracer = Tracer()

    try:
        with tracer.start_trace("failing_task") as trace:
            with trace.span("risky_operation") as s:
                s.set_attribute("attempt", 1)
                raise ValueError("Something went wrong")
    except ValueError:
        pass

    print("=== 错误追踪 ===")
    print(f"Trace status: {trace.status.value}")
    for s in trace.spans:
        print(f"  {s.name}: status={s.status.value}, error={s.attributes.get('error.message', '')}")
    print()


def statistics():
    """聚合统计。"""
    tracer = Tracer()

    # 模拟多次交互
    for i in range(5):
        with tracer.start_trace(f"interaction_{i}") as trace:
            with trace.span("process") as s:
                time.sleep(0.005)
                if i == 3:
                    s.set_status(SpanStatus.ERROR, "timeout")

    print("=== 聚合统计 ===")
    import json
    print(json.dumps(tracer.stats(), indent=2))
    print(f"Total traces: {tracer.trace_count}")
    print(f"Latest: {tracer.latest().summary()}")
    print()


def json_export():
    """JSON 导出。"""
    tracer = Tracer()

    with tracer.start_trace("export_demo") as trace:
        with trace.span("step") as s:
            s.set_attribute("key", "value")

    print("=== JSON 导出（前 300 字符）===")
    print(tracer.export_json(pretty=True)[:300])
    print()


if __name__ == "__main__":
    basic_tracing()
    nested_spans()
    error_handling()
    statistics()
    json_export()
