"""Agent Graph 工作流引擎示例。

演示 DAG 定义、条件路由和并行执行。
# 离线可运行
"""

from __future__ import annotations

import json

from pinocchio.graph import AgentGraph, GraphNode, GraphEdge, GraphExecutor


def linear_pipeline():
    """线性三步管道。"""
    graph = AgentGraph("linear_pipeline")

    # 定义节点（handler 接收 context dict，返回任意结果）
    graph.add_node(GraphNode(
        node_id="fetch",
        handler=lambda ctx: {"data": "raw data from API", "count": 42},
        description="Fetch data from source",
    ))
    graph.add_node(GraphNode(
        node_id="transform",
        handler=lambda ctx: f"Processed {ctx.get('fetch', {}).get('count', 0)} records",
        description="Transform and clean data",
    ))
    graph.add_node(GraphNode(
        node_id="output",
        handler=lambda ctx: f"Report: {ctx.get('transform', '')}",
        description="Generate final report",
    ))

    # 定义边
    graph.add_edge(GraphEdge("fetch", "transform"))
    graph.add_edge(GraphEdge("transform", "output"))

    # 执行
    executor = GraphExecutor()
    results = executor.run(graph, initial_input={"query": "test"})

    print("=== 线性管道 ===")
    for node_id, result in results.items():
        print(f"  {node_id}: success={result.success}, output={result.output}")
    print()


def parallel_fanout():
    """并行扇出/汇聚模式。"""
    graph = AgentGraph("parallel_fanout")

    # 一个源节点，两个并行处理节点，一个汇聚节点
    graph.add_node(GraphNode(
        node_id="source",
        handler=lambda ctx: "shared data",
    ))
    graph.add_node(GraphNode(
        node_id="branch_a",
        handler=lambda ctx: f"A processed: {ctx.get('source', '')}",
    ))
    graph.add_node(GraphNode(
        node_id="branch_b",
        handler=lambda ctx: f"B processed: {ctx.get('source', '')}",
    ))
    graph.add_node(GraphNode(
        node_id="merge",
        handler=lambda ctx: f"Merged: [{ctx.get('branch_a', '')}, {ctx.get('branch_b', '')}]",
    ))

    graph.add_edge(GraphEdge("source", "branch_a"))
    graph.add_edge(GraphEdge("source", "branch_b"))
    graph.add_edge(GraphEdge("branch_a", "merge"))
    graph.add_edge(GraphEdge("branch_b", "merge"))

    results = GraphExecutor(max_workers=2).run(graph)

    print("=== 并行扇出/汇聚 ===")
    for nid, r in results.items():
        print(f"  {nid}: {r.output}")
    print()


def conditional_routing():
    """条件路由 — 根据前置节点结果决定是否执行。"""
    graph = AgentGraph("conditional")

    graph.add_node(GraphNode(
        node_id="check",
        handler=lambda ctx: {"valid": True, "score": 85},
    ))
    graph.add_node(GraphNode(
        node_id="process",
        handler=lambda ctx: "Processed successfully",
    ))
    graph.add_node(GraphNode(
        node_id="error_handler",
        handler=lambda ctx: "Handling error...",
    ))

    # 只有当 check 成功时才执行 process
    graph.add_edge(GraphEdge("check", "process",
                             condition=lambda r: r.success and r.output.get("valid")))
    # 只有当 check 失败时才执行 error_handler
    graph.add_edge(GraphEdge("check", "error_handler",
                             condition=lambda r: not r.success))

    results = GraphExecutor().run(graph)

    print("=== 条件路由 ===")
    for nid, r in results.items():
        status = "✓ executed" if r.success and r.error == "" else f"✗ skipped ({r.error})"
        print(f"  {nid}: {status}")
    print()


def graph_validation():
    """图验证 — 检测问题。"""
    graph = AgentGraph("validate_test")

    # 没有 handler 的节点
    graph.add_node(GraphNode(node_id="incomplete"))

    issues = graph.validate()

    print("=== 图验证 ===")
    for issue in issues:
        print(f"  ⚠ {issue}")

    # 序列化
    print(f"\n  Graph JSON: {json.dumps(graph.to_dict(), indent=2)}")
    print()


if __name__ == "__main__":
    linear_pipeline()
    parallel_fanout()
    conditional_routing()
    graph_validation()
