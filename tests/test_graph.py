"""Tests for Gap 5: Agent Graph (DAG workflow engine)."""

from __future__ import annotations

import pytest

from pinocchio.graph.agent_graph import (
    AgentGraph,
    GraphEdge,
    GraphExecutor,
    GraphNode,
    NodeResult,
)


class TestNodeResult:
    def test_defaults(self):
        r = NodeResult()
        assert r.success is True
        assert r.output is None

    def test_to_dict(self):
        r = NodeResult(node_id="n1", success=True, output="data", elapsed_ms=12.3)
        d = r.to_dict()
        assert d["node_id"] == "n1"
        assert d["success"] is True
        assert d["elapsed_ms"] == 12.3


class TestGraphNode:
    def test_hash_and_eq(self):
        a = GraphNode(node_id="a", handler=lambda x: x)
        b = GraphNode(node_id="a", handler=lambda x: x)
        assert a == b
        assert hash(a) == hash(b)

    def test_not_eq_different_id(self):
        a = GraphNode(node_id="a")
        b = GraphNode(node_id="b")
        assert a != b


class TestAgentGraph:
    def test_add_node(self):
        g = AgentGraph("test")
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        assert "a" in g.nodes

    def test_add_node_empty_id_raises(self):
        g = AgentGraph()
        with pytest.raises(ValueError, match="non-empty"):
            g.add_node(GraphNode(node_id=""))

    def test_add_edge(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        g.add_edge(GraphEdge(source="a", target="b"))
        assert len(g.edges) == 1

    def test_add_edge_invalid_source(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        with pytest.raises(ValueError, match="Source"):
            g.add_edge(GraphEdge(source="missing", target="b"))

    def test_add_edge_invalid_target(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        with pytest.raises(ValueError, match="Target"):
            g.add_edge(GraphEdge(source="a", target="missing"))

    def test_roots(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        g.add_node(GraphNode(node_id="c", handler=lambda x: x))
        g.add_edge(GraphEdge(source="a", target="b"))
        g.add_edge(GraphEdge(source="a", target="c"))
        roots = g.roots()
        assert roots == ["a"]

    def test_topological_sort(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        g.add_node(GraphNode(node_id="c", handler=lambda x: x))
        g.add_edge(GraphEdge(source="a", target="b"))
        g.add_edge(GraphEdge(source="b", target="c"))
        order = g.topological_sort()
        assert order == ["a", "b", "c"]

    def test_cycle_detection(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        g.add_edge(GraphEdge(source="a", target="b"))
        g.add_edge(GraphEdge(source="b", target="a"))
        with pytest.raises(ValueError, match="cycle"):
            g.topological_sort()

    def test_validate_empty(self):
        g = AgentGraph()
        issues = g.validate()
        assert any("no nodes" in i for i in issues)

    def test_validate_no_handler(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=None))
        issues = g.validate()
        assert any("no handler" in i for i in issues)

    def test_validate_valid(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        issues = g.validate()
        assert issues == []

    def test_successors(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        g.add_node(GraphNode(node_id="c", handler=lambda x: x))
        g.add_edge(GraphEdge(source="a", target="b"))
        g.add_edge(GraphEdge(source="a", target="c"))
        succs = g.successors("a")
        assert len(succs) == 2

    def test_predecessors(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda x: x))
        g.add_node(GraphNode(node_id="b", handler=lambda x: x))
        g.add_edge(GraphEdge(source="a", target="b"))
        preds = g.predecessors("b")
        assert preds == ["a"]

    def test_to_dict(self):
        g = AgentGraph("test_graph")
        g.add_node(GraphNode(node_id="a", handler=lambda x: x, description="first"))
        d = g.to_dict()
        assert d["name"] == "test_graph"
        assert len(d["nodes"]) == 1


class TestGraphExecutor:
    def test_simple_linear_graph(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="a", handler=lambda ctx: "result_a"))
        g.add_node(GraphNode(node_id="b", handler=lambda ctx: f"got_{ctx.get('a', '')}"))
        g.add_edge(GraphEdge(source="a", target="b"))

        executor = GraphExecutor()
        results = executor.run(g)
        assert results["a"].success is True
        assert results["a"].output == "result_a"
        assert results["b"].success is True
        assert results["b"].output == "got_result_a"

    def test_parallel_nodes(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="root", handler=lambda ctx: "start"))
        g.add_node(GraphNode(node_id="a", handler=lambda ctx: "branch_a"))
        g.add_node(GraphNode(node_id="b", handler=lambda ctx: "branch_b"))
        g.add_node(GraphNode(node_id="merge", handler=lambda ctx: f"{ctx.get('a', '')},{ctx.get('b', '')}"))
        g.add_edge(GraphEdge(source="root", target="a"))
        g.add_edge(GraphEdge(source="root", target="b"))
        g.add_edge(GraphEdge(source="a", target="merge"))
        g.add_edge(GraphEdge(source="b", target="merge"))

        executor = GraphExecutor(max_workers=2)
        results = executor.run(g)
        assert results["merge"].success is True
        assert "branch_a" in results["merge"].output
        assert "branch_b" in results["merge"].output

    def test_conditional_edge(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="check", handler=lambda ctx: "pass"))
        g.add_node(GraphNode(node_id="success", handler=lambda ctx: "ok"))
        g.add_node(GraphNode(node_id="fail", handler=lambda ctx: "nope"))

        g.add_edge(GraphEdge(
            source="check", target="success",
            condition=lambda r: r.success and r.output == "pass",
        ))
        g.add_edge(GraphEdge(
            source="check", target="fail",
            condition=lambda r: not r.success,
        ))

        executor = GraphExecutor()
        results = executor.run(g)
        assert results["success"].success is True
        assert results["fail"].success is False  # condition not met → skipped

    def test_node_failure(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="fail", handler=lambda ctx: 1 / 0))
        executor = GraphExecutor()
        results = executor.run(g)
        assert results["fail"].success is False
        assert "division" in results["fail"].error.lower()

    def test_node_retry(self):
        call_count = {"n": 0}

        def flaky(ctx):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("temporary failure")
            return "ok"

        g = AgentGraph()
        g.add_node(GraphNode(node_id="flaky", handler=flaky, retry=2))
        executor = GraphExecutor()
        results = executor.run(g)
        assert results["flaky"].success is True
        assert results["flaky"].output == "ok"

    def test_invalid_graph_raises(self):
        g = AgentGraph()  # empty
        executor = GraphExecutor()
        with pytest.raises(ValueError, match="Invalid"):
            executor.run(g)

    def test_single_node(self):
        g = AgentGraph()
        g.add_node(GraphNode(node_id="solo", handler=lambda ctx: 42))
        executor = GraphExecutor()
        results = executor.run(g)
        assert results["solo"].output == 42

    def test_initial_input_passed(self):
        g = AgentGraph()
        g.add_node(GraphNode(
            node_id="reader",
            handler=lambda ctx: ctx.get("query", "none"),
        ))
        executor = GraphExecutor()
        results = executor.run(g, initial_input={"query": "hello"})
        assert results["reader"].output == "hello"
