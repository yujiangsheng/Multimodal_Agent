"""Agent Graph — DAG-based workflow engine for Pinocchio.

Defines nodes (processing steps), edges (control flow), and an executor
that walks the graph respecting dependencies, conditions, and parallel
fan-out/fan-in semantics.

Usage::

    graph = AgentGraph("research_pipeline")

    # Define nodes
    graph.add_node(GraphNode("search", handler=search_fn))
    graph.add_node(GraphNode("summarize", handler=summarize_fn))
    graph.add_node(GraphNode("format", handler=format_fn))

    # Define edges (with optional condition)
    graph.add_edge(GraphEdge("search", "summarize"))
    graph.add_edge(GraphEdge("summarize", "format", condition=lambda r: r.success))

    # Execute
    executor = GraphExecutor()
    results = executor.run(graph, initial_input={"query": "quantum"})
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    """Result from executing a single graph node."""

    node_id: str = ""
    success: bool = True
    output: Any = None
    error: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the node result to a JSON-friendly dictionary."""
        return {
            "node_id": self.node_id,
            "success": self.success,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class GraphNode:
    """A processing step in the agent graph.

    Parameters
    ----------
    node_id : str
        Unique identifier for the node.
    handler : callable
        Function ``(input: dict) -> Any`` that performs this step.
    description : str
        Human-readable description.
    retry : int
        Number of retries on failure (default 0).
    """

    node_id: str = ""
    handler: Any = None  # Callable[[dict], Any]
    description: str = ""
    retry: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by node_id so nodes can be used in sets."""
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        """Two nodes are equal when they share the same node_id."""
        if isinstance(other, GraphNode):
            return self.node_id == other.node_id
        return NotImplemented


@dataclass
class GraphEdge:
    """A directed edge between two nodes.

    Parameters
    ----------
    source : str
        Source node ID.
    target : str
        Target node ID.
    condition : callable or None
        Optional predicate ``(NodeResult) -> bool``. If provided, the
        edge is only followed when the condition returns True.
    """

    source: str = ""
    target: str = ""
    condition: Any = None  # Optional Callable[[NodeResult], bool]


class AgentGraph:
    """A directed acyclic graph (DAG) of processing steps.

    Nodes represent processing steps and edges define dependencies and
    optional conditional routing.
    """

    def __init__(self, name: str = "default") -> None:
        """Create a new graph with the given *name*."""
        self.name = name
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        """Register a node in the graph. *node_id* must be non-empty."""
        if not node.node_id:
            raise ValueError("Node must have a non-empty node_id")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed edge. Both source and target must exist."""
        if edge.source not in self._nodes:
            raise ValueError(f"Source node '{edge.source}' not in graph")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node '{edge.target}' not in graph")
        self._edges.append(edge)

    @property
    def nodes(self) -> dict[str, GraphNode]:
        """Snapshot of all registered nodes (id → node)."""
        return dict(self._nodes)

    @property
    def edges(self) -> list[GraphEdge]:
        """Snapshot of all registered edges."""
        return list(self._edges)

    def roots(self) -> list[str]:
        """Return nodes with no incoming edges (entry points)."""
        targets = {e.target for e in self._edges}
        return [nid for nid in self._nodes if nid not in targets]

    def successors(self, node_id: str) -> list[GraphEdge]:
        """Return outgoing edges from a node."""
        return [e for e in self._edges if e.source == node_id]

    def predecessors(self, node_id: str) -> list[str]:
        """Return IDs of nodes that have edges into this node."""
        return [e.source for e in self._edges if e.target == node_id]

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order. Raises ValueError on cycles."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for e in self._edges:
            in_degree[e.target] = in_degree.get(e.target, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[str] = []

        while queue:
            # Sort for determinism
            queue.sort()
            node_id = queue.pop(0)
            result.append(node_id)
            for e in self.successors(node_id):
                in_degree[e.target] -= 1
                if in_degree[e.target] == 0:
                    queue.append(e.target)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains a cycle — DAG required")
        return result

    def validate(self) -> list[str]:
        """Validate the graph. Returns list of issues (empty = valid)."""
        issues: list[str] = []
        if not self._nodes:
            issues.append("Graph has no nodes")
        for node in self._nodes.values():
            if node.handler is None:
                issues.append(f"Node '{node.node_id}' has no handler")
        try:
            self.topological_sort()
        except ValueError as e:
            issues.append(str(e))
        return issues

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph structure (nodes and edges) to a dict."""
        return {
            "name": self.name,
            "nodes": [
                {"id": n.node_id, "description": n.description}
                for n in self._nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target,
                 "conditional": e.condition is not None}
                for e in self._edges
            ],
        }


class GraphExecutor:
    """Execute an AgentGraph respecting DAG ordering.

    Nodes at the same topological level (no dependencies between them)
    can be executed in parallel.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Create an executor with up to *max_workers* parallel threads."""
        self._max_workers = max_workers

    def run(
        self,
        graph: AgentGraph,
        initial_input: dict[str, Any] | None = None,
    ) -> dict[str, NodeResult]:
        """Execute all nodes in the graph.

        Returns a dict mapping node_id → NodeResult.
        """
        issues = graph.validate()
        if issues:
            raise ValueError(f"Invalid graph: {'; '.join(issues)}")

        order = graph.topological_sort()
        results: dict[str, NodeResult] = {}
        context = dict(initial_input or {})

        # Group nodes by level for parallel execution
        levels = self._compute_levels(graph, order)

        for level_nodes in levels:
            if len(level_nodes) == 1:
                nid = level_nodes[0]
                result = self._execute_node(graph, nid, context, results)
                results[nid] = result
                if result.success and result.output is not None:
                    context[nid] = result.output
            else:
                # Parallel execution within level
                with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                    futures = {
                        pool.submit(self._execute_node, graph, nid, context, results): nid
                        for nid in level_nodes
                    }
                    for future in as_completed(futures):
                        nid = futures[future]
                        result = future.result()
                        results[nid] = result
                        if result.success and result.output is not None:
                            context[nid] = result.output

        return results

    def _execute_node(
        self,
        graph: AgentGraph,
        node_id: str,
        context: dict[str, Any],
        results: dict[str, NodeResult],
    ) -> NodeResult:
        """Execute a single node, checking edge conditions first."""
        node = graph.nodes[node_id]

        # Check predecessors — skip node if any predecessor failed
        for pred_id in graph.predecessors(node_id):
            pred_result = results.get(pred_id)
            if pred_result and not pred_result.success:
                return NodeResult(
                    node_id=node_id,
                    success=False,
                    error=f"Skipped: predecessor '{pred_id}' failed",
                )

        # Check if all incoming conditional edges are satisfied
        for pred_id in graph.predecessors(node_id):
            pred_result = results.get(pred_id)
            if not pred_result:
                continue
            # Find the edge and check condition
            for edge in graph.successors(pred_id):
                if edge.target == node_id and edge.condition:
                    if not edge.condition(pred_result):
                        return NodeResult(
                            node_id=node_id,
                            success=False,
                            error="Skipped: edge condition not met",
                        )

        # Execute with retry
        last_error = ""
        for attempt in range(node.retry + 1):
            start = time.time()
            try:
                output = node.handler(context)
                elapsed = (time.time() - start) * 1000
                return NodeResult(
                    node_id=node_id, success=True,
                    output=output, elapsed_ms=elapsed,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Node '%s' attempt %d failed: %s",
                    node_id, attempt + 1, last_error,
                )

        return NodeResult(node_id=node_id, success=False, error=last_error)

    @staticmethod
    def _compute_levels(
        graph: AgentGraph, order: list[str],
    ) -> list[list[str]]:
        """Group topologically-sorted nodes into parallel levels."""
        levels: list[list[str]] = []
        node_level: dict[str, int] = {}

        for nid in order:
            preds = graph.predecessors(nid)
            if not preds:
                lv = 0
            else:
                lv = max(node_level.get(p, 0) for p in preds) + 1
            node_level[nid] = lv
            while len(levels) <= lv:
                levels.append([])
            levels[lv].append(nid)

        return levels
