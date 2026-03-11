"""Pinocchio — a multimodal self-evolving agent.

Top-level package.  The primary public API is the :class:`Pinocchio`
orchestrator, which drives the 6-phase cognitive loop::

    from pinocchio import Pinocchio

    agent = Pinocchio()
    print(agent.chat("Hello, Pinocchio!"))

For multimodal input, pass file paths via keyword arguments::

    agent.chat("Describe this image", image_paths=["photo.jpg"])
    agent.chat("Summarise this recording", audio_paths=["meeting.wav"])

All internal sub-modules (agents, memory, models, multimodal, utils)
are implementation details and should not normally be imported directly.
"""

from pinocchio.orchestrator import Pinocchio
from pinocchio.tools import Tool, ToolRegistry, ToolExecutor, tool

# New subsystems (Gaps 1–7)
from pinocchio.planning.planner import TaskPlanner, TaskPlan, TaskStep
from pinocchio.planning.react import ReActExecutor
from pinocchio.sandbox.code_sandbox import CodeSandbox
from pinocchio.rag.document_store import DocumentStore, DocumentChunk
from pinocchio.mcp.mcp_client import MCPClient, MCPToolBridge
from pinocchio.graph.agent_graph import AgentGraph, GraphNode, GraphEdge, GraphExecutor
from pinocchio.collaboration.team import AgentTeam, TeamMember
from pinocchio.tracing.tracer import Tracer, Trace, Span

__all__ = [
    "Pinocchio", "Tool", "ToolRegistry", "ToolExecutor", "tool",
    # Planning
    "TaskPlanner", "TaskPlan", "TaskStep", "ReActExecutor",
    # Sandbox
    "CodeSandbox",
    # RAG
    "DocumentStore", "DocumentChunk",
    # MCP
    "MCPClient", "MCPToolBridge",
    # Graph
    "AgentGraph", "GraphNode", "GraphEdge", "GraphExecutor",
    # Collaboration
    "AgentTeam", "TeamMember",
    # Tracing
    "Tracer", "Trace", "Span",
]
