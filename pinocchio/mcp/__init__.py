"""MCP (Model Context Protocol) support for Pinocchio.

Allows the agent to discover and call tools hosted on external MCP servers,
bridging them into the Pinocchio ToolRegistry transparently.
"""

from pinocchio.mcp.mcp_client import MCPClient, MCPToolBridge

__all__ = ["MCPClient", "MCPToolBridge"]
