"""MCP Client — connect to Model Context Protocol servers.

Implements a lightweight MCP client that:
1. Discovers tools from one or more MCP-compatible servers.
2. Wraps each remote tool as a Pinocchio ``Tool`` and registers it.
3. Proxies tool calls over HTTP/JSON-RPC.

The MCP specification is a JSON-RPC 2.0 protocol over HTTP (or stdio).
We support the HTTP transport here.

Usage::

    bridge = MCPToolBridge(tool_registry)
    bridge.connect("http://localhost:8080/mcp")
    # All tools from the MCP server are now in the registry and callable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MCPToolSpec:
    """Specification of a tool advertised by an MCP server."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    server_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the tool specification to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "server_url": self.server_url,
        }


class MCPClient:
    """Low-level JSON-RPC 2.0 client for MCP servers."""

    def __init__(self, server_url: str, *, timeout: float = 30.0) -> None:
        """Initialise the MCP client.

        Args:
            server_url: Base URL of the MCP server (e.g. ``http://localhost:8080/mcp``).
            timeout: HTTP request timeout in seconds.
        """
        self._url = server_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)
        self._request_id = 0

    def _next_id(self) -> int:
        """Generate a monotonically increasing JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    def _rpc_call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC 2.0 request and return the result."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        resp = self._http.post(self._url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"MCP error {err.get('code', '?')}: {err.get('message', '')}")
        return body.get("result")

    # -- MCP protocol methods --

    def list_tools(self) -> list[MCPToolSpec]:
        """Discover available tools from the MCP server."""
        result = self._rpc_call("tools/list")
        tools: list[MCPToolSpec] = []
        for item in (result or {}).get("tools", []):
            tools.append(MCPToolSpec(
                name=item.get("name", "unknown"),
                description=item.get("description", ""),
                parameters=item.get("inputSchema", {}),
                server_url=self._url,
            ))
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke a tool on the MCP server."""
        result = self._rpc_call("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        # MCP returns content array; extract text
        if isinstance(result, dict):
            content = result.get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            if texts:
                return "\n".join(texts)
            return json.dumps(result, ensure_ascii=False)
        return str(result) if result else ""

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._http.close()


class MCPToolBridge:
    """Bridge MCP servers into the Pinocchio ToolRegistry.

    Discovers tools from MCP servers and registers them as callable
    Pinocchio tools with an ``mcp_`` prefix.
    """

    def __init__(self, tool_registry: Any) -> None:
        """Initialise the bridge.

        Args:
            tool_registry: A Pinocchio :class:`ToolRegistry` instance
                where discovered MCP tools will be registered.
        """
        self._registry = tool_registry
        self._clients: dict[str, MCPClient] = {}
        self._tool_specs: list[MCPToolSpec] = []

    def connect(self, server_url: str, *, prefix: str = "mcp_") -> list[str]:
        """Connect to an MCP server and register its tools.

        Returns the list of registered tool names.
        """
        client = MCPClient(server_url)
        self._clients[server_url] = client

        registered: list[str] = []
        try:
            specs = client.list_tools()
        except Exception as exc:
            logger.warning("Failed to list tools from %s: %s", server_url, exc)
            return registered

        for spec in specs:
            self._tool_specs.append(spec)
            tool_name = f"{prefix}{spec.name}"

            # Build a closure that calls this specific MCP tool
            def _make_caller(
                s_url: str, t_name: str,
            ) -> Any:
                def _call(**kwargs: Any) -> str:
                    c = self._clients.get(s_url)
                    if not c:
                        return f"Error: MCP server {s_url} not connected"
                    try:
                        return c.call_tool(t_name, kwargs)
                    except Exception as e:
                        return f"MCP tool error: {e}"
                _call.__name__ = tool_name
                _call.__doc__ = spec.description or f"MCP tool: {t_name}"
                return _call

            caller = _make_caller(server_url, spec.name)

            from pinocchio.tools import Tool
            t = Tool(
                name=tool_name,
                description=spec.description or f"Remote MCP tool: {spec.name}",
                parameters=spec.parameters.get("properties", {}),
                function=caller,
            )
            self._registry.register(t)
            registered.append(tool_name)

        logger.info("Registered %d MCP tools from %s", len(registered), server_url)
        return registered

    def disconnect(self, server_url: str) -> None:
        """Disconnect from an MCP server."""
        client = self._clients.pop(server_url, None)
        if client:
            client.close()

    def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    @property
    def connected_servers(self) -> list[str]:
        """List of server URLs currently connected."""
        return list(self._clients.keys())

    @property
    def tool_specs(self) -> list[MCPToolSpec]:
        """All tool specifications discovered from connected servers."""
        return list(self._tool_specs)
