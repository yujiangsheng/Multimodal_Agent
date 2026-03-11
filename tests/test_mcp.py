"""Tests for Gap 4: MCP protocol support (MCPClient + MCPToolBridge)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.mcp.mcp_client import MCPClient, MCPToolBridge, MCPToolSpec


class TestMCPToolSpec:
    def test_defaults(self):
        spec = MCPToolSpec(name="test_tool")
        assert spec.name == "test_tool"
        assert spec.description == ""
        assert spec.server_url == ""

    def test_to_dict(self):
        spec = MCPToolSpec(
            name="fetch",
            description="Fetch data",
            server_url="http://localhost:8080",
        )
        d = spec.to_dict()
        assert d["name"] == "fetch"
        assert d["description"] == "Fetch data"


class TestMCPClient:
    def test_init(self):
        client = MCPClient("http://localhost:8080/mcp")
        assert client._url == "http://localhost:8080/mcp"

    def test_rpc_call_builds_correct_payload(self):
        client = MCPClient("http://localhost:8080")
        with patch.object(client._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            result = client._rpc_call("test/method", {"key": "value"})
            assert result == {"ok": True}

            call_args = mock_post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            assert payload["method"] == "test/method"
            assert payload["jsonrpc"] == "2.0"

    def test_rpc_call_error(self):
        client = MCPClient("http://localhost:8080")
        with patch.object(client._http, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "jsonrpc": "2.0", "id": 1,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="MCP error"):
                client._rpc_call("bad/method")

    def test_list_tools(self):
        client = MCPClient("http://localhost:8080")
        with patch.object(client, "_rpc_call") as mock_rpc:
            mock_rpc.return_value = {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "inputSchema": {"properties": {"query": {"type": "string"}}},
                    }
                ]
            }
            tools = client.list_tools()
            assert len(tools) == 1
            assert tools[0].name == "search"
            assert tools[0].description == "Search the web"

    def test_call_tool(self):
        client = MCPClient("http://localhost:8080")
        with patch.object(client, "_rpc_call") as mock_rpc:
            mock_rpc.return_value = {
                "content": [{"type": "text", "text": "Search results here"}]
            }
            result = client.call_tool("search", {"query": "test"})
            assert result == "Search results here"

    def test_call_tool_non_text(self):
        client = MCPClient("http://localhost:8080")
        with patch.object(client, "_rpc_call") as mock_rpc:
            mock_rpc.return_value = {"data": "binary"}
            result = client.call_tool("get_data")
            assert "data" in result

    def test_close(self):
        client = MCPClient("http://localhost:8080")
        with patch.object(client._http, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()


class TestMCPToolBridge:
    def test_init(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)
        assert bridge.connected_servers == []

    def test_connect_registers_tools(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)

        with patch.object(MCPClient, "list_tools") as mock_list:
            mock_list.return_value = [
                MCPToolSpec(name="tool_a", description="Does A"),
                MCPToolSpec(name="tool_b", description="Does B"),
            ]
            registered = bridge.connect("http://localhost:8080")

        assert len(registered) == 2
        assert "mcp_tool_a" in registered
        assert "mcp_tool_b" in registered
        assert registry.register.call_count == 2

    def test_connect_failure(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)

        with patch.object(MCPClient, "list_tools", side_effect=Exception("connection refused")):
            registered = bridge.connect("http://dead:9999")

        assert registered == []

    def test_disconnect(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)

        with patch.object(MCPClient, "list_tools", return_value=[]):
            bridge.connect("http://localhost:8080")

        assert "http://localhost:8080" in bridge.connected_servers

        with patch.object(MCPClient, "close"):
            bridge.disconnect("http://localhost:8080")

        assert bridge.connected_servers == []

    def test_disconnect_all(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)

        with patch.object(MCPClient, "list_tools", return_value=[]):
            bridge.connect("http://server1:8080")
            bridge.connect("http://server2:8080")

        assert len(bridge.connected_servers) == 2

        with patch.object(MCPClient, "close"):
            bridge.disconnect_all()

        assert bridge.connected_servers == []

    def test_tool_caller_invokes_mcp(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)

        with patch.object(MCPClient, "list_tools") as mock_list:
            mock_list.return_value = [
                MCPToolSpec(name="echo", description="Echo back"),
            ]
            bridge.connect("http://localhost:8080")

        # Get the registered tool's function
        call_args = registry.register.call_args
        registered_tool = call_args[0][0]
        assert registered_tool.name == "mcp_echo"

    def test_tool_specs_property(self):
        registry = MagicMock()
        bridge = MCPToolBridge(registry)

        with patch.object(MCPClient, "list_tools") as mock_list:
            mock_list.return_value = [
                MCPToolSpec(name="t1", description="Tool 1"),
            ]
            bridge.connect("http://localhost:8080")

        assert len(bridge.tool_specs) == 1
        assert bridge.tool_specs[0].name == "t1"
