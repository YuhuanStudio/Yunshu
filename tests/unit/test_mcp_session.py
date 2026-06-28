"""Tests for MCP data model, MCPSession, and tool registration.

Tests the Phase 3 additions:
- MCPTool dataclass
- MCPServerConfig dataclass
- MCPSession class (initialize, tools/list, tools/call dispatch)
- register_mcp_tools global registry
"""

import pytest

from yunshu_gateway.routers.mcp import (
    JSONRPCError,
    MCPServerConfig,
    MCPSession,
    MCPTool,
    _extra_tools_registry,
    register_mcp_tools,
)

# ── MCPTool ──


class TestMCPTool:
    def test_to_dict_with_schema(self):
        tool = MCPTool(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        d = tool.to_dict()
        assert d["name"] == "search"
        assert d["description"] == "Search the web"
        assert d["inputSchema"]["type"] == "object"
        assert "query" in d["inputSchema"]["properties"]

    def test_to_dict_without_schema(self):
        tool = MCPTool(name="ping", description="Ping")
        d = tool.to_dict()
        assert d["inputSchema"] == {"type": "object", "properties": {}}

    def test_equality_by_name(self):
        t1 = MCPTool(name="a", description="desc1")
        t2 = MCPTool(name="a", description="desc2")
        # dataclass equality checks all fields
        assert t1.name == t2.name


# ── MCPServerConfig ──


class TestMCPServerConfig:
    def test_defaults(self):
        cfg = MCPServerConfig()
        assert cfg.server_name == "yunshu"
        assert cfg.version == "0.1.0-dev"
        assert cfg.tools == []

    def test_custom_config(self):
        tools = [MCPTool(name="t1", description="d1")]
        cfg = MCPServerConfig(server_name="custom", version="1.0", tools=tools)
        assert cfg.server_name == "custom"
        assert len(cfg.tools) == 1


# ── MCPSession ──


class TestMCPSession:
    def _make_session(self, tools=None):
        cfg = MCPServerConfig(
            tools=tools or [
                MCPTool(
                    name="calculator",
                    description="Do math",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                ),
            ]
        )
        return MCPSession(cfg)

    # --- initialize ---

    @pytest.mark.asyncio
    async def test_initialize(self):
        session = self._make_session()
        assert not session.is_initialized

        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
        })

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "yunshu"
        assert "tools" in result["capabilities"]
        assert session.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self):
        cfg = MCPServerConfig(server_name="my-server", version="2.0.0")
        session = MCPSession(cfg)
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
        })
        assert resp["result"]["serverInfo"]["name"] == "my-server"
        assert resp["result"]["serverInfo"]["version"] == "2.0.0"

    # --- tools/list ---

    @pytest.mark.asyncio
    async def test_tools_list(self):
        session = self._make_session()
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2,
        })

        tools = resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert tools[0]["description"] == "Do math"
        assert "inputSchema" in tools[0]

    @pytest.mark.asyncio
    async def test_tools_list_empty(self):
        session = MCPSession(MCPServerConfig(tools=[]))
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 3,
        })
        assert resp["result"]["tools"] == []

    # --- tools/call ---

    @pytest.mark.asyncio
    async def test_tools_call_known_tool(self):
        # Tool WITH a handler → real execution (isError False + the actual result).
        async def _calc(args):
            return str(eval(args["expression"], {"__builtins__": {}}, {}))  # "2+2" → "4"
        session = self._make_session(tools=[MCPTool(
            name="calculator", description="Do math",
            input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
            handler=_calc)])
        resp = await session.handle_message({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "2+2"}},
            "id": 4,
        })
        assert resp["jsonrpc"] == "2.0"
        result = resp["result"]
        assert result["isError"] is False
        assert "4" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_handler_less_tool_is_error(self):
        # /653: a registered tool with NO handler can't execute → isError
        # True (not a misleading "acknowledged" success).
        session = self._make_session()  # default calculator has no handler
        resp = await session.handle_message({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "2+2"}},
            "id": 5,
        })
        result = resp["result"]
        assert result["isError"] is True
        assert "calculator" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_unknown_tool(self):
        session = self._make_session()
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
            "id": 5,
        })
        assert "error" in resp
        assert resp["error"]["code"] == JSONRPCError.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tools_call_missing_params(self):
        session = self._make_session()
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 6,
        })
        assert "error" in resp
        assert resp["error"]["code"] == JSONRPCError.INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_tools_call_null_params(self):
        session = self._make_session()
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": None,
            "id": 7,
        })
        assert "error" in resp

    # --- method not found ---

    @pytest.mark.asyncio
    async def test_unknown_method(self):
        session = self._make_session()
        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "nonexistent/method",
            "id": 8,
        })
        assert "error" in resp
        assert resp["error"]["code"] == JSONRPCError.METHOD_NOT_FOUND

    # --- tool registration ---

    @pytest.mark.asyncio
    async def test_register_tool(self):
        session = MCPSession(MCPServerConfig(tools=[]))
        assert len(session.tools) == 0

        session.register_tool(MCPTool(name="new_tool", description="A new tool"))
        assert len(session.tools) == 1

        resp = await session.handle_message({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 10,
        })
        assert resp["result"]["tools"][0]["name"] == "new_tool"

    def test_unregister_tool(self):
        session = self._make_session()
        assert session.unregister_tool("calculator")
        assert not session.unregister_tool("nonexistent")

    def test_register_tool_overwrites(self):
        session = MCPSession(MCPServerConfig(tools=[]))
        session.register_tool(MCPTool(name="t", description="v1"))
        session.register_tool(MCPTool(name="t", description="v2"))
        assert len(session.tools) == 1
        assert session.tools[0].description == "v2"


# ── register_mcp_tools (global registry) ──


class TestRegisterMCPTools:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Clear extra tools registry before each test."""
        saved = dict(_extra_tools_registry)
        _extra_tools_registry.clear()
        yield
        _extra_tools_registry.clear()
        _extra_tools_registry.update(saved)

    def test_register_single_tool(self):
        register_mcp_tools([
            MCPTool(
                name="weather",
                description="Get weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ])
        assert "weather" in _extra_tools_registry

    def test_register_multiple_tools(self):
        register_mcp_tools([
            MCPTool(name="t1", description="Tool 1"),
            MCPTool(name="t2", description="Tool 2"),
        ])
        assert "t1" in _extra_tools_registry
        assert "t2" in _extra_tools_registry

    @pytest.mark.asyncio
    async def test_registered_tool_appears_in_list(self):
        register_mcp_tools([
            MCPTool(name="search", description="Search the web"),
        ])

        from yunshu_gateway.routers.mcp import _handle_tools_list

        result = await _handle_tools_list(None, 1)
        tool_names = [t["name"] for t in result["result"]["tools"]]
        assert "search" in tool_names
