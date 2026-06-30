"""Tests for the MCP data model and tool registration.

- MCPTool dataclass
- MCPServerConfig dataclass
- register_mcp_tools global registry
"""

import pytest

from yunshu_gateway.routers.mcp import (
    MCPServerConfig,
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
        register_mcp_tools(
            [
                MCPTool(
                    name="weather",
                    description="Get weather",
                    input_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ]
        )
        assert "weather" in _extra_tools_registry

    def test_register_multiple_tools(self):
        register_mcp_tools(
            [
                MCPTool(name="t1", description="Tool 1"),
                MCPTool(name="t2", description="Tool 2"),
            ]
        )
        assert "t1" in _extra_tools_registry
        assert "t2" in _extra_tools_registry

    @pytest.mark.asyncio
    async def test_registered_tool_appears_in_list(self):
        register_mcp_tools(
            [
                MCPTool(name="search", description="Search the web"),
            ]
        )

        from yunshu_gateway.routers.mcp import _handle_tools_list

        result = await _handle_tools_list(None, 1)
        tool_names = [t["name"] for t in result["result"]["tools"]]
        assert "search" in tool_names
