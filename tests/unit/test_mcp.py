"""MCP (Model Context Protocol) gateway tests."""

import pytest

from yunshu_gateway.routers.mcp import (
    JSONRPCRequest,
    _handle_initialize,
    _handle_tools_list,
    _handle_tools_call,
    _handle_prompts_list,
    _handle_resources_list,
)


class TestMCPInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self):
        result = await _handle_initialize(None, 1)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"]["protocolVersion"] == "2024-11-05"
        assert "tools" in result["result"]["capabilities"]
        assert result["result"]["serverInfo"]["name"] == "yunshu"


class TestMCPToolsList:
    @pytest.mark.asyncio
    async def test_tools_list_returns_tools(self):
        result = await _handle_tools_list(None, 2)
        tools = result["result"]["tools"]
        assert len(tools) >= 3

        tool_names = [t["name"] for t in tools]
        assert "generate" in tool_names
        assert "synthesize_speech" in tool_names
        assert "generate_image" in tool_names

    @pytest.mark.asyncio
    async def test_generate_tool_has_input_schema(self):
        result = await _handle_tools_list(None, 1)
        gen_tool = next(t for t in result["result"]["tools"] if t["name"] == "generate")
        assert "inputSchema" in gen_tool
        assert "messages" in gen_tool["inputSchema"]["properties"]


class TestMCPToolsCall:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        result = await _handle_tools_call({"name": "nonexistent"}, 1)
        assert "error" in result
        assert result["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_missing_params_returns_error(self):
        result = await _handle_tools_call(None, 1)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_speech_tool_without_text_returns_error(self):
        result = await _handle_tools_call({"name": "synthesize_speech", "arguments": {}}, 1)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_image_tool_without_prompt_returns_error(self):
        result = await _handle_tools_call({"name": "generate_image", "arguments": {}}, 1)
        assert "error" in result


class TestMCPPrompts:
    @pytest.mark.asyncio
    async def test_prompts_list(self):
        result = await _handle_prompts_list(None, 1)
        prompts = result["result"]["prompts"]
        assert len(prompts) >= 2
        names = [p["name"] for p in prompts]
        assert "summarize" in names
        assert "translate" in names


class TestMCPResources:
    @pytest.mark.asyncio
    async def test_resources_list_no_manager(self):
        result = await _handle_resources_list(None, 1)
        resources = result["result"]["resources"]
        # Without model manager, list is empty
        assert isinstance(resources, list)


class TestJSONRPCRequest:
    def test_valid_request(self):
        req = JSONRPCRequest(method="initialize", id=1)
        assert req.jsonrpc == "2.0"
        assert req.method == "initialize"
        assert req.id == 1
