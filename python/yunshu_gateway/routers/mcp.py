from __future__ import annotations
"""MCP (Model Context Protocol) gateway router.

Implements MCP server protocol (2024-11-05 spec) for LLM tool use:
- JSON-RPC 2.0 over HTTP+SSE
- tools/list: advertise available tools
- tools/call: execute tool calls via the engine
- resources/list + resources/read: expose model info
- prompts/list: expose built-in prompts

MCP allows external agents to use Yunshu's inference capabilities
through a standardized protocol, enabling tool-calling workflows.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])


# ── MCP Data Model (Phase 3) ──


@dataclass
class MCPTool:
    """A single tool advertised by the MCP server.

    Attributes:
        name: Tool identifier (unique within the server).
        description: Human-readable description of what the tool does.
        input_schema: JSON Schema dict describing the tool's input parameters.
        handler: Optional async callable that takes (arguments: dict) -> str.
    """

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    handler: Any = None  # async callable: (arguments: dict) -> str

    def to_dict(self) -> dict:
        """Serialize to MCP tool definition format."""
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.input_schema:
            result["inputSchema"] = self.input_schema
        else:
            result["inputSchema"] = {"type": "object", "properties": {}}
        return result


@dataclass
class MCPServerConfig:
    """Configuration for the MCP server identity.

    Attributes:
        server_name: Name reported in the initialize response.
        version: Version string reported in the initialize response.
        tools: List of tools registered with this server.
    """

    server_name: str = "yunshu"
    version: str = "0.1.0-dev"
    tools: list[MCPTool] = field(default_factory=list)


class MCPSession:
    """Manages a single MCP protocol session.

    Handles JSON-RPC 2.0 message dispatch for the MCP protocol:
    - initialize: returns server info and capabilities
    - tools/list: returns available tools
    - tools/call: executes a tool call
    """

    def __init__(self, config: MCPServerConfig | None = None) -> None:
        self._config = config or MCPServerConfig()
        self._tool_registry: dict[str, MCPTool] = {
            t.name: t for t in self._config.tools
        }
        self._initialized = False

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def handle_message(self, message: dict) -> dict:
        """Dispatch an incoming MCP JSON-RPC message to the correct handler."""
        method = message.get("method", "")
        params = message.get("params")
        req_id = message.get("id")

        dispatch = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        handler = dispatch.get(method)
        if handler is None:
            return _rpc_error(
                JSONRPCError.METHOD_NOT_FOUND,
                f"Method not found: {method}",
                req_id,
            )

        result = handler(params, req_id)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def _handle_initialize(self, params: dict | None, req_id: Any) -> dict:
        """Handle MCP initialize request.

        Returns server info and capabilities per the MCP 2024-11-05 spec.
        """
        self._initialized = True
        return _rpc_response(
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": bool(self._tool_registry)},
                },
                "serverInfo": {
                    "name": self._config.server_name,
                    "version": self._config.version,
                },
            },
            req_id,
        )

    def _handle_tools_list(self, params: dict | None, req_id: Any) -> dict:
        """Handle tools/list request — return all registered tools."""
        tools = [t.to_dict() for t in self._tool_registry.values()]
        return _rpc_response({"tools": tools}, req_id)

    async def _handle_tools_call(self, params: dict | None, req_id: Any) -> dict:
        """Handle tools/call request — execute a tool."""
        if params is None:
            return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing params", req_id)

        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in self._tool_registry:
            return _rpc_error(
                JSONRPCError.METHOD_NOT_FOUND,
                f"Unknown tool: {tool_name}",
                req_id,
            )

        tool = self._tool_registry[tool_name]
        if tool.handler is not None:
            try:
                result_text = tool.handler(arguments)
                if asyncio.iscoroutine(result_text):
                    result_text = await result_text
                return _rpc_response(
                    {
                        "content": [
                            {"type": "text", "text": str(result_text)},
                        ],
                        "isError": False,
                    },
                    req_id,
                )
            except Exception as e:
                return _rpc_response(
                    {
                        "content": [
                            {"type": "text", "text": f"Error: {e}"},
                        ],
                        "isError": True,
                    },
                    req_id,
                )
        else:
            return _rpc_response(
                {
                    "content": [
                        {"type": "text", "text": f"[stub] Tool '{tool_name}' acknowledged. No handler registered."},
                    ],
                    "isError": False,
                },
                req_id,
            )

    def register_tool(self, tool: MCPTool) -> None:
        """Register a single tool with this session."""
        self._tool_registry[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        return self._tool_registry.pop(name, None) is not None

    @property
    def tools(self) -> list[MCPTool]:
        """Return all registered tools."""
        return list(self._tool_registry.values())


def register_mcp_tools(tools: list[MCPTool]) -> None:
    """Register additional tools into the global MCP tool registry.

    Tools registered here are included in the built-in tools/list
    response alongside the default generate/synthesize/generate_image tools.

    Args:
        tools: List of MCPTool instances to register.
    """
    for tool in tools:
        _extra_tools_registry[tool.name] = tool
        logger.info("Registered MCP tool: %s", tool.name)


# Global registry for dynamically registered tools
_extra_tools_registry: dict[str, MCPTool] = {}


# ── JSON-RPC 2.0 ──


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict | None = None
    id: int | str | None = None


class JSONRPCError:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


def _rpc_response(result: Any, req_id: int | str | None) -> dict:
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


def _rpc_error(code: int, message: str, req_id: int | str | None = None) -> dict:
    return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}


# ── MCP Methods ──

async def _handle_initialize(params: dict | None, req_id: int | str | None) -> dict:
    """MCP initialize — server capabilities."""
    return _rpc_response({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {
            "name": "yunshu",
            "version": "0.1.0-dev",
        },
    }, req_id)


async def _handle_tools_list(params: dict | None, req_id: int | str | None) -> dict:
    """List available tools."""
    tools = [
        {
            "name": "generate",
            "description": "Generate text using an LLM. Supports streaming.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model ID"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "max_tokens": {"type": "integer", "default": 512},
                    "temperature": {"type": "number", "default": 0.7},
                    "top_p": {"type": "number", "default": 1.0},
                    "stop": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["model", "messages"],
            },
        },
        {
            "name": "synthesize_speech",
            "description": "Convert text to speech audio.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to synthesize"},
                    "voice": {"type": "string", "default": "A cheerful female voice"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "generate_image",
            "description": "Generate an image from a text prompt.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description"},
                    "width": {"type": "integer", "default": 512},
                    "height": {"type": "integer", "default": 512},
                },
                "required": ["prompt"],
            },
        },
    ]

    # Append dynamically registered tools from the global registry
    for _tool in _extra_tools_registry.values():
        tools.append(_tool.to_dict())

    return _rpc_response({"tools": tools}, req_id)


async def _handle_tools_call(params: dict | None, req_id: int | str | None) -> dict:
    """Execute a tool call."""
    if params is None:
        return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing params", req_id)

    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "generate":
        return await _tool_generate(arguments, req_id)
    elif tool_name == "synthesize_speech":
        return await _tool_synthesize_speech(arguments, req_id)
    elif tool_name == "generate_image":
        return await _tool_generate_image(arguments, req_id)
    elif tool_name in _extra_tools_registry:
        tool = _extra_tools_registry[tool_name]
        if tool.handler is not None:
            try:
                import asyncio
                result_text = await tool.handler(arguments)
                return _rpc_response({
                    "content": [
                        {"type": "text", "text": str(result_text)},
                    ],
                    "isError": False,
                }, req_id)
            except Exception as e:
                return _rpc_response({
                    "content": [
                        {"type": "text", "text": f"Tool execution error: {e}"},
                    ],
                    "isError": True,
                }, req_id)
        else:
            return _rpc_response({
                "content": [
                    {"type": "text", "text": f"Tool '{tool_name}' has no handler registered. Define a handler when registering the tool."},
                ],
                "isError": True,
            }, req_id)
    else:
        return _rpc_error(JSONRPCError.METHOD_NOT_FOUND, f"Unknown tool: {tool_name}", req_id)


async def _tool_generate(args: dict, req_id: int | str | None) -> dict:
    """Execute LLM generation tool."""
    from ..engine import get_engine_for_model, get_engine
    from yunshu_engine.batched_engine import BatchedEngine

    model = args.get("model", "")
    messages = args.get("messages", [])
    max_tokens = args.get("max_tokens", 512)
    temperature = args.get("temperature", 0.7)
    top_p = args.get("top_p", 1.0)
    stop = args.get("stop")

    try:
        # Try BatchedEngine first (multi-model)
        try:
            engine = await get_engine_for_model(model)
        except (KeyError, Exception):
            engine = get_engine()

        if engine is None:
            return _rpc_error(JSONRPCError.INTERNAL_ERROR, "No engine available", req_id)

        is_batched = isinstance(engine, BatchedEngine)

        if is_batched:
            result = await engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            text = result.text
            finish_reason = result.finish_reason
        else:
            state = await engine.generate(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            text = state.generated_text
            finish_reason = state.finish_reason

        return _rpc_response({
            "content": [
                {"type": "text", "text": text},
            ],
            "isError": False,
        }, req_id)

    except Exception as e:
        logger.error(f"MCP generate error: {e}", exc_info=True)
        return _rpc_response({
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        }, req_id)


async def _tool_synthesize_speech(args: dict, req_id: int | str | None) -> dict:
    """Execute TTS tool via TTSEngine."""
    text = args.get("text", "")
    if not text:
        return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing 'text' parameter", req_id)

    from ..engine import get_model_manager
    manager = get_model_manager()
    if manager is None:
        return _rpc_response({
            "content": [{"type": "text", "text": "No model manager available for TTS"}],
            "isError": True,
        }, req_id)

    try:
        from yunshu_engine.audio_engine import TTSEngine
        tts_engine = None
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), TTSEngine):
                tts_engine = entry.engine
                break

        if tts_engine is None:
            return _rpc_response({
                "content": [{"type": "text", "text": "No TTS engine loaded"}],
                "isError": True,
            }, req_id)

        import base64
        voice = args.get("voice", "alloy")
        wav_bytes = await tts_engine.synthesize(text=text, voice=voice)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return _rpc_response({
            "content": [
                {"type": "text", "text": f"Generated speech for: {text[:50]}..."},
                {"type": "audio", "data": audio_b64, "mimeType": "audio/wav"},
            ],
            "isError": False,
        }, req_id)
    except Exception as e:
        logger.debug(f"TTS tool error: {e}", exc_info=True)
        return _rpc_response({
            "content": [{"type": "text", "text": f"TTS error: {e}"}],
            "isError": True,
        }, req_id)


async def _tool_generate_image(args: dict, req_id: int | str | None) -> dict:
    """Execute image generation tool via ImageGenEngine."""
    prompt = args.get("prompt", "")
    if not prompt:
        return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing 'prompt' parameter", req_id)

    from ..engine import get_model_manager
    manager = get_model_manager()
    if manager is None:
        return _rpc_response({
            "content": [{"type": "text", "text": "No model manager available for image generation"}],
            "isError": True,
        }, req_id)

    try:
        from yunshu_engine.image_engine import ImageGenEngine
        img_engine = None
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), ImageGenEngine):
                img_engine = entry.engine
                break

        if img_engine is None:
            return _rpc_response({
                "content": [{"type": "text", "text": "No image generation engine loaded"}],
                "isError": True,
            }, req_id)

        size = args.get("size", "1024x1024")
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)
        except (ValueError, AttributeError):
            width, height = 1024, 1024

        png_bytes = await img_engine.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=args.get("num_inference_steps", 4),
            seed=args.get("seed"),
        )

        import base64
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return _rpc_response({
            "content": [
                {"type": "image", "data": b64, "mimeType": "image/png"},
                {"type": "text", "text": f"Generated {width}x{height} image for: {prompt[:50]}..."},
            ],
            "isError": False,
        }, req_id)
    except Exception as e:
        logger.debug(f"image generation tool error: {e}", exc_info=True)
        return _rpc_response({
            "content": [{"type": "text", "text": f"Image generation error: {e}"}],
            "isError": True,
        }, req_id)


async def _handle_resources_list(params: dict | None, req_id: int | str | None) -> dict:
    """List available resources."""
    from ..engine import get_model_manager

    resources = []
    manager = get_model_manager()
    if manager:
        for model_info in manager.list_models():
            resources.append({
                "uri": f"yunshu://models/{model_info['id']}",
                "name": model_info["id"],
                "description": f"{model_info['type']} model ({model_info['size_gb']:.1f} GB)",
                "mimeType": "application/json",
            })

    return _rpc_response({"resources": resources}, req_id)


async def _handle_resources_read(params: dict | None, req_id: int | str | None) -> dict:
    """Read a specific resource."""
    if params is None or "uri" not in params:
        return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing uri", req_id)

    uri = params["uri"]

    if uri.startswith("yunshu://models/"):
        model_id = uri.replace("yunshu://models/", "")
        from ..engine import get_model_manager
        manager = get_model_manager()
        if manager:
            models = {m["id"]: m for m in manager.list_models()}
            if model_id in models:
                return _rpc_response({
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(models[model_id]),
                    }],
                }, req_id)

    return _rpc_error(JSONRPCError.INVALID_PARAMS, f"Resource not found: {uri}", req_id)


async def _handle_prompts_list(params: dict | None, req_id: int | str | None) -> dict:
    """List available prompts."""
    prompts = [
        {
            "name": "summarize",
            "description": "Summarize the given text",
            "arguments": [
                {"name": "text", "description": "Text to summarize", "required": True},
            ],
        },
        {
            "name": "translate",
            "description": "Translate text between languages",
            "arguments": [
                {"name": "text", "description": "Text to translate", "required": True},
                {"name": "target_language", "description": "Target language", "required": True},
            ],
        },
    ]
    return _rpc_response({"prompts": prompts}, req_id)


# ── Method dispatch ──


async def _noop_handler(params, req_id):
    return _rpc_response({}, req_id)


_METHODS = {
    "initialize": _handle_initialize,
    "notifications/initialized": _noop_handler,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "prompts/list": _handle_prompts_list,
    "ping": _noop_handler,
}


# ── Built-in prompts ──

_PROMPTS = {
    "summarize": {
        "template": "Please summarize the following text concisely:\n\n{text}",
        "arguments": ["text"],
    },
    "translate": {
        "template": "Translate the following text to {target_language}:\n\n{text}",
        "arguments": ["text", "target_language"],
    },
    "code_review": {
        "template": "Review the following {language} code for bugs, style issues, and improvements:\n\n```{language}\n{code}\n```",
        "arguments": ["code", "language"],
    },
    "explain": {
        "template": "Explain the following concept in simple terms:\n\n{concept}",
        "arguments": ["concept"],
    },
}


async def _handle_prompts_get(params: dict | None, req_id: int | str | None) -> dict:
    """Get a specific prompt template with arguments filled in."""
    if params is None or "name" not in params:
        return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing 'name' parameter", req_id)

    name = params["name"]
    if name not in _PROMPTS:
        return _rpc_error(JSONRPCError.INVALID_PARAMS, f"Unknown prompt: {name}", req_id)

    prompt_def = _PROMPTS[name]
    arguments = params.get("arguments", {})

    try:
        import re
        template = re.sub(r'\{(\w+)\}', lambda m: str(arguments.get(m.group(1), m.group(0))), prompt_def["template"])
    except (KeyError, ValueError):
        return _rpc_error(JSONRPCError.INVALID_PARAMS, "Missing template argument", req_id)

    return _rpc_response({
        "description": f"Prompt template: {name}",
        "messages": [
            {"role": "user", "content": {"type": "text", "text": template}},
        ],
    }, req_id)


_METHODS["prompts/get"] = _handle_prompts_get


# ── Endpoint ──


@router.post("/mcp", response_model=None)
async def mcp_endpoint(req: JSONRPCRequest):
    """MCP JSON-RPC endpoint."""
    if req.jsonrpc != "2.0":
        return JSONResponse(_rpc_error(JSONRPCError.INVALID_REQUEST, "Invalid jsonrpc version", req.id))

    handler = _METHODS.get(req.method)
    if handler is None:
        return JSONResponse(_rpc_error(JSONRPCError.METHOD_NOT_FOUND, f"Method not found: {req.method}", req.id))

    try:
        result = await handler(req.params, req.id)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"MCP handler error for {req.method}: {e}", exc_info=True)
        return JSONResponse(_rpc_error(JSONRPCError.INTERNAL_ERROR, str(e), req.id))


@router.get("/mcp/tools")
async def mcp_tools_discovery():
    """REST endpoint for MCP tool discovery."""
    result = await _handle_tools_list(None, None)
    return result.get("result", {})


@router.get("/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """MCP SSE endpoint for streaming connections."""
    import asyncio

    async def _event_stream():
        # Send initial connection event
        yield f"event: endpoint\ndata: /v1/mcp\n\n"
        while True:
            # Poll disconnect every 5s, send keepalive ping every 15s
            try:
                if await request.is_disconnected():
                    break
            except Exception:
                logger.debug("SSE disconnect check failed", exc_info=True)
                break
            await asyncio.sleep(15)
            yield f"event: ping\ndata: {{}}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ── MCP Client endpoints (LLM → external MCP tool servers) ──


@router.get("/mcp/client/status")
async def mcp_client_status(request: Request) -> dict:
    """Get MCP client connection status and discovered tools."""
    mcp_mgr = getattr(request.app.state, "mcp_client", None)
    if mcp_mgr is None:
        return {"enabled": False, "connected_servers": 0, "total_tools": 0}
    return {"enabled": True, **mcp_mgr.get_stats()}


@router.get("/mcp/client/tools")
async def mcp_client_tools(request: Request) -> dict:
    """List tools discovered from connected MCP servers.

    Returns tools in both MCP format and OpenAI function format.
    """
    mcp_mgr = getattr(request.app.state, "mcp_client", None)
    if mcp_mgr is None:
        return {"tools": [], "openai_format": []}
    return {
        "tools": mcp_mgr.list_tools(),
        "openai_format": mcp_mgr.get_tools_as_openai(),
    }
