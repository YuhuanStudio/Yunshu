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
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])

# ── Safety limits ──

# Maximum time to wait for an external tool handler (seconds).
_TOOL_CALL_TIMEOUT = 30

# Maximum size of a single tool result text (bytes). Larger results are truncated.
_MAX_TOOL_RESULT_BYTES = 1_000_000  # 1 MB


def _truncate_tool_result(text: str, max_bytes: int = _MAX_TOOL_RESULT_BYTES) -> str:
    """Truncate tool result text if it exceeds max_bytes when UTF-8 encoded."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="replace")
    return truncated + f"\n... [truncated, {len(encoded)} bytes total]"


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

    async def handle_message(self, message: dict) -> dict | None:
        """Dispatch an incoming MCP JSON-RPC message to the correct handler.

        Returns None for notifications (no 'id' field) per JSON-RPC 2.0 spec.
        """
        method = message.get("method", "")
        params = message.get("params")
        req_id = message.get("id")

        # Per JSON-RPC 2.0: notifications (no id) MUST NOT receive a response
        is_notification = "id" not in message

        dispatch = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        handler = dispatch.get(method)
        if handler is None:
            if is_notification:
                return None
            return _rpc_error(
                JSONRPCError.METHOD_NOT_FOUND,
                f"Method not found: {method}",
                req_id,
            )

        result = handler(params, req_id)
        if asyncio.iscoroutine(result):
            result = await result

        if is_notification:
            return None
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
                    result_text = await asyncio.wait_for(
                        result_text, timeout=_TOOL_CALL_TIMEOUT,
                    )
                return _rpc_response(
                    {
                        "content": [
                            {"type": "text", "text": _truncate_tool_result(str(result_text))},
                        ],
                        "isError": False,
                    },
                    req_id,
                )
            except TimeoutError:
                return _rpc_response(
                    {
                        "content": [
                            {"type": "text", "text": f"Tool '{tool_name}' timed out after {_TOOL_CALL_TIMEOUT}s"},
                        ],
                        "isError": True,
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
            # A registered tool with no handler can't actually run. Report it as an
            # ERROR (not a misleading success) so the model/caller doesn't believe a
            # no-op tool executed — an MCP server is meant to EXECUTE tools/call.
            return _rpc_response(
                {
                    "content": [
                        {"type": "text", "text": f"Tool '{tool_name}' has no handler registered and cannot be executed."},
                    ],
                    "isError": True,
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
                    "top_k": {"type": "integer", "default": 0},
                    "min_p": {"type": "number", "default": 0.0},
                    "repetition_penalty": {"type": "number", "default": 1.0},
                    "frequency_penalty": {"type": "number", "default": 0.0},
                    "presence_penalty": {"type": "number", "default": 0.0},
                    "enable_thinking": {"type": "boolean"},
                    "seed": {"type": "integer"},
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
                    # advertised default must match the tool's actual default
                    # (args.get("voice","alloy")); the prior "A cheerful female voice"
                    # over-advertised a default the tool never used.
                    "voice": {"type": "string", "default": "alloy"},
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
                result_text = await asyncio.wait_for(
                    tool.handler(arguments), timeout=_TOOL_CALL_TIMEOUT,
                )
                return _rpc_response({
                    "content": [
                        {"type": "text", "text": _truncate_tool_result(str(result_text))},
                    ],
                    "isError": False,
                }, req_id)
            except TimeoutError:
                return _rpc_response({
                    "content": [
                        {"type": "text", "text": f"Tool '{tool_name}' timed out after {_TOOL_CALL_TIMEOUT}s"},
                    ],
                    "isError": True,
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
    from yunshu_engine.batched_engine import BatchedEngine

    from ..engine import get_engine, get_engine_for_model

    model = args.get("model", "")
    messages = args.get("messages", [])
    max_tokens = args.get("max_tokens", 512)
    temperature = args.get("temperature", 0.7)
    top_p = args.get("top_p", 1.0)
    top_k = args.get("top_k", 0)
    min_p = args.get("min_p", 0.0)
    repetition_penalty = args.get("repetition_penalty", 1.0)
    frequency_penalty = args.get("frequency_penalty", 0.0)
    presence_penalty = args.get("presence_penalty", 0.0)
    enable_thinking = args.get("enable_thinking")
    seed = args.get("seed")
    stop = args.get("stop")

    try:
        # SECURITY: resolve ONLY the requested model — never silently fall
        # back to the default engine. The old `except (KeyError, Exception): get_engine()`
        # caught EVERY failure and served the default model, which (1) returned a
        # different model's output with no error when the requested one failed to load,
        # and (2) bypassed the per-key model-isolation gate: the auth boundary checks
        # the REQUESTED model, but the served engine was the default → a key scoped away
        # from the default could drive it by naming a model that fails to resolve. Mirror
        # embeddings/scoring (which refuse rather than cross-model fall back). An OMITTED
        # model legitimately uses the default engine (gated at the dispatch boundary).
        if model:
            try:
                engine = await get_engine_for_model(model)
            except KeyError:
                return _rpc_error(
                    JSONRPCError.INVALID_PARAMS,
                    f"Model '{model}' not found or not loaded", req_id,
                )
            # Any other failure (load/OOM) propagates to the outer handler as an
            # isError result — NOT a silent fallback to a different model.
        else:
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
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                enable_thinking=enable_thinking,
                seed=seed,
                stop=stop,
            )
            text = result.text
        else:
            state = await engine.generate(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                enable_thinking=enable_thinking,
                seed=seed,
                stop=stop,
            )
            text = state.generated_text

        return _rpc_response({
            "content": [
                {"type": "text", "text": _truncate_tool_result(text)},
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

        # Truncate oversized base64 audio to prevent response explosion
        audio_data = audio_b64 if len(audio_b64) <= _MAX_TOOL_RESULT_BYTES else audio_b64[:_MAX_TOOL_RESULT_BYTES] + "... [truncated]"

        return _rpc_response({
            "content": [
                {"type": "text", "text": f"Generated speech for: {text[:50]}..."},
                {"type": "audio", "data": audio_data, "mimeType": "audio/wav"},
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

        # honor the ADVERTISED width/height integer params (the inputSchema
        # declares width/height, default 512). The tool previously read an unadvertised
        # `size` "WxH" string and IGNORED width/height entirely, so a spec-conformant
        # client sending {"width":512,"height":512} silently always got 1024x1024. Read
        # width/height first; accept a `size` string only as a backward-compat fallback
        # for whichever dimension was not given; default 512 to match the schema.
        width = args.get("width")
        height = args.get("height")
        if width is None or height is None:
            size = args.get("size")
            if size:
                try:
                    _w, _h = str(size).lower().split("x")
                    width = int(_w) if width is None else width
                    height = int(_h) if height is None else height
                except (ValueError, AttributeError):
                    pass
        try:
            width = int(width) if width is not None else 512
            height = int(height) if height is not None else 512
        except (ValueError, TypeError):
            width, height = 512, 512

        png_bytes = await img_engine.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=args.get("num_inference_steps", 4),
            seed=args.get("seed"),
        )

        import base64
        b64 = base64.b64encode(png_bytes).decode("ascii")
        # Truncate oversized base64 image to prevent response explosion
        image_data = b64 if len(b64) <= _MAX_TOOL_RESULT_BYTES else b64[:_MAX_TOOL_RESULT_BYTES] + "... [truncated]"
        return _rpc_response({
            "content": [
                {"type": "image", "data": image_data, "mimeType": "image/png"},
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


async def _handle_resources_list(params: dict | None, req_id: int | str | None,
                                 rbac_key: Any = None) -> dict:
    """List available resources.

    SECURITY: a model-scoped RBAC key must NOT see (id/type/size of) models it
    can't use — mirror the per-key isolation that /v1/models already applies. The MCP
    resources surface enumerated EVERY model behind only can_infer, leaking the existence,
    type, and size of inaccessible models.
    """
    from ..engine import get_model_manager

    resources = []
    manager = get_model_manager()
    if manager:
        for model_info in manager.list_models():
            if rbac_key is not None and not rbac_key.can_access_model(model_info["id"]):
                continue
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
    """List available prompts. Built from _PROMPTS (single source of truth) so it
    can never diverge from what prompts/get actually serves."""
    prompts = [
        {
            "name": name,
            "description": spec["description"],
            "arguments": spec["arguments"],
        }
        for name, spec in _PROMPTS.items()
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
        "description": "Summarize the given text",
        "template": "Please summarize the following text concisely:\n\n{text}",
        "arguments": [
            {"name": "text", "description": "Text to summarize", "required": True},
        ],
    },
    "translate": {
        "description": "Translate text between languages",
        "template": "Translate the following text to {target_language}:\n\n{text}",
        "arguments": [
            {"name": "text", "description": "Text to translate", "required": True},
            {"name": "target_language", "description": "Target language", "required": True},
        ],
    },
    "code_review": {
        "description": "Review code for bugs, style issues, and improvements",
        "template": "Review the following {language} code for bugs, style issues, and improvements:\n\n```{language}\n{code}\n```",
        "arguments": [
            {"name": "code", "description": "Source code to review", "required": True},
            {"name": "language", "description": "Programming language", "required": True},
        ],
    },
    "explain": {
        "description": "Explain a concept in simple terms",
        "template": "Explain the following concept in simple terms:\n\n{concept}",
        "arguments": [
            {"name": "concept", "description": "Concept to explain", "required": True},
        ],
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

    # Enforce declared required arguments. The regex substitution below leaves the
    # literal placeholder ("{text}") in place for a missing key, so without this
    # check a caller that omits a required arg would get a malformed prompt with
    # unfilled placeholders instead of an INVALID_PARAMS error.
    missing = [
        a["name"] for a in prompt_def["arguments"]
        if a.get("required") and a["name"] not in arguments
    ]
    if missing:
        return _rpc_error(
            JSONRPCError.INVALID_PARAMS,
            f"Missing required argument(s): {', '.join(missing)}", req_id,
        )

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
async def mcp_endpoint(request: Request):
    """MCP JSON-RPC endpoint.

    Per JSON-RPC 2.0 spec, notifications (requests without an 'id') MUST NOT
    receive a response. We acknowledge them silently.
    """
    from pydantic import ValidationError

    from .models import _check_model_access, _check_permission
    # an MCP permission/auth denial must return a JSON-RPC error object (not the
    # OpenAI HTTP envelope a JSON-RPC client can't parse). _check_permission raises
    # HTTPException — convert it. id is unknown (body not parsed yet) → null per JSON-RPC.
    try:
        _check_permission(request, "can_infer")
    except HTTPException as _perm_err:
        return JSONResponse(
            _rpc_error(JSONRPCError.INVALID_REQUEST, f"Forbidden: {_perm_err.detail}", None),
            status_code=_perm_err.status_code,
        )

    # (protocol conformance): parse + validate the JSON-RPC envelope
    # OURSELVES. The endpoint previously took `req: JSONRPCRequest`, so FastAPI's
    # pydantic validation rejected malformed bodies (missing method, wrong-typed
    # params, invalid JSON) with an OpenAI-style HTTP error envelope — which a
    # JSON-RPC/MCP client cannot parse and which drops the request id. Emit proper
    # JSON-RPC 2.0 error objects (PARSE_ERROR / INVALID_REQUEST) instead.
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_rpc_error(JSONRPCError.PARSE_ERROR, "Parse error: invalid JSON", None))
    if not isinstance(body, dict):
        # Batch arrays / scalars are not supported — respond with a single error.
        return JSONResponse(_rpc_error(
            JSONRPCError.INVALID_REQUEST, "Invalid Request: expected a JSON-RPC object", None))
    # Per JSON-RPC 2.0 an explicit `id: null` is a REQUEST; only an OMITTED id is a
    # notification. pydantic collapses both to None, so read the RAW body to tell
    # them apart (previously `req.id is None` treated id:null as a notification).
    _raw_id = body.get("id")
    is_notification = "id" not in body
    try:
        req = JSONRPCRequest(**body)
    except ValidationError as e:
        if is_notification:
            return JSONResponse(content=None, status_code=204)
        _msg = e.errors()[0].get("msg", "validation error") if e.errors() else "validation error"
        return JSONResponse(_rpc_error(JSONRPCError.INVALID_REQUEST, f"Invalid Request: {_msg}", _raw_id))
    # the `generate` tool resolves an arbitrary
    # body["arguments"]["model"] via get_engine_for_model with no model-access check,
    # so a key scoped to model A could run inference on any loaded model B through MCP.
    # Enforce per-key model isolation at the auth boundary (mirrors every other route).
    # a model-isolation denial here (HTTPException 403) must also surface as a
    # JSON-RPC error, not the OpenAI HTTP envelope. _raw_id is known now, so echo it.
    try:
      if req.method == "tools/call" and isinstance(req.params, dict):
        _p = req.params
        if _p.get("name") == "generate":
            # gate REGARDLESS of the arguments shape. The old `and
            # isinstance(arguments, dict)` meant a tools/call for generate with arguments
            # OMITTED (or a non-dict) skipped the check entirely → _tool_generate fell
            # through to the default engine (model=""), so a key scoped away from the
            # default model could drive it by just omitting arguments (the model-access hole).
            _args = _p.get("arguments")
            _gen_model = _args.get("model") if isinstance(_args, dict) else None
            if not _gen_model:
                # an omitted model makes _tool_generate serve the DEFAULT
                # engine; gate THAT model's id (else _check_model_access no-ops on the
                # empty string and a key scoped away from the default drives it).
                try:
                    from ..engine import get_engine as _ge
                    _gen_model = getattr(_ge(), "model_name", None)
                except Exception:
                    _gen_model = None
            _check_model_access(request, _gen_model)
        elif _p.get("name") in ("synthesize_speech", "generate_image"):
            # SECURITY: these modality tools resolve the first loaded
            # TTS/Image engine with NO access check, so a key scoped away from that
            # model could drive it via MCP — bypassing the isolation enforced on
            # /v1/audio/speech and /v1/images. Resolve the model the tool would use
            # and gate it (mirrors the `generate` branch).
            try:
                from yunshu_engine.model_manager import ModelType

                from ..engine import get_model_manager
                _mgr = get_model_manager()
                _want = ModelType.TTS if _p["name"] == "synthesize_speech" else ModelType.IMAGE_GEN
                _mid = None
                if _mgr is not None:
                    for _e in _mgr.list_entries():
                        if _e.is_loaded and _e.model_type == _want:
                            _mid = _e.model_id
                            break
            except Exception:
                _mid = None
            _check_model_access(request, _mid)
      elif req.method == "resources/read" and isinstance(req.params, dict):
        # resources/read returns full per-model metadata for an arbitrary
        # yunshu://models/<id> URI behind only can_infer — gate the resolved model id so
        # a key scoped away from it can't read its status/size/load-error.
        _uri = req.params.get("uri", "")
        if isinstance(_uri, str) and _uri.startswith("yunshu://models/"):
            _check_model_access(request, _uri.replace("yunshu://models/", "", 1))
    except HTTPException as _acc_err:
        if is_notification:
            return JSONResponse(content=None, status_code=204)
        return JSONResponse(
            _rpc_error(JSONRPCError.INVALID_REQUEST, f"Forbidden: {_acc_err.detail}", _raw_id),
            status_code=_acc_err.status_code,
        )
    # is_notification was determined from the RAW body above (an OMITTED id, not an
    # explicit id:null). The Server MUST NOT reply to a Notification.

    if req.jsonrpc != "2.0":
        if is_notification:
            return JSONResponse(content=None, status_code=204)
        return JSONResponse(_rpc_error(JSONRPCError.INVALID_REQUEST, "Invalid jsonrpc version", req.id))

    handler = _METHODS.get(req.method)
    if handler is None:
        if is_notification:
            return JSONResponse(content=None, status_code=204)
        return JSONResponse(_rpc_error(JSONRPCError.METHOD_NOT_FOUND, f"Method not found: {req.method}", req.id))

    try:
        # resources/list must filter by the caller's per-key model access.
        if req.method == "resources/list":
            result = await _handle_resources_list(
                req.params, req.id, getattr(request.state, "rbac_key", None))
        else:
            result = await handler(req.params, req.id)
        if is_notification:
            return JSONResponse(content=None, status_code=204)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"MCP handler error for {req.method}: {e}", exc_info=True)
        if is_notification:
            return JSONResponse(content=None, status_code=204)
        # do NOT echo the raw exception string (str(e)) to the client — it can
        # carry a file/model path or internal detail. The traceback is logged above;
        # return a generic message (mirrors the global exception handler's non-leak policy).
        return JSONResponse(_rpc_error(JSONRPCError.INTERNAL_ERROR, "Internal error", req.id))


@router.get("/mcp/tools")
async def mcp_tools_discovery(request: Request):
    """REST endpoint for MCP tool discovery."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    result = await _handle_tools_list(None, None)
    return result.get("result", {})


@router.get("/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """MCP SSE endpoint for streaming connections."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    import asyncio

    async def _event_stream():
        # Send initial connection event
        yield "event: endpoint\ndata: /v1/mcp\n\n"
        loop = asyncio.get_running_loop()
        last_ping = loop.time()
        while True:
            # Poll for disconnect every 5s
            await asyncio.sleep(5)
            try:
                if await request.is_disconnected():
                    break
            except Exception:
                logger.debug("SSE disconnect check failed", exc_info=True)
                break
            # Send keepalive ping every 15s
            now = loop.time()
            if now - last_ping >= 15:
                last_ping = now
                yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ── MCP Client endpoints (LLM → external MCP tool servers) ──


@router.get("/mcp/client/status")
async def mcp_client_status(request: Request) -> dict:
    """Get MCP client connection status and discovered tools."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    mcp_mgr = getattr(request.app.state, "mcp_client", None)
    if mcp_mgr is None:
        return {"enabled": False, "connected_servers": 0, "total_tools": 0}
    return {"enabled": True, **mcp_mgr.get_stats()}


@router.get("/mcp/client/tools")
async def mcp_client_tools(request: Request) -> dict:
    """List tools discovered from connected MCP servers.

    Returns tools in both MCP format and OpenAI function format.
    """
    from .models import _check_permission
    _check_permission(request, "can_infer")
    mcp_mgr = getattr(request.app.state, "mcp_client", None)
    if mcp_mgr is None:
        return {"tools": [], "openai_format": []}
    return {
        "tools": mcp_mgr.list_tools(),
        "openai_format": mcp_mgr.get_tools_as_openai(),
    }
