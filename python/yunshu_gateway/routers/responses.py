"""OpenAI Responses API compatible router.

The Responses API is OpenAI's newest API format that combines
chat completions with tool use, structured output, and streaming
into a unified interface.

Supports:
- Text and chat responses
- Streaming via SSE
- Tool use (function calling)
- Structured output (response_format)
- Image and audio input (routed to VLM/Omni)
"""
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..engine import get_engine, get_engine_for_model

logger = logging.getLogger(__name__)
from ..streaming import format_openai_chunk

router = APIRouter(tags=["responses"])


class ResponseInputText(BaseModel):
    type: str = "message"
    role: str = "user"
    content: str | list[dict]


class ResponseTool(BaseModel):
    type: str = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[ResponseInputText]
    instructions: Optional[str] = None
    max_output_tokens: int = 2048
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    tools: Optional[list[ResponseTool]] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = Field(default=None, ge=1, le=32768)
    reasoning_effort: Optional[str] = None
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = None
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    stop: Optional[list[str]] = None
    lora_adapter: Optional[str] = None


def _convert_to_messages(req: ResponsesRequest) -> list[dict]:
    """Convert Responses API input to OpenAI chat messages."""
    messages = []

    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
    elif isinstance(req.input, list):
        for item in req.input:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                messages.append({"role": role, "content": content})
            elif hasattr(item, "content"):
                role = getattr(item, "role", "user")
                content = item.content
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    messages.append({"role": role, "content": content})

    return messages


def _parse_response_format(rf: dict | None) -> dict | None:
    if rf is None:
        return None
    rf_type = rf.get("type")
    if rf_type == "json_schema":
        js = rf.get("json_schema", {})
        return js.get("schema", js)
    elif rf_type == "json_object":
        return {}
    return None


@router.post("/responses", response_model=None)
async def create_response(req: ResponsesRequest, request: Request):
    """OpenAI Responses API endpoint."""
    messages = _convert_to_messages(req)
    json_schema = _parse_response_format(req.response_format)

    engine = get_engine()
    if engine is None or not engine.is_loaded or not engine.resolve_model_id(req.model):
        try:
            engine = await get_engine_for_model(req.model)
        except (KeyError, Exception) as e:
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found: {e}")

    # Check for VLM/audio routing
    from .chat import _has_images, _has_audio
    has_media = _has_images(messages) or _has_audio(messages)
    if has_media:
        from .chat import _handle_vlm_chat
        from .chat import ChatCompletionRequest
        chat_req = ChatCompletionRequest(
            model=req.model,
            messages=messages,
            max_tokens=req.max_output_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            enable_thinking=req.enable_thinking,
            stop=req.stop,
            stream=False,
            lora_adapter=req.lora_adapter,
        )
        return await _handle_vlm_chat(chat_req, messages, request)

    # Inject tool definitions
    if req.tools:
        from .chat import _inject_tool_system_prompt
        from .chat import ToolDefinition, ToolFunction
        tools = [
            ToolDefinition(function=ToolFunction(
                name=t.name, description=t.description, parameters=t.parameters,
            ))
            for t in req.tools
        ]
        messages = _inject_tool_system_prompt(messages, tools)

    response_id = f"resp-{uuid.uuid4().hex[:24]}"

    # LoRA adapter lifecycle
    from .chat import _apply_lora_adapter, _release_lora_adapter
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)

    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(engine, req, messages, response_id, json_schema),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming
        from yunshu_engine.batched_engine import BatchedEngine
        is_batched = isinstance(engine, BatchedEngine)

        if is_batched:
            result = await engine.generate(
                prompt=messages,
                max_tokens=req.max_output_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                min_p=req.min_p,
                json_schema=json_schema,
                stop=req.stop,
            )
            text = result.text
            pt = result.prompt_tokens
            ct = result.completion_tokens
            finish_reason = result.finish_reason or "stop"
        else:
            state = await engine.generate(
                prompt=messages,
                max_tokens=req.max_output_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                min_p=req.min_p,
                stop=req.stop,
            )
            text = state.generated_text
            pt = state.prompt_token_count
            ct = state.completion_token_count
            finish_reason = state.finish_reason or "stop"
            _reasoning_tokens = getattr(state, 'reasoning_tokens', 0)

        # Extract tool calls
        tool_calls = None
        if req.tools:
            from .chat import extract_tool_calls, clean_tool_call_markup
            tool_calls = extract_tool_calls(text)
            if tool_calls:
                text = clean_tool_call_markup(text)
                finish_reason = "tool_calls"

        # Build output items
        output_items = []
        content_parts = [{"type": "output_text", "text": text.strip()}]
        output_items.append({
            "type": "message",
            "id": f"msg-{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "content": content_parts,
        })

        if tool_calls:
            for tc in tool_calls:
                output_items.append({
                    "type": "function_call",
                    "id": f"fc-{uuid.uuid4().hex[:24]}",
                    "call_id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                })

        return JSONResponse({
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": req.model,
            "status": "completed",
            "output": output_items,
            "usage": {
                "input_tokens": pt,
                "output_tokens": ct,
                "total_tokens": pt + ct,
                **({"output_tokens_details": {"reasoning_tokens": _reasoning_tokens}} if _reasoning_tokens else {}),
            },
        })
    finally:
        _release_lora_adapter(engine, loaded_adapter)


async def _stream_response(engine, req, messages, response_id, json_schema):
    """SSE streaming for Responses API."""
    from ..streaming import with_sse_keepalive
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    async def _token_source():
        if is_batched:
            async for output in engine.stream_generate(
                prompt=messages,
                max_tokens=req.max_output_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                min_p=req.min_p,
                json_schema=json_schema,
                stop=req.stop,
            ):
                yield format_openai_chunk(
                    completion_id=response_id,
                    model=req.model,
                    delta_content=output.new_text,
                    finish_reason=output.finish_reason,
                )
        else:
            async for output in engine.generate_stream(
                prompt=messages,
                max_tokens=req.max_output_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                min_p=req.min_p,
                stop=req.stop,
            ):
                yield format_openai_chunk(
                    completion_id=response_id,
                    model=req.model,
                    delta_content=output.new_text,
                    finish_reason=output.finish_reason,
                )

    async for chunk in with_sse_keepalive(_token_source()):
        yield chunk
