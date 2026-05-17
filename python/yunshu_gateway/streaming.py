"""Yunshu Production Streaming — SSE keepalive, disconnect guard, thinking parser.

Production-grade streaming implementation based on deep study of:
- oMLX's _with_sse_keepalive (prevents client timeout during long prefill)
- oMLX's _run_with_disconnect_guard (cancels on client disconnect)
- oMLX's _safe_anext (prevents StopAsyncIteration through asyncio.Task)
- oMLX's ThinkingParser (separates reasoning from visible output)
- mlx-lm's NaiveStreamingDetokenizer (correct incremental UTF-8)

Key design decisions:
1. SSE keepalive: inject `: keep-alive\n\n` comments every N seconds during
   prefill (ignored by all SSE parsers but reset client read timeout)
2. Client disconnect detection via is_disconnected() polling + task cancellation
3. No detokenizer pooling (oMLX lesson: reset() leaks internal byte buffers)
4. Thinking tag routing for reasoning models (separate think/reasoning channels)
5. Tool call extraction from model output (XML-based parsing for function calls)
"""


import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

# ── Sentinel for _safe_anext ──

_KEEPALIVE_SENTINEL = object()

logger = logging.getLogger(__name__)


# ── SSE Keepalive Wrapper (oMLX pattern) ──


async def _safe_anext(ait):
    """Wrapper for __anext__ that converts StopAsyncIteration to a sentinel.

    StopAsyncIteration cannot propagate through asyncio.Task (raises RuntimeError),
    so we catch it here and return a sentinel value instead.

    Direct replication of oMLX's _safe_anext pattern.
    """
    try:
        return await ait.__anext__()
    except StopAsyncIteration:
        return _KEEPALIVE_SENTINEL


async def with_sse_keepalive(
    generator: AsyncIterator[str],
    http_request=None,
    interval: float = 10.0,
    disconnect_poll: float = 2.0,
    cancel_event=None,
) -> AsyncIterator[str]:
    """Wrap an SSE generator to send periodic keep-alive comments.

    During long prefill (e.g. 90k tokens), no SSE events are emitted,
    causing clients with read timeouts (like Claude Code) to disconnect.
    This wrapper sends SSE comments (: keep-alive) that are ignored by
    SSE parsers but keep the HTTP connection alive.

    When http_request is provided, also polls for client disconnect
    between prefill steps. This detects cancellation during long prefills
    where uvicorn's ASGI disconnect message is not delivered until after
    the generator yields.

    When cancel_event is provided (an asyncio.Event), it is set when a
    client disconnect is detected. This allows the engine's generation
    loop to cooperatively stop GPU work immediately rather than waiting
    for the next iteration.

    Direct replication of oMLX's _with_sse_keepalive pattern.
    """
    ait = generator.__aiter__()
    task = None
    keepalive_elapsed = 0.0

    # Send initial keepalive immediately so clients with short read
    # timeouts (e.g. openclaw ~15s) don't disconnect during prefill.
    yield ": keep-alive\n\n"

    try:
        while True:
            task = asyncio.ensure_future(_safe_anext(ait))
            keepalive_elapsed = 0.0
            while not task.done():
                # Use shorter poll interval for disconnect detection,
                # accumulate time for keepalive emission
                wait_time = disconnect_poll if http_request else interval
                done, _ = await asyncio.wait({task}, timeout=wait_time)
                if done:
                    break
                # Check for client disconnect
                if http_request is not None:
                    try:
                        disconnected = await http_request.is_disconnected()
                        if disconnected:
                            # Signal the engine to stop GPU work immediately
                            if cancel_event is not None:
                                cancel_event.set()
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, StopAsyncIteration):
                                pass
                            return
                    except Exception:
                        logger.debug("is_disconnected() failed (scope may be closed)", exc_info=True)
                # Send keepalive at the configured interval
                keepalive_elapsed += wait_time
                if keepalive_elapsed >= interval:
                    keepalive_elapsed = 0.0
                    yield ": keep-alive\n\n"
            if task.done():
                try:
                    result = task.result()
                except Exception as e:
                    error_data = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if result is _KEEPALIVE_SENTINEL:
                    return
                yield result
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        if hasattr(ait, 'aclose'):
            await ait.aclose()


async def run_with_disconnect_guard(
    http_request,
    coro,
    poll_interval: float = 1.0,
):
    """Run a coroutine with client disconnect detection.

    For non-streaming requests, FastAPI/uvicorn does NOT automatically cancel
    the handler coroutine when a client disconnects. This helper polls
    is_disconnected() periodically and cancels the task on disconnect,
    which triggers CancelledError -> abort_request() to free GPU resources.

    Direct replication of oMLX's _run_with_disconnect_guard pattern.
    """
    task = asyncio.create_task(coro)
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=poll_interval)
        if done:
            break
        try:
            if await http_request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return None
        except Exception:
            logger.debug("is_disconnected() failed in disconnect guard", exc_info=True)
    return task.result()


# ── Thinking Parser ──


class ThinkingParser:
    """Separates thinking/reasoning content from visible output.

    For models that use <think/>...</think/> tags (DeepSeek-R1, Qwen3, etc.),
    this parser routes:
    - Content inside <think/> → thinking field (not shown to user by default)
    - Content outside → visible text field

    Based on oMLX's ThinkingParser with full streaming support.
    """

    THINK_START = "<think/>"
    THINK_END = "</think/>"

    def __init__(self):
        self.buffer = ""
        self.in_thinking = False
        self.thinking_text = ""
        self.visible_text = ""

    def _retain_tail(self, buf: str) -> tuple[str, str]:
        """Split buffer into safe-to-emit prefix and potential tag-tail suffix."""
        if not buf:
            return "", ""
        for i in range(len(buf) - 1, max(-1, len(buf) - max(len(self.THINK_START), len(self.THINK_END)) - 1), -1):
            tail = buf[i:]
            if self.THINK_START.startswith(tail) or self.THINK_END.startswith(tail):
                return buf[:i], tail
        return buf, ""

    def process_chunk(self, chunk: str) -> dict:
        """Process a text chunk and return structured output."""
        self.buffer += chunk
        visible_parts = []
        thinking_parts = []

        while self.buffer:
            if self.in_thinking:
                end_idx = self.buffer.find(self.THINK_END)
                if end_idx != -1:
                    thinking_parts.append(self.buffer[:end_idx])
                    self.thinking_text += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + len(self.THINK_END):]
                    self.in_thinking = False
                else:
                    emit, retain = self._retain_tail(self.buffer)
                    if emit:
                        thinking_parts.append(emit)
                        self.thinking_text += emit
                    self.buffer = retain
                    break
            else:
                start_idx = self.buffer.find(self.THINK_START)
                if start_idx != -1:
                    if start_idx > 0:
                        visible_parts.append(self.buffer[:start_idx])
                        self.visible_text += self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx + len(self.THINK_START):]
                    self.in_thinking = True
                else:
                    emit, retain = self._retain_tail(self.buffer)
                    if emit:
                        visible_parts.append(emit)
                        self.visible_text += emit
                    self.buffer = retain
                    break

        return {
            "visible": "".join(visible_parts),
            "thinking": "".join(thinking_parts),
            "in_thinking": self.in_thinking,
        }

    def finalize(self) -> dict:
        """Flush any remaining buffer."""
        result = ""
        if self.buffer:
            if self.in_thinking:
                self.thinking_text += self.buffer
                result = self.buffer
            else:
                self.visible_text += self.buffer
                result = self.buffer
            self.buffer = ""
        return {"visible": result if not self.in_thinking else "",
                "thinking": result if self.in_thinking else "",
                "in_thinking": self.in_thinking}


# ── Static Thinking Extraction (for complete outputs) ──

_THINKING_PATTERN = re.compile(r"<think/>(.*?)</think/>", re.DOTALL)
_THINKING_TAIL_PATTERN = re.compile(r"^(.*?)</think/>", re.DOTALL)


def extract_thinking(text: str, model_name: str | None = None) -> tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: <think/>reasoning</think/>answer -> ("reasoning", "answer")
    - No thinking: "just answer" -> ("", "just answer")
    - Partial (no open tag): "reasoning</think/>answer" -> ("reasoning", "answer")
    - Empty think: <think/></think/>answer -> ("", "answer")
    - Think only: <think/>reasoning</think/> -> ("reasoning", "")
    - Gemma4: <start_think/>...</end_think/> markers
    - Harmony: [REASONING]...[/REASONING] markers

    Uses ReasoningParser factory for model-specific extraction when
    model_name is provided. Falls back to oMLX-style <think/> parsing.
    """
    if not text:
        return ("", "")

    # Try model-specific reasoning parser (Gemma, Harmony, etc.)
    if model_name:
        try:
            from yunshu_engine.reasoning_parser import get_reasoning_parser
            parser = get_reasoning_parser(model_name)
            if parser.family_name() != "generic":
                out = parser.parse(text)
                if out.reasoning:
                    return (out.reasoning, out.content)
        except Exception:
            logger.debug("reasoning parser failed", exc_info=True)

    # Also handle Gemma4 <start_think/>...</end_think/> without model_name
    _GEMMA_THINK_PATTERN = re.compile(r"<start_think\s*/?\s*>(.*?)</end_think\s*/?\s*>(.*)", re.DOTALL)
    gemma_m = _GEMMA_THINK_PATTERN.match(text)
    if gemma_m:
        return (gemma_m.group(1).strip(), gemma_m.group(2).strip())

    # Also handle Harmony [REASONING]...[/REASONING]
    _HARMONY_PATTERN = re.compile(r"\[REASONING\](.*?)\[/REASONING\](.*)", re.DOTALL | re.IGNORECASE)
    harmony_m = _HARMONY_PATTERN.match(text)
    if harmony_m:
        return (harmony_m.group(1).strip(), harmony_m.group(2).strip())

    thinking_parts = []
    remaining = text

    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]

    if thinking_parts:
        thinking = "\n".join(thinking_parts).strip()
        return (thinking, remaining.strip())

    # Handle partial: content before </think/> without <think/> tag
    if "</think/>" in text and "<think/>" not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            return (thinking, remaining)

    return ("", text)


# ── Tool Call Extraction ──

# Alternative pattern: function call in markdown code blocks
_FUNC_CALL_PATTERN = re.compile(
    r'```(?:python|json|tool)\s*\n(.*?)\n```',
    re.DOTALL,
)


def _sanitize_arguments(args: Any) -> str:
    """Ensure tool call arguments are a valid JSON object string."""
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return "{}"
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return "{}"


def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output.

    Supports multiple formats:
    1. <tool_call/> tags (Hermes format)
    2. Qwen/Llama XML: <function=name>...</function>
    3. Direct JSON with name/arguments
    4. Code block embedded JSON

    Returns list of {"name": str, "arguments": str} dicts.
    """
    tool_calls = []

    # Pattern 1: <tool_call/> tags (Hermes/Llama format)
    hermes_pattern = re.compile(
        r"<tool_call\s*/?\s*>(.*?)</tool_call\s*/?\s*>",
        re.DOTALL,
    )
    for match in hermes_pattern.finditer(text):
        content = match.group(1).strip()
        try:
            data = json.loads(content)
            if "name" in data:
                tool_calls.append({
                    "name": data["name"],
                    "arguments": _sanitize_arguments(data.get("arguments", data.get("parameters", {}))),
                })
        except json.JSONDecodeError:
            continue

    # Pattern 2: Qwen/Llama XML format <function=name>...</function>
    if not tool_calls:
        qwen_pattern = re.compile(
            r"<function\s*=\s*(\w+)>(.*?)</function>",
            re.DOTALL,
        )
        for match in qwen_pattern.finditer(text):
            name = match.group(1).strip()
            body = match.group(2).strip()
            try:
                data = json.loads(body)
                tool_calls.append({
                    "name": name,
                    "arguments": _sanitize_arguments(data),
                })
            except json.JSONDecodeError:
                # Try parameter extraction
                params = {}
                param_pattern = re.compile(r"<parameter\s*=\s*(\w+)>(.*?)</parameter>", re.DOTALL)
                for pm in param_pattern.finditer(body):
                    params[pm.group(1)] = pm.group(2).strip()
                if params:
                    tool_calls.append({
                        "name": name,
                        "arguments": json.dumps(params, ensure_ascii=False),
                    })

    # Pattern 3: Direct JSON with name field
    if not tool_calls:
        for match in _FUNC_CALL_PATTERN.finditer(text):
            content = match.group(1).strip()
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append({
                        "name": data["name"],
                        "arguments": _sanitize_arguments(data.get("arguments", data.get("parameters", {}))),
                    })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            tool_calls.append({
                                "name": item["name"],
                                "arguments": _sanitize_arguments(item.get("arguments", item.get("parameters", {}))),
                            })
            except json.JSONDecodeError:
                continue

    return tool_calls


# ── Additional Tool Call Parsers (C15: vllm-mlx model format coverage) ──

# Pattern 5: Mistral-style {"function": {"name": ..., "arguments": ...}}
_MISTRAL_TOOL_RE = re.compile(
    r'\{[\s\S]*?"function"[\s\S]*?"name"[\s\S]*?\}',
)
# Pattern 6: ChatML [TOOL_CALLS] [{...}]
_CHATML_TOOL_RE = re.compile(
    r'\[TOOL_CALLS\]\s*(\[.*?\])',
    re.DOTALL,
)
# Pattern 7: DeepSeek-style ✿FUNCTION✿ markers
_DEEPSEEK_TOOL_RE = re.compile(
    r'✿FUNCTION✿\s*(\{.*?\})\s*✿',
    re.DOTALL,
)


def extract_tool_calls_v2(text: str) -> list[dict]:
    """Extended tool call parser with 7 format support.

    Adds to the original extract_tool_calls:
    5. Mistral function-call JSON format
    6. ChatML [TOOL_CALLS] array format
    7. DeepSeek ✿FUNCTION✿ markers

    Falls back to extract_tool_calls for formats 1-4.
    """
    # Try the original parser first
    calls = extract_tool_calls(text)
    if calls:
        return calls

    # Pattern 5: Mistral-style {"function": {"name": ..., "arguments": ...}}
    # Try to find and parse JSON objects containing "function" key
    _brace_depth = 0
    _json_start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if _brace_depth == 0:
                _json_start = i
            _brace_depth += 1
        elif ch == '}':
            _brace_depth -= 1
            if _brace_depth == 0 and _json_start >= 0:
                candidate = text[_json_start:i + 1]
                try:
                    data = json.loads(candidate)
                    # Mistral format: {"function": {"name": ...}}
                    func = data.get("function", {})
                    name = func.get("name", "") if isinstance(func, dict) else ""
                    # Also handle bare {"name": ..., "arguments": ...} format
                    if not name and "name" in data and isinstance(data.get("name"), str):
                        name = data["name"]
                        func = data
                    if name:
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
                except (json.JSONDecodeError, KeyError):
                    pass
                _json_start = -1
    if calls:
        return calls

    # Pattern 6: ChatML [TOOL_CALLS] [{...}]
    for match in _CHATML_TOOL_RE.finditer(text):
        try:
            arr = json.loads(match.group(1))
            for item in arr:
                if isinstance(item, dict):
                    name = item.get("name", item.get("function", {}).get("name", ""))
                    args = item.get("arguments", item.get("function", {}).get("arguments", {}))
                    if name:
                        if isinstance(args, str):
                            args = json.loads(args)
                        calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else args})
        except (json.JSONDecodeError, KeyError):
            continue
    if calls:
        return calls

    # Pattern 7: DeepSeek ✿FUNCTION✿ markers
    for match in _DEEPSEEK_TOOL_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            if name:
                args = data.get("arguments", data.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else args})
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


def clean_tool_call_markup(text: str) -> str:
    """Remove tool call markup from text, leaving clean content."""
    # Remove <tool_call/>...</tool_call/> blocks
    text = re.sub(r"<tool_call\s*/?\s*>.*?</tool_call\s*/?\s*>", "", text, flags=re.DOTALL)
    # Remove <function=name>...</function> blocks
    text = re.sub(r"<function\s*=\s*\w+>.*?</function>", "", text, flags=re.DOTALL)
    # Remove empty lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Model-aware tool call extraction using ToolCallParser (C15)
_tool_call_parsers: dict[str, Any] = {}


def extract_tool_calls_model_aware(text: str, model_name: str = "") -> list[dict]:
    """Extract tool calls using model-aware format detection (C15).

    Falls back to extract_tool_calls_v2 for formats not covered
    by the ToolCallParser module.
    """
    if not text:
        return []

    # Get or create parser for this model
    parser = _tool_call_parsers.get(model_name)
    if parser is None:
        from yunshu_engine.tool_call_parsers import ToolCallParser
        parser = ToolCallParser(model_name=model_name)
        _tool_call_parsers[model_name] = parser

    results = parser.parse(text)
    if results:
        return [{"name": r.name, "arguments": r.arguments} for r in results]

    # Fall back to v2 parser
    return extract_tool_calls_v2(text)


# ── Context Window Validation ──


def get_max_context_window(model_id: str | None = None, engine=None) -> int | None:
    """Get effective max context window limit.

    Checks model config for max_position_embeddings or similar fields.
    Returns None if not determinable.
    """
    if engine is None:
        return None

    # Check tokenizer/model for context window
    model = getattr(engine, '_model', None)
    if model is not None:
        config = getattr(model, 'config', None)
        if config is not None:
            for key in ('max_position_embeddings', 'max_seq_len', 'n_positions', 'model_max_length'):
                val = getattr(config, key, None) or (config.get(key) if isinstance(config, dict) else None)
                if val is not None:
                    return int(val)

    # Check tokenizer
    tokenizer = getattr(engine, '_tokenizer', None)
    if tokenizer is not None:
        model_max = getattr(tokenizer, 'model_max_length', None)
        if model_max is not None and model_max < 1_000_000:
            return int(model_max)

    return None


def validate_context_window(
    num_prompt_tokens: int,
    model_id: str | None = None,
    engine=None,
) -> None:
    """Validate that prompt token count does not exceed max context window.

    Raises HTTPException 400 if the prompt is too long.
    Direct replication of oMLX's validate_context_window pattern.
    """
    from fastapi import HTTPException

    max_ctx = get_max_context_window(model_id, engine)
    if max_ctx and num_prompt_tokens > max_ctx:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too long: {num_prompt_tokens} tokens exceeds "
                f"max context window of {max_ctx} tokens"
            ),
        )


# ── OpenAI SSE Formatter ──


def format_openai_chunk(
    completion_id: str,
    model: str,
    delta_content: str,
    finish_reason: Optional[str] = None,
    thinking_content: Optional[str] = None,
    tool_calls: Optional[list[dict]] = None,
    logprobs: Optional[dict] = None,
    include_role: bool = False,
    choice_index: int = 0,
) -> str:
    """Format a single SSE chunk in OpenAI Chat Completions format."""
    delta = {"content": delta_content}
    if include_role:
        delta["role"] = "assistant"
    if thinking_content:
        delta["reasoning_content"] = thinking_content

    choice: dict[str, Any] = {
        "index": choice_index,
        "delta": delta,
        "finish_reason": finish_reason,
    }
    if logprobs:
        choice["logprobs"] = logprobs

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_openai_done() -> str:
    """SSE stream termination signal."""
    return "data: [DONE]\n\n"


def format_openai_completion_chunk(
    completion_id: str,
    model: str,
    text: str,
    finish_reason: Optional[str] = None,
    logprobs: Optional[dict] = None,
    choice_index: int = 0,
) -> str:
    """Format a single SSE chunk in OpenAI Text Completions format.

    OpenAI /v1/completions streaming uses choices[].text (not choices[].delta).
    The object type is "text_completion" (not "chat.completion.chunk").
    """
    choice: dict[str, Any] = {
        "index": choice_index,
        "text": text,
        "finish_reason": finish_reason,
    }
    if logprobs:
        choice["logprobs"] = logprobs

    chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_openai_completion_usage_chunk(
    completion_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> str:
    """Format final SSE chunk with usage stats for Text Completions endpoint.

    Same as format_openai_usage_chunk but uses "text_completion" object type.
    """
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if reasoning_tokens > 0:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    if cached_tokens > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_openai_usage_chunk(
    completion_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> str:
    """Format final SSE chunk with usage stats (stream_options.include_usage).

    OpenAI sends a final chunk with usage when stream_options.include_usage=True.
    The chunk has an empty delta and the usage field at the top level.
    """
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if reasoning_tokens > 0:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    if cached_tokens > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_openai_non_stream(
    completion_id: str,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    thinking_content: Optional[str] = None,
    tool_calls: Optional[list[dict]] = None,
    logprobs: Optional[dict] = None,
) -> dict:
    """Format a complete (non-streaming) OpenAI Chat Completion response."""
    message = {"role": "assistant", "content": content}
    if thinking_content:
        message["reasoning_content"] = thinking_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i:x}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": _sanitize_arguments(tc.get("arguments", {})),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
        finish_reason = "tool_calls"

    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
    }
    if logprobs:
        choice["logprobs"] = logprobs

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ── Anthropic SSE Formatter ──


def format_anthropic_chunk(
    message_id: str,
    model: str,
    delta_text: str,
    event_type: str = "content_block_delta",
) -> str:
    """Format a single SSE chunk in Anthropic Messages format."""
    data = {
        "type": event_type,
        "index": 0,
    }
    if event_type == "content_block_delta":
        data["delta"] = {"type": "text_delta", "text": delta_text}
    elif event_type == "message_start":
        data["message"] = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    elif event_type == "message_delta":
        data["delta"] = {"stop_reason": "end_turn"}
        data["usage"] = {"output_tokens": 1}

    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Non-streaming keepalive wrapper ──

async def with_json_keepalive(
    http_request,
    coro,
    interval_s: float = 5.0,
):
    """For non-streaming requests, send space keepalive during long prefill.

    JSON parsers ignore leading whitespace, so we send space characters
    to keep the connection alive during long prefill operations.

    This is oMLX's _with_json_keepalive pattern.
    """
    task = asyncio.create_task(coro)

    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=interval_s)
        if done:
            break
        # Check for client disconnect
        try:
            if await http_request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return
        except Exception:
            logger.debug("is_disconnected() failed in json keepalive", exc_info=True)
        yield " "

    if task.done() and not task.cancelled():
        result = task.result()
        yield result


