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

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Optional

# ── Sentinel for _safe_anext ──

_KEEPALIVE_SENTINEL = object()


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
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, StopAsyncIteration):
                                pass
                            return
                    except Exception:
                        pass  # is_disconnected() can fail if scope is already closed
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
            pass
    return task.result()


# ── Legacy SSEKeepaliveWrapper (backward compatible) ──


class SSEKeepaliveWrapper:
    """Simplified SSE keepalive for backward compatibility.

    Prefer with_sse_keepalive() for production use.
    """

    def __init__(
        self,
        source: AsyncIterator,
        interval_s: float = 10.0,
        disconnect_check=None,
    ):
        self._source = source
        self._interval = interval_s
        self._disconnect_check = disconnect_check

    async def __aiter__(self):
        while True:
            try:
                result = await asyncio.wait_for(
                    self._source.__anext__(),
                    timeout=self._interval,
                )
                yield result
            except asyncio.TimeoutError:
                if self._disconnect_check and self._disconnect_check():
                    return
                yield ": keep-alive\n\n"
            except StopAsyncIteration:
                return


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


def extract_thinking(text: str) -> tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: <think/>reasoning</think/>answer -> ("reasoning", "answer")
    - No thinking: "just answer" -> ("", "just answer")
    - Partial (no open tag): "reasoning</think/>answer" -> ("reasoning", "answer")
    - Empty think: <think/></think/>answer -> ("", "answer")
    - Think only: <think/>reasoning</think/> -> ("reasoning", "")

    Direct replication of oMLX's extract_thinking pattern.
    """
    if not text:
        return ("", "")

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

_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call\s*>.*?["\']name["\']\s*:\s*["\']([^"\']+)["\'].*?["\']arguments["\']\s*:\s*(\{.*?\})\s*.*?</tool_call\s*>',
    re.DOTALL,
)

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


def clean_tool_call_markup(text: str) -> str:
    """Remove tool call markup from text, leaving clean content."""
    # Remove <tool_call/>...</tool_call/> blocks
    text = re.sub(r"<tool_call\s*/?\s*>.*?</tool_call\s*/?\s*>", "", text, flags=re.DOTALL)
    # Remove <function=name>...</function> blocks
    text = re.sub(r"<function\s*=\s*\w+>.*?</function>", "", text, flags=re.DOTALL)
    # Remove empty lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
) -> str:
    """Format a single SSE chunk in OpenAI Chat Completions format."""
    delta = {"role": "assistant", "content": delta_content}
    if thinking_content:
        delta["reasoning_content"] = thinking_content

    choice: dict[str, Any] = {
        "index": 0,
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


def format_openai_usage_chunk(
    completion_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> str:
    """Format final SSE chunk with usage stats (stream_options.include_usage).

    OpenAI sends a final chunk with usage when stream_options.include_usage=True.
    The chunk has an empty delta and the usage field at the top level.
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
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
                    "arguments": tc["arguments"],
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
            pass
        yield " "

    if task.done() and not task.cancelled():
        result = task.result()
        yield result


# ── Token Rate Tracker ──


class TokenRateTracker:
    """Tracks generation speed (tokens/sec) during streaming."""

    def __init__(self):
        self._tokens: int = 0
        self._start_time: Optional[float] = None

    def record(self, token_count: int = 1) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        self._tokens += token_count

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self._tokens / elapsed

    @property
    def total_tokens(self) -> int:
        return self._tokens

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


# ── Stop Sequence Detector ──


class StopSequenceDetector:
    """Detects stop sequences that may span across streaming chunks."""

    def __init__(self, stop_sequences: list[str]):
        self._sequences = stop_sequences
        self._buffer = ""

    def check(self, text: str) -> tuple[str, str | None]:
        """Check text for stop sequences.

        Returns (clean_text, matched_stop or None).
        """
        if not self._sequences:
            return text, None

        self._buffer += text
        # Check if any stop sequence appears in the combined buffer
        for seq in self._sequences:
            idx = self._buffer.find(seq)
            if idx != -1:
                clean = self._buffer[:idx]
                self._buffer = ""
                return clean, seq

        # Check if buffer tail could be a partial stop sequence
        safe_end = len(self._buffer)
        for seq in self._sequences:
            for i in range(1, min(len(seq), len(self._buffer)) + 1):
                if seq.startswith(self._buffer[-i:]):
                    safe_end = min(safe_end, len(self._buffer) - i)
                    break

        if safe_end < len(self._buffer):
            safe_text = self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return safe_text, None

        safe_text = self._buffer
        self._buffer = ""
        return safe_text, None

    def finalize(self) -> str | None:
        """Flush remaining buffer. No stop detection on finalize."""
        remaining = self._buffer
        self._buffer = ""
        return remaining if remaining else None
