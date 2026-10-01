"""Yunshu Production Streaming — SSE keepalive, disconnect guard, thinking parser.

Production-grade streaming implementation:
- _with_sse_keepalive (prevents client timeout during long prefill)
- _run_with_disconnect_guard (cancels on client disconnect)
- _safe_anext (prevents StopAsyncIteration through asyncio.Task)
- ThinkingParser (separates reasoning from visible output)
- mlx-lm's NaiveStreamingDetokenizer (correct incremental UTF-8)

Key design decisions:
1. SSE keepalive: inject `: keep-alive\n\n` comments every N seconds during
   prefill (ignored by all SSE parsers but reset client read timeout)
2. Client disconnect detection via is_disconnected() polling + task cancellation
3. No detokenizer pooling (reset() leaks internal byte buffers)
4. Thinking tag routing for reasoning models (separate think/reasoning channels)
5. Tool call extraction from model output (XML-based parsing for function calls)
"""

import asyncio
import contextlib
import contextvars
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import anyio
from starlette.responses import StreamingResponse

from yunshu_engine import settings
from yunshu_engine.tool_arguments import arguments_json

from .middleware.disconnect import client_disconnected

# ── Sentinel for _safe_anext ──

_KEEPALIVE_SENTINEL = object()

logger = logging.getLogger(__name__)


class ClosingStreamingResponse(StreamingResponse):
    """Close the body iterator when ASGI send fails after a client disconnect.

    Starlette's stream_response iterates the body but does not close an async
    generator if send raises. That leaves nested inference generators suspended
    with their request lease and cleanup finally blocks unexecuted.
    """

    async def stream_response(self, send):
        try:
            await super().stream_response(send)
        finally:
            if hasattr(self.body_iterator, "aclose"):
                # Starlette's ASGI <2.4 disconnect listener cancels the task
                # group. Cleanup awaits need shielding or cancellation aborts
                # them before the nested inference generator can release its
                # request lease.
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()


# ── SSE Keepalive Wrapper ──


async def _safe_anext(ait):
    """Wrapper for __anext__ that converts StopAsyncIteration to a sentinel.

    StopAsyncIteration cannot propagate through asyncio.Task (raises RuntimeError),
    so we catch it here and return a sentinel value instead.
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
                        disconnected = await client_disconnected(http_request)
                        if disconnected:
                            # Signal the engine to stop GPU work immediately
                            if cancel_event is not None:
                                cancel_event.set()
                            task.cancel()
                            with contextlib.suppress(
                                asyncio.CancelledError, StopAsyncIteration
                            ):
                                await task
                            return
                    except Exception:
                        logger.debug(
                            "is_disconnected() failed (scope may be closed)",
                            exc_info=True,
                        )
                # Send keepalive at the configured interval
                keepalive_elapsed += wait_time
                if keepalive_elapsed >= interval:
                    keepalive_elapsed = 0.0
                    yield ": keep-alive\n\n"
            if task.done():
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    return
                except StopAsyncIteration:
                    return
                except Exception:
                    # Re-raise so the calling router (OpenAI/Anthropic) can
                    # emit a protocol-correct error event.  Yielding OpenAI-
                    # style data: [DONE] here breaks Anthropic streaming which
                    # uses event: message_stop instead.
                    raise
                if result is _KEEPALIVE_SENTINEL:
                    return
                yield result
    finally:
        # on ANY exit — crucially a client disconnect, which starlette
        # surfaces by throwing GeneratorExit/CancelledError INTO this generator (both
        # BaseException, not Exception) so it BYPASSES the is_disconnected() poll above
        # and the routers' `except` branches that set cancel_event — signal the engine's
        # decode loop to stop. task.cancel() alone is a no-op on a generate_step already
        # running on the max_workers=1 MLX executor thread; only cancel_event stops it,
        # else a disconnected stream runs to max_tokens, wasting GPU AND head-of-line-
        # blocking every subsequent request on the serialized executor. Harmless on
        # normal completion (the loop already ended).
        if cancel_event is not None:
            with contextlib.suppress(Exception):
                cancel_event.set()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await task
        if hasattr(ait, "aclose"):
            await ait.aclose()


async def run_with_disconnect_guard(
    http_request,
    coro,
    poll_interval: float = 1.0,
    cancel_event=None,
):
    """Run a coroutine with client disconnect detection.

    For non-streaming requests, FastAPI/uvicorn does NOT automatically cancel
    the handler coroutine when a client disconnects. This helper polls
    is_disconnected() periodically and cancels the task on disconnect,
    which triggers CancelledError -> abort_request() to free GPU resources.

    Cancelling the asyncio task does NOT interrupt the engine's decode
    loop running on the max_workers=1 MLX executor thread — that loop only stops
    when its threading/asyncio `cancel_event` is set. Without setting it, a
    non-streaming client disconnect let generation run to max_tokens / the 300s
    timeout, wasting the GPU AND head-of-line-blocking every other request on the
    serial executor. Set `cancel_event` (if provided) on disconnect so the loop
    breaks promptly (the streaming path already does this via with_sse_keepalive).
    """
    task = asyncio.create_task(coro)
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=poll_interval)
        if done:
            break
        try:
            if await client_disconnected(http_request):
                if cancel_event is not None:
                    try:
                        cancel_event.set()
                    except Exception:
                        logger.debug(
                            "cancel_event.set() failed on disconnect", exc_info=True
                        )
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return None
        except Exception:
            logger.debug("is_disconnected() failed in disconnect guard", exc_info=True)
    try:
        return task.result()
    except asyncio.CancelledError:
        return None


# ── Thinking Parser ──


class ThinkingParser:
    """Separates thinking/reasoning content from visible output.

    For models that use <think/>...</think/> tags (DeepSeek-R1, Qwen3, etc.),
    this parser routes:
    - Content inside <think/> → thinking field (not shown to user by default)
    - Content outside → visible text field

    Supports ALL common tag variants that models actually emit:
      <think/>, <think >, <think\\>, <think\\n>, </think/>, </think >, etc.
    Plus the most common format: plain <think...> and </think...> (without
    self-closing slash), consistent with extract_thinking()'s regex pattern
    ``<think\\s*/?\\s*>``.

    Full streaming support.
    """

    # All known literal opening tag variants (models emit these interchangeably).
    # The regex-based finder also handles the generic <think...> and </think...>.
    THINK_STARTS = ("<think/>", "<think >", "<think\\>")
    THINK_ENDS = ("</think/>", "</think >", "</think\\>")

    # Regex patterns for generic tag matching — same grammar as extract_thinking().
    _OPEN_RE = re.compile(r"<think\s*/?\s*>")
    _CLOSE_RE = re.compile(r"</think\s*/?\s*>")

    def __init__(self):
        self.buffer = ""
        self.in_thinking = False
        self.thinking_text = ""
        self.visible_text = ""

    def _find_tag_start(self, buf: str) -> tuple[int, int]:
        """Find the earliest opening think tag in buf.

        Returns (position, tag_length) or (-1, 0) if not found.
        Uses regex to support all variants including plain <think...>.
        """
        m = self._OPEN_RE.search(buf)
        if m:
            return m.start(), m.end() - m.start()
        return -1, 0

    def _find_tag_end(self, buf: str) -> tuple[int, int]:
        """Find the earliest closing think tag in buf.

        Returns (position, tag_length) or (-1, 0) if not found.
        Uses regex to support all variants including plain </think...>.
        """
        m = self._CLOSE_RE.search(buf)
        if m:
            return m.start(), m.end() - m.start()
        return -1, 0

    # Maximum tag length for _retain_tail buffer retention.
    # "</think/>" = 9 chars; "</think >" = 9; "</think\\>" = 9;
    # The regex can match "<think\\n>" = 8; use 10 for safety.
    _MAX_TAG_LEN = 10
    _MAX_ACCUMULATOR_SIZE = 1 * 1024 * 1024  # 1 MB per accumulator

    @staticmethod
    def _is_partial_tag(tail: str, tag_prefix: str) -> bool:
        """True when ``tail`` can still grow into ``<think\\s*/?\\s*>`` (or the closing form)."""
        if len(tail) <= len(tag_prefix):
            return tag_prefix.startswith(tail)
        if not tail.startswith(tag_prefix):
            return False
        return re.fullmatch(r"\s*/?\s*", tail[len(tag_prefix) :]) is not None

    def _retain_tail(self, buf: str) -> tuple[str, str]:
        """Split buffer into safe-to-emit prefix and potential tag-tail suffix.

        Only considers tags relevant to the current mode:
        - When not in thinking: only opening tag prefixes (THINK_STARTS + ``<think``)
        - When in thinking: only closing tag prefixes (THINK_ENDS + ``</think``)

        This prevents unnecessary retention of irrelevant tag prefixes
        (e.g. retaining ``</thi`` when not in thinking mode).
        """
        if not buf:
            return "", ""
        # Scan backwards from the end of the buffer to find the longest
        # suffix that could be the start of a tag.  Everything before
        # that suffix is safe to emit immediately.
        tag_prefix = "</think" if self.in_thinking else "<think"
        # The tag grammar is ``<think\s*/?\s*>``: a tail such as ``<think  `` (any
        # amount of whitespace / a slash, not yet the ``>``) is a partial tag too,
        # however the stream is chunked. A partial tag holds exactly one ``<``, at
        # its start, so only the last ``<`` can begin one.
        lt = buf.rfind("<")
        if (
            lt != -1
            and len(buf) - lt <= 64
            and self._is_partial_tag(buf[lt:], tag_prefix)
        ):
            return buf[:lt], buf[lt:]
        for i in range(len(buf) - 1, max(-1, len(buf) - self._MAX_TAG_LEN - 1), -1):
            tail = buf[i:]
            # Check if tail could be a prefix of any relevant tag.
            # The literal THINK_STARTS/THINK_ENDS cover specific variants,
            # and tag_prefix covers the generic <think...> / </think...>.
            # only `tag_prefix.startswith(tail)` (tail IS a partial tag
            # prefix). The old `or tail.startswith(tag_prefix)` over-retained literal
            # text that merely CONTAINS the prefix (e.g. "<thinker", "<think A>"),
            # withholding it from the visible stream; complete tags at the tail are
            # already covered by the THINK_STARTS/ENDS check below.
            is_tag_prefix = tag_prefix.startswith(tail)
            if not is_tag_prefix:
                tags = self.THINK_ENDS if self.in_thinking else self.THINK_STARTS
                for tag in tags:
                    if tag.startswith(tail):
                        is_tag_prefix = True
                        break
            if is_tag_prefix:
                return buf[:i], tail
        return buf, ""

    def process_chunk(self, chunk: str) -> dict:
        """Process a text chunk and return structured output."""
        self.buffer += chunk
        # Truncate unbounded accumulators to prevent memory growth
        if len(self.thinking_text) > self._MAX_ACCUMULATOR_SIZE:
            self.thinking_text = self.thinking_text[-self._MAX_ACCUMULATOR_SIZE // 2 :]
        if len(self.visible_text) > self._MAX_ACCUMULATOR_SIZE:
            self.visible_text = self.visible_text[-self._MAX_ACCUMULATOR_SIZE // 2 :]
        visible_parts = []
        thinking_parts = []

        while self.buffer:
            if self.in_thinking:
                end_idx, end_len = self._find_tag_end(self.buffer)
                if end_idx != -1:
                    thinking_parts.append(self.buffer[:end_idx])
                    self.thinking_text += self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + end_len :]
                    self.in_thinking = False
                else:
                    emit, retain = self._retain_tail(self.buffer)
                    if emit:
                        thinking_parts.append(emit)
                        self.thinking_text += emit
                    self.buffer = retain
                    break
            else:
                start_idx, start_len = self._find_tag_start(self.buffer)
                if start_idx != -1:
                    if start_idx > 0:
                        visible_parts.append(self.buffer[:start_idx])
                        self.visible_text += self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx + start_len :]
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
                # Check for partial tag at the end — at stream end, any
                # partial tag is just text, not a real tag marker.
                emit, retain = self._retain_tail(self.buffer)
                full_emit = emit + retain
                if full_emit:
                    self.visible_text += full_emit
                    result = full_emit
                self.buffer = ""
        return {
            "visible": result if not self.in_thinking else "",
            "thinking": result if self.in_thinking else "",
            "in_thinking": self.in_thinking,
        }


# ── Static Thinking Extraction (for complete outputs) ──

_THINKING_PATTERN = re.compile(r"<think\s*/?\s*>(.*?)</think\s*/?\s*>", re.DOTALL)
_THINKING_TAIL_PATTERN = re.compile(r"^(.*?)</think\s*/?\s*>", re.DOTALL)
_GEMMA_THINK_PATTERN = re.compile(
    r"<start_think\s*/?\s*>(.*?)</end_think\s*/?\s*>(.*)", re.DOTALL
)
_HARMONY_PATTERN = re.compile(
    r"\[REASONING\](.*?)\[/REASONING\](.*)", re.DOTALL | re.IGNORECASE
)


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
    model_name is provided. Falls back to <think/> parsing.
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
    gemma_m = _GEMMA_THINK_PATTERN.match(text)
    if gemma_m:
        return (gemma_m.group(1).strip(), gemma_m.group(2).strip())

    # Also handle Harmony [REASONING]...[/REASONING]
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
        remaining = remaining[: match.start()] + remaining[match.end() :]

    if thinking_parts:
        thinking = "\n".join(thinking_parts).strip()
        return (thinking, remaining.strip())

    # Handle partial: content before </think ...> without <think ...> tag
    _CLOSE_RE = re.compile(r"</think\s*/?\s*>")
    _OPEN_RE = re.compile(r"<think\s*/?\s*>")
    if _CLOSE_RE.search(text) and not _OPEN_RE.search(text):
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end() :].strip()
            return (thinking, remaining)

    return ("", text)


# ── Context Window Validation ──


def get_max_context_window(model_id: str | None = None, engine=None) -> int | None:
    """Get effective max context window limit.

    Checks model config for max_position_embeddings or similar fields.
    Returns None if not determinable.
    """
    if engine is None:
        return None

    # Check the model for the real context window. mlx-lm Model objects
    # store their ModelArgs in `.args`, NOT `.config` — the old code only checked
    # `.config`, so for every standard mlx-lm LLM the config branch was dead and it
    # fell through to tokenizer.model_max_length, which is often far larger than
    # the true window (e.g. Qwen2.5-3B: model_max_length=131072 vs real 32768),
    # letting over-long prompts past the guard into RoPE-extrapolated garbage. The
    # model's own value MUST win over the tokenizer fallback. Mirror the correct
    # resolver in routers/tokenize.py (_resolve_context_limit).
    model = getattr(engine, "_model", None)
    if model is not None:
        for attr in ("max_seq_len", "max_position_embeddings"):
            v = getattr(model, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        for sub in ("config", "args"):
            s = getattr(model, sub, None)
            if s is not None:
                for key in (
                    "max_position_embeddings",
                    "max_seq_len",
                    "n_positions",
                    "context_length",
                ):
                    val = getattr(s, key, None)
                    if val is None and isinstance(s, dict):
                        val = s.get(key)
                    if isinstance(val, int) and val > 0:
                        return int(val)

    # Engine-level config dict (VLM path)
    for cfg_attr in ("_config", "config"):
        cfg = getattr(engine, cfg_attr, None)
        if isinstance(cfg, dict):
            for k in ("max_position_embeddings", "context_length", "max_seq_len"):
                v = cfg.get(k)
                if isinstance(v, int) and v > 0:
                    return int(v)
            tc = cfg.get("text_config") or (cfg.get("thinker_config") or {}).get(
                "text_config"
            )
            if isinstance(tc, dict):
                for k in ("max_position_embeddings", "context_length", "max_seq_len"):
                    v = tc.get(k)
                    if isinstance(v, int) and v > 0:
                        return int(v)

    # Tokenizer fallback (least trustworthy — only when the model exposed nothing)
    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is not None:
        model_max = getattr(tokenizer, "model_max_length", None)
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
    """
    from fastapi import HTTPException

    max_ctx = get_max_context_window(model_id, engine)
    if max_ctx and num_prompt_tokens > max_ctx:
        raise HTTPException(
            status_code=400,
            # The leading clause is Anthropic's own wording: Claude Code parses
            # "prompt is too long: N tokens > M" to learn the real window and compact.
            detail=(
                f"prompt is too long: {num_prompt_tokens} tokens > {max_ctx} maximum; "
                f"the prompt exceeds max context window of {max_ctx} tokens"
            ),
        )


def get_max_prefill_tokens() -> int:
    """Optional per-request prefill-token cap — OFF by default.

    mlx-lm's generate_step already CHUNKS the prefill (prefill_step_size=2048), so
    the prefill activation peak is bounded and prompts up to the full context
    window are processable: measured a 200k-token prefill at peak 12.3GB / 167s on
    a 36GB M3 Max (KV is only ~13KB/token → ~2.6GB at 200k). So there is no OOM to
    guard against by default — a huge prompt just takes longer.

    This cap exists only as an OPT-IN operator control: a single huge prompt
    monopolizes the single-threaded MLX executor for the duration of its prefill,
    so a multi-tenant / latency-sensitive deployment may want to bound it. Default
    is effectively unlimited (the context-window check is the real bound); set
    YUNSHU_MAX_PREFILL_TOKENS to a positive value to enforce a cap.
    """

    v = settings.get("YUNSHU_MAX_PREFILL_TOKENS")
    # 0 / unset / non-positive → no cap (very large sentinel)
    return v if v > 0 else 1_000_000_000


def validate_prefill_memory(num_prompt_tokens: int) -> None:
    """Enforce the OPTIONAL operator prefill cap (YUNSHU_MAX_PREFILL_TOKENS).

    OFF by default — prompts up to the context window prefill fine (just slowly);
    see get_max_prefill_tokens(). When an operator sets a cap (e.g. for multi-
    tenant latency/fairness, so one huge prompt can't monopolize the executor),
    an over-cap prompt is rejected 413. Runs AFTER validate_context_window so a
    genuinely over-window prompt still gets the 400 "context window" error.
    """
    from fastapi import HTTPException

    cap = get_max_prefill_tokens()
    if num_prompt_tokens > cap:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Prompt exceeds the configured prefill limit: {num_prompt_tokens} "
                f"tokens > YUNSHU_MAX_PREFILL_TOKENS={cap} (operator latency cap; "
                f"raise or unset it to allow larger prompts)"
            ),
        )


# ── OpenAI SSE Formatter ──


def format_openai_chunk(
    completion_id: str,
    model: str,
    delta_content: str,
    finish_reason: str | None = None,
    thinking_content: str | None = None,
    tool_calls: list[dict] | None = None,
    logprobs: dict | None = None,
    include_role: bool = False,
    choice_index: int = 0,
) -> str:
    """Format a single SSE chunk in OpenAI Chat Completions format."""
    # OpenAI spec: first (role-only) chunk sends content: null, not content: ""
    if include_role and not delta_content:
        delta = {"content": None}
    else:
        delta = {"content": delta_content}
    if include_role:
        delta["role"] = "assistant"
    if thinking_content:
        delta["reasoning_content"] = thinking_content
    if tool_calls:
        delta["tool_calls"] = tool_calls

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
    finish_reason: str | None = None,
    logprobs: dict | None = None,
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

    ``completion_tokens`` MUST already include reasoning tokens (the engine's
    n_tok counts every generated token); ``reasoning_tokens`` is the detail
    SUBSET, per OpenAI semantics — do NOT add it to the totals or reasoning is
    double-counted.
    """
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
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

    ``completion_tokens`` MUST already include reasoning tokens (the engine's
    n_tok counts every generated token); ``reasoning_tokens`` is the detail
    SUBSET, per OpenAI semantics — do NOT add it to the totals or reasoning is
    double-counted (was inflating completion_tokens/total_tokens for thinking
    models with include_usage=true).
    """
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    # Detail fields are subsets of their totals, never addends or larger than them.
    usage["completion_tokens_details"] = {
        "reasoning_tokens": max(0, min(reasoning_tokens, completion_tokens))
    }
    usage["prompt_tokens_details"] = {
        "cached_tokens": max(0, min(cached_tokens, prompt_tokens))
    }
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
    thinking_content: str | None = None,
    tool_calls: list[dict] | None = None,
    logprobs: dict | None = None,
) -> dict:
    """Format a complete (non-streaming) OpenAI Chat Completion response."""
    message = {"role": "assistant", "content": content}
    if thinking_content:
        message["reasoning_content"] = thinking_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": arguments_json(tc.get("arguments", {})),
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


# ── OpenAI Responses API SSE Formatters ──

# The Responses API uses its own SSE event types (response.created,
# response.output_text.delta, response.completed, etc.) distinct from
# Chat Completions.  Each SSE line uses the form:
# event: <event_type>\ndata: <json>\n\n
# The JSON payload carries a `type` field matching the event type plus
# event-specific fields defined in the openai.types.responses package.


# The request's echoed configuration (instructions, temperature, tools, truncation, ...):
# set by the Responses route so every response.* event carries the same fields the
# final Response object has, as OpenAI's do.
RESPONSES_ECHO: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "yunshu_responses_echo", default=None
)


def _responses_base_response(
    response_id: str,
    model: str,
    status: str = "in_progress",
    created_at: float | None = None,
    output: list | None = None,
    usage: dict | None = None,
) -> dict:
    """Build a minimal Response object for Responses API events."""
    resp: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "model": model,
        "status": status,
        "output": output or [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    echo = RESPONSES_ECHO.get()
    if echo:
        resp.update(echo)
    if usage is not None:
        resp["usage"] = usage
    return resp


def format_responses_created(
    response_id: str,
    model: str,
    seq: int = 0,
) -> str:
    """response.created — initial event with empty response object."""
    data = {
        "type": "response.created",
        "response": _responses_base_response(response_id, model, status="created"),
        "sequence_number": seq,
    }
    return f"event: response.created\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_in_progress(
    response_id: str,
    model: str,
    seq: int = 1,
) -> str:
    """response.in_progress — response processing started."""
    data = {
        "type": "response.in_progress",
        "response": _responses_base_response(response_id, model, status="in_progress"),
        "sequence_number": seq,
    }
    return (
        f"event: response.in_progress\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def format_responses_output_item_added(
    response_id: str,
    model: str,
    item_id: str,
    output_index: int = 0,
    seq: int = 2,
    item_type: str = "message",
    call_id: str | None = None,
    name: str | None = None,
    arguments: str = "",
) -> str:
    """response.output_item.added — new output item (message or function_call) added."""
    if item_type == "message":
        item = {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [],
            "status": "in_progress",
        }
    elif item_type == "function_call":
        # the OpenAI Responses spec requires the function_call item on
        # output_item.added to carry call_id/name/arguments (a strict client reads name/
        # call_id off the `added` event). The old generic branch emitted only {type,id,
        # status}, so those were missing until the later .done event.
        item = {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id or "",
            "name": name or "",
            "arguments": arguments,
            "status": "in_progress",
        }
    else:
        # Generic item — caller will fill details via done event
        item = {
            "type": item_type,
            "id": item_id,
            "status": "in_progress",
        }
    data = {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": item,
        "sequence_number": seq,
    }
    return f"event: response.output_item.added\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_content_part_added(
    item_id: str,
    output_index: int = 0,
    content_index: int = 0,
    seq: int = 3,
) -> str:
    """response.content_part.added — content part added to output item."""
    part = {
        "type": "output_text",
        "text": "",
        "annotations": [],
    }
    data = {
        "type": "response.content_part.added",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "part": part,
        "sequence_number": seq,
    }
    return f"event: response.content_part.added\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_text_delta(
    delta: str,
    item_id: str,
    output_index: int = 0,
    content_index: int = 0,
    logprobs: list | None = None,
    seq: int = 4,
) -> str:
    """response.output_text.delta — text content delta."""
    data: dict[str, Any] = {
        "type": "response.output_text.delta",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "delta": delta,
        "logprobs": logprobs or [],
        "sequence_number": seq,
    }
    return f"event: response.output_text.delta\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_text_done(
    text: str,
    item_id: str,
    output_index: int = 0,
    content_index: int = 0,
    logprobs: list | None = None,
    seq: int = 0,
) -> str:
    """response.output_text.done — text content completed."""
    data: dict[str, Any] = {
        "type": "response.output_text.done",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "text": text,
        "logprobs": logprobs or [],
        "sequence_number": seq,
    }
    return f"event: response.output_text.done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_content_part_done(
    item_id: str,
    text: str = "",
    output_index: int = 0,
    content_index: int = 0,
    seq: int = 0,
) -> str:
    """response.content_part.done — content part completed."""
    part = {
        "type": "output_text",
        "text": text,
        "annotations": [],
    }
    data = {
        "type": "response.content_part.done",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "part": part,
        "sequence_number": seq,
    }
    return f"event: response.content_part.done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_output_item_done(
    item_id: str,
    text: str = "",
    output_index: int = 0,
    seq: int = 0,
) -> str:
    """response.output_item.done — output item completed."""
    item: dict[str, Any] = {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
        "status": "completed",
    }
    data = {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": item,
        "sequence_number": seq,
    }
    return f"event: response.output_item.done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_completed(
    response_id: str,
    model: str,
    output: list,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    seq: int = 0,
) -> str:
    """response.completed — final event with full response and usage."""
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
    completed_at = int(time.time())
    resp = _responses_base_response(
        response_id,
        model,
        status="completed",
        output=output,
        usage=usage,
    )
    resp["completed_at"] = completed_at
    data = {
        "type": "response.completed",
        "response": resp,
        "sequence_number": seq,
    }
    return (
        f"event: response.completed\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def format_responses_failed(
    response_id: str,
    model: str,
    error_code: str = "server_error",
    error_message: str = "An internal error occurred",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    reasoning_tokens: int = 0,
    seq: int = 0,
) -> str:
    """response.failed — terminal event when generation encounters an error."""
    failed_at = int(time.time())
    resp = _responses_base_response(response_id, model, status="failed")
    resp["failed_at"] = failed_at
    resp["error"] = {"code": error_code, "message": error_message}
    if input_tokens > 0 or output_tokens > 0:
        usage: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if reasoning_tokens > 0:
            usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        resp["usage"] = usage
    data = {
        "type": "response.failed",
        "response": resp,
        "sequence_number": seq,
    }
    return f"event: response.failed\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_responses_incomplete(
    response_id: str,
    model: str,
    reason: str = "max_output_tokens",
    output: list | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    seq: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> str:
    """response.incomplete — terminal event when generation is interrupted."""
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
    incomplete_at = int(time.time())
    resp = _responses_base_response(
        response_id,
        model,
        status="incomplete",
        output=output or [],
        usage=usage,
    )
    resp["incomplete_at"] = incomplete_at
    resp["incomplete_details"] = {"reason": reason}
    data = {
        "type": "response.incomplete",
        "response": resp,
        "sequence_number": seq,
    }
    return (
        f"event: response.incomplete\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


# ── Anthropic SSE Formatter ──


def format_anthropic_chunk(
    message_id: str,
    model: str,
    delta_text: str,
    event_type: str = "content_block_delta",
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_reason: str = "end_turn",
) -> str:
    """Format a single SSE chunk in Anthropic Messages format.

    Supports event types:
    - content_block_delta: text delta (delta_text carries the text)
    - message_start: initial message with usage (input_tokens/output_tokens)
    - message_delta: final delta with stop_reason and output_tokens
    - ping: keepalive event (Anthropic uses this instead of SSE comments)
    """
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
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    elif event_type == "message_delta":
        # real Anthropic's message_delta always carries BOTH stop_reason and
        # stop_sequence (null unless a custom stop sequence fired). This standalone
        # formatter omitted stop_sequence — the live router builds message_delta inline
        # (with stop_sequence), but keep this spec-correct so a future caller can't ship
        # a malformed event.
        data["delta"] = {"stop_reason": stop_reason, "stop_sequence": None}
        data["usage"] = {"output_tokens": output_tokens}
    elif event_type == "ping":
        # Anthropic keepalive event — used instead of SSE comments
        data = {"type": "ping"}

    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
