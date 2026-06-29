from __future__ import annotations

"""Yunshu Tool Call Streamer — incremental tool call detection from streaming tokens.

Processes tokens one at a time from a streaming LLM response, detecting
<tool_call/>...</tool_call/> patterns incrementally. Designed for the
streaming SSE path where we must decide whether each chunk is regular
text or part of a tool call.

Key design:
1. Token buffering — tokens are buffered until we can determine if they
   are a tool call opening tag or regular text.
2. Tag detection — recognizes <tool_call/> and <tool_call\\> opening tags,
   and </tool_call/> closing tags, even when split across chunks.
3. Flush timeout — if buffered text isn't recognized as a tag within
   a configurable number of tokens, it is flushed as regular content.
4. Multiple tool calls — supports multiple sequential tool calls in a
   single response.
"""


import json
import logging
import re
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class StreamFormat(Enum):
    """Detected tool-call output format."""

    XML = auto()  # <tool_call...>...</tool_call...>
    DEEPSEEK_FUNCTION = auto()  # ✿FUNCTION✿ format
    # formats the incremental state machine does NOT understand (Mistral
    # [TOOL_CALLS], Qwen <function=name>, etc.). The non-streaming parse_tool_calls handles
    # them; for streaming we buffer everything from the marker on (emitting nothing) and run
    # the full parser at flush — so the markup never leaks as text and the calls aren't
    # silently dropped (they were: the streamer only knew XML + DeepSeek).
    BUFFER_ALL = auto()


class StreamState(Enum):
    """States for the tool call stream parser."""

    TEXT = auto()  # Normal text output
    TAG_START = auto()  # Potentially inside opening <tool_call...> tag
    TOOL_JSON = auto()  # Inside tool call JSON body
    TAG_END = auto()  # Potentially inside closing </tool_call...> tag


# We need to detect these opening tag variants:
# <tool_call/>   <tool_call\>   <tool_call >   <tool_call\ >
# And the closing tag:
# </tool_call/>   </tool_call\>   </tool_call >   </tool_call >
TOOL_CALL_OPEN = "<tool_call"
TOOL_CALL_CLOSE = "</tool_call"

# DeepSeek alternative format marker
DEEPSEEK_FUNCTION_MARKER = "✿FUNCTION✿"  # ✿FUNCTION✿

# markers for formats the incremental state machine can't stream — detecting any
# of these switches the streamer to BUFFER_ALL (hold + parse-at-flush via parse_tool_calls).
_BUFFER_ALL_MARKERS = (
    "[TOOL_CALLS]",
    "<function=",
    "<｜tool▁calls▁begin｜>",
    # GLM-4.x older block form. The incremental machine has no handler for it,
    # so it leaked raw into delta.content as text; buffer-and-parse-at-flush instead (the
    # GLM parser is now in the flush registry). The newer GLM-4.6/4.7 `<tool_call>name
    # <arg_key>…` form shares the `<tool_call` opener and is handled at the close-tag
    # fallback above, not here.
    "<|tool_call_block_begin|>",
)


@dataclass
class ToolCallResult:
    """A parsed tool call extracted from the stream."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class StreamOutput:
    """Output from the ToolCallStreamer for one chunk.

    At most one output type is set per chunk:
    - 'text': regular text to emit as content
    - 'tool_call': complete tool call (legacy, used by flush)
    - 'tool_call_start': beginning of a tool call (carries id + name)
    - 'tool_call_args_delta': incremental argument fragment for streaming
    """

    text: str = ""
    tool_call: ToolCallResult | None = None
    state: StreamState = StreamState.TEXT
    # Incremental streaming fields — emitted token-by-token while inside
    # a <tool_call...>...</tool_call...> block so the gateway can produce
    # per-token SSE deltas with function.arguments fragments.
    tool_call_start: ToolCallResult | None = None
    tool_call_args_delta: str = ""


class ToolCallStreamer:
    """Incremental tool call detector for streaming LLM output.

    Usage:
        streamer = ToolCallStreamer()
        for token in llm_stream:
            outputs = streamer.process_token(token)
            for out in outputs:
                if out.text:
                    yield text_chunk(out.text)
                elif out.tool_call:
                    yield tool_call_chunk(out.tool_call)

        # Flush any remaining buffered text
        for out in streamer.flush():
            ...

    The streamer buffers tokens until it can determine whether they form
    a <tool_call/> tag or are regular text. If the buffer grows beyond
    `flush_threshold` tokens without matching a tag, it flushes as text.
    """

    def __init__(
        self,
        flush_threshold: int = 20,
        call_id_prefix: str = "call_",
        json_buffer_limit: int = 65536,
        forced_tool_name: str | None = None,
        allow_parallel: bool = True,
        model_name: str | None = None,
    ) -> None:
        self._flush_threshold = flush_threshold
        self._call_id_prefix = call_id_prefix
        self._call_counter = 0
        self._json_buffer_limit = json_buffer_limit
        # model hint for the BUFFER_ALL flush parser (parse_tool_calls accepts
        # None and auto-detects by content, so this is best-effort).
        self._model_name = model_name
        # safety net: when a tool marker is split across the text-emit boundary
        # (e.g. a long preamble before Mistral [TOOL_CALLS] flushes the buffer first),
        # early BUFFER_ALL detection misses it and the markup leaks as text with the call
        # DROPPED — which breaks the agent loop. Track all seen text + whether ANY call was
        # surfaced; at flush, if none was, re-parse the full text and recover the calls.
        self._seen_text = ""
        self._emitted_tool_call = False

        # streaming-side tool_choice / parallel_tool_calls enforcement.
        # The non-streaming path runs _enforce_tool_choice post-generation, but the
        # streaming generators emitted whatever the model produced — so a named
        # tool_choice did NOT filter a call to a different function and
        # parallel_tool_calls=False did NOT cap to one call (OpenAI contract break,
        # only advised by the injected system prompt which the model can ignore).
        # Enforcing at the streamer source fixes every consumer (single + multi-choice
        # streaming) at once. When unconstrained (default) the filter is a pure
        # pass-through, so there is zero behaviour change for normal requests.
        self._forced_tool_name = forced_tool_name
        self._allow_parallel = allow_parallel
        self._tc_accepted = 0  # number of tool calls surfaced so far
        self._tc_suppressing = False  # is the in-flight call being dropped?
        self._tc_in_call = False  # have we seen a start not yet completed?

        self._state = StreamState.TEXT
        self._format: StreamFormat = StreamFormat.XML
        self._buffer = ""
        self._json_buffer = ""  # Accumulated JSON inside tool call
        self._pending_json_text = (
            ""  # Saved JSON when closing tag is split across tokens
        )
        # Incremental streaming state
        self._current_tc_id: str = ""  # ID of the tool call being streamed
        self._current_tc_name: str = ""  # Name of the tool call being streamed
        self._current_tc_start_emitted: bool = (
            False  # Whether we emitted tool_call_start
        )
        self._json_name_parsed: bool = (
            False  # Whether we extracted name from partial JSON
        )
        self._args_emit_offset: int = 0  # Buffer offset where args portion begins
        self._args_emitted_up_to: int = (
            0  # Buffer offset up to which args have been emitted
        )

    @property
    def state(self) -> StreamState:
        return self._state

    def _should_suppress(self, name: str) -> bool:
        """Whether a tool call named ``name`` violates the tool_choice constraint."""
        if self._forced_tool_name is not None and name != self._forced_tool_name:
            return True
        return not self._allow_parallel and self._tc_accepted >= 1

    def _apply_tool_choice(self, results: list[StreamOutput]) -> list[StreamOutput]:
        """Filter streamer outputs to honour the tool_choice / parallel_tool_calls
        constraint. A suppressed call has its start, every args delta, and
        its final tool_call dropped so the consumer never surfaces it. Text passes
        through untouched. No constraint → pure pass-through (no behaviour change)."""
        # note whether the model produced ANY recognized tool call (pre-suppression
        # — a constrained-away call still means no recovery is needed at flush).
        if not self._emitted_tool_call:
            for _o in results:
                if _o.tool_call is not None or _o.tool_call_start is not None:
                    self._emitted_tool_call = True
                    break
        if self._forced_tool_name is None and self._allow_parallel:
            return results
        filtered: list[StreamOutput] = []
        for out in results:
            if out.tool_call_start is not None:
                self._tc_in_call = True
                if self._should_suppress(out.tool_call_start.name):
                    self._tc_suppressing = True
                    continue
                self._tc_suppressing = False
                self._tc_accepted += 1
                filtered.append(out)
            elif out.tool_call_args_delta:
                if self._tc_suppressing:
                    continue
                filtered.append(out)
            elif out.tool_call is not None:
                if self._tc_in_call:
                    # completes a call whose start we already ruled on
                    suppress = self._tc_suppressing
                else:
                    # standalone complete call (e.g. flush with no prior start)
                    suppress = self._should_suppress(out.tool_call.name)
                    if not suppress:
                        self._tc_accepted += 1
                self._tc_suppressing = False
                self._tc_in_call = False
                if suppress:
                    continue
                filtered.append(out)
            else:
                filtered.append(out)
        return filtered

    def _next_call_id(self) -> str:
        call_id = f"{self._call_id_prefix}{self._call_counter:x}"
        self._call_counter += 1
        return call_id

    def _reset_tool_call_state(self) -> None:
        """Reset per-tool-call streaming state (between tool calls)."""
        self._current_tc_id = ""
        self._current_tc_name = ""
        self._current_tc_start_emitted = False
        self._json_name_parsed = False
        self._args_emit_offset = 0
        self._args_emitted_up_to = 0

    @staticmethod
    def _args_value_end(buf: str, start: int) -> int:
        """Return the offset just past the END of the arguments VALUE in ``buf``.

        ``start`` is the offset right after ``"arguments":``. The value is a
        balanced JSON object/array; this returns the offset just past its closing
        brace (quote-aware), or -1 if the value hasn't fully arrived yet (or is a
        primitive we don't bound). Used to stop emitting arg deltas BEFORE the
        outer object's closing '}' and the </tool_call> tag — otherwise those
        leak into the streamed arguments and break JSON parsing client-side.
        """
        n = len(buf)
        i = start
        while i < n and buf[i] in " \t\r\n":
            i += 1
        if i >= n:
            return -1  # value hasn't started yet
        ch0 = buf[i]
        if ch0 in "{[":
            open_ch = ch0
            close_ch = "}" if open_ch == "{" else "]"
            depth = 0
            in_str = False
            esc = False
            while i < n:
                ch = buf[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return i + 1
                i += 1
            return -1
        # PRIMITIVE values must be bounded too — otherwise -1 told the
        # caller "don't cap" and the outer wrapper '}' (and the closing tag) leaked
        # into the streamed args. Strings end at the matching close-quote; numbers/
        # bool/null end at the first , } ] or whitespace.
        if ch0 == '"':
            i += 1
            esc = False
            while i < n:
                ch = buf[i]
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    return i + 1
                i += 1
            return -1  # string not closed yet
        j = i
        while j < n and buf[j] not in ",}] \t\r\n":
            j += 1
        if j >= n:
            return -1  # primitive may still be growing (no terminator yet)
        return j

    def _try_parse_name(self, json_text: str) -> None:
        """Try to extract the function name from partial JSON.

        Uses regex so we can detect the name before the JSON is complete.
        """
        if self._current_tc_name:
            return
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_text)
        if name_match:
            self._current_tc_name = name_match.group(1)

    @staticmethod
    def _find_closing_tag_outside_strings(text: str) -> int:
        """Find TOOL_CALL_CLOSE in *text*, skipping occurrences inside JSON strings.

        JSON strings are delimited by unescaped double quotes. Any occurrence
        of TOOL_CALL_CLOSE that is inside such a quoted region is a false
        positive and must be ignored.

        Returns the index of the first valid occurrence, or -1 if none found.
        """
        search_start = 0
        tag_len = len(TOOL_CALL_CLOSE)
        while search_start <= len(text) - tag_len:
            idx = text.find(TOOL_CALL_CLOSE, search_start)
            if idx == -1:
                return -1
            # Check whether position idx is inside a JSON string by scanning
            # from the beginning and tracking quote state.
            in_string = False
            escape_next = False
            for ci in range(idx):
                ch = text[ci]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
            if not in_string:
                return idx
            # This occurrence is inside a string — skip past it and try again.
            search_start = idx + 1
        return -1

    def process_token(self, token: str) -> list[StreamOutput]:
        """Process a single streaming token, returning zero or more outputs.

        The token is typically a single token from the LLM (could be a
        partial word, whitespace, etc.).

        Returns a list because buffering may produce multiple outputs:
        - If we were buffering and it turns out to be text, we flush
          the buffer as text and continue.
        - If a complete tool call is detected, we emit it.

        Multi-format support: the streamer auto-detects the tool-call
        format from the accumulated text.  Once a known marker is seen
        the appropriate parsing mode is activated.
        """
        # accumulate all seen text (capped) for the flush-time recovery net.
        if len(self._seen_text) < 262144:
            self._seen_text += token

        # ── Multi-format detection (only while still in TEXT state) ──
        if self._state == StreamState.TEXT:
            # Check for DeepSeek ✿FUNCTION✿ marker in accumulated buffer
            if self._format == StreamFormat.XML:
                _combined = self._buffer + token
                if DEEPSEEK_FUNCTION_MARKER in _combined:
                    self._format = StreamFormat.DEEPSEEK_FUNCTION
                    # do NOT pre-assign self._buffer = buffer+token here — the
                    # DEEPSEEK_FUNCTION handler dispatched immediately below ALSO does
                    # `self._buffer += token`, so pre-assigning appended the transition token
                    # TWICE → the doubled text later leaked to the user when the handler
                    # flushed unfenced text. Let the handler own the single append.
                else:
                    # a format the incremental machine can't stream (Mistral
                    # [TOOL_CALLS], Qwen <function=>, …). Emit any legit text BEFORE the
                    # marker, then buffer from the marker on and parse at flush.
                    _ba_idx = min(
                        (
                            _combined.find(_m)
                            for _m in _BUFFER_ALL_MARKERS
                            if _m in _combined
                        ),
                        default=-1,
                    )
                    if _ba_idx != -1:
                        self._format = StreamFormat.BUFFER_ALL
                        _pre = _combined[:_ba_idx]
                        self._buffer = _combined[_ba_idx:]
                        _pre_out = (
                            [StreamOutput(text=_pre, state=self._state)] if _pre else []
                        )
                        return self._apply_tool_choice(_pre_out)

        results: list[StreamOutput] = []

        # ── Dispatch by detected format ──
        if self._format == StreamFormat.BUFFER_ALL:
            # Hold everything; the parser runs at flush(). Emit nothing now.
            self._buffer += token
            return self._apply_tool_choice([])
        if self._format == StreamFormat.DEEPSEEK_FUNCTION:
            results.extend(self._handle_deepseek_function(token))
        elif self._state == StreamState.TEXT:
            results.extend(self._handle_text_state(token))
        elif self._state == StreamState.TAG_START:
            results.extend(self._handle_tag_start_state(token))
        elif self._state == StreamState.TOOL_JSON:
            results.extend(self._handle_tool_json_state(token))
        elif self._state == StreamState.TAG_END:
            results.extend(self._handle_tag_end_state(token))

        return self._apply_tool_choice(results)

    def _handle_text_state(self, token: str) -> list[StreamOutput]:
        """Handle tokens while in normal text mode.

        We accumulate text and check if we see the start of a <tool_call tag.
        If so, switch to TAG_START state. If buffer grows too large without
        a tag, flush as text.
        """
        results: list[StreamOutput] = []
        self._buffer += token

        # Check if buffer contains <tool_call opening tag
        idx = self._buffer.find(TOOL_CALL_OPEN)
        if idx != -1:
            # Check if there's a complete opening tag (ends with > or />)
            after = self._buffer[idx + len(TOOL_CALL_OPEN) :]
            close_idx = after.find(">")
            if close_idx != -1:
                # Validate: char after <tool_call must be a valid delimiter
                # (space, >, /, \) to reject e.g. <tool_calls>
                first_after = after[0] if after else ">"
                if first_after not in (">", " ", "/", "\\"):
                    # Not a valid tool_call tag — fall through to flush check
                    pass
                else:
                    # Emit any text before the tag
                    before = self._buffer[:idx]
                    if before:
                        results.append(StreamOutput(text=before, state=self._state))

                    # Check if this is actually a closing tag </tool_call...>
                    if self._buffer[idx:].startswith(TOOL_CALL_CLOSE):
                        # It's a closing tag appearing without opening — treat as text
                        closing_text = self._buffer[idx:]
                        self._state = StreamState.TEXT
                        self._buffer = ""
                        results.append(
                            StreamOutput(
                                text=closing_text,
                                state=self._state,
                            )
                        )
                        return results

                    # Move to JSON mode
                    self._state = StreamState.TOOL_JSON
                    self._json_buffer = ""
                    self._buffer = ""
                    self._reset_tool_call_state()

                    # If there's content after the closing >, process it in TOOL_JSON
                    remaining = after[close_idx + 1 :]
                    if remaining:
                        results.extend(self._handle_tool_json_state(remaining))

                    return results
            else:
                # Partial tag — might be building up <tool_call...>
                # Check if what we have so far could still become a tag
                partial = self._buffer[idx:]
                if TOOL_CALL_OPEN.startswith(partial) or partial.startswith(
                    TOOL_CALL_OPEN
                ):
                    # Could still become a tag. But if buffer before the
                    # partial is large, we should flush it.
                    before = self._buffer[:idx]
                    if len(before) > 0 and len(partial) >= len(TOOL_CALL_OPEN):
                        # We have a full <tool_call prefix but no > yet
                        # Switch to TAG_START to wait for >
                        self._state = StreamState.TAG_START
                        if before:
                            results.append(
                                StreamOutput(text=before, state=StreamState.TEXT)
                            )
                        self._buffer = partial
                        return results
                    elif len(before) >= self._flush_threshold:
                        # Too much text before potential tag — flush
                        results.append(StreamOutput(text=before, state=self._state))
                        self._buffer = partial
                        return results
                    # else: keep buffering, partial tag might complete
                    return results
                else:
                    # Not a tag prefix — treat as text
                    pass

        # No tag found — check flush threshold
        if len(self._buffer) >= self._flush_threshold:
            # Check if buffer tail could still become a tag
            safe_len = len(self._buffer)
            for i in range(
                len(self._buffer) - 1,
                max(-1, len(self._buffer) - len(TOOL_CALL_OPEN) - 1),
                -1,
            ):
                tail = self._buffer[i:]
                if TOOL_CALL_OPEN.startswith(tail):
                    safe_len = i
                    break

            if safe_len > 0:
                text = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                results.append(StreamOutput(text=text, state=self._state))

        return results

    def _handle_tag_start_state(self, token: str) -> list[StreamOutput]:
        """Waiting for > to complete the opening <tool_call...> tag."""
        results: list[StreamOutput] = []
        self._buffer += token

        close_idx = self._buffer.find(">")
        if close_idx != -1:
            # validate the char immediately after the "<tool_call" prefix is a
            # real tag delimiter (space, >, /, \). The inline detector in _handle_text_state
            # (L322) rejects non-tags like <tool_calls> / <tool_call_id> / <tool_callback>,
            # but when such a token arrives SPLIT across deltas it reaches TAG_START instead,
            # which previously accepted ANY "<tool_call…>" and consumed the text as a bogus
            # tool call → the surrounding content was silently dropped from the stream
            # (verified: ['Use ','<tool','_calls','>',' here'] → 'Use  here'). The buffer here
            # always starts with "<tool_call" and ">" sits past it, so the delimiter char is
            # at len(TOOL_CALL_OPEN).
            first_after = (
                self._buffer[len(TOOL_CALL_OPEN)]
                if len(self._buffer) > len(TOOL_CALL_OPEN)
                else ">"
            )
            if first_after not in (">", " ", "/", "\\"):
                # Not a tool_call tag — revert to TEXT and re-scan via the unified text
                # logic (it flushes the buffer as plain content; nothing is lost).
                self._state = StreamState.TEXT
                return self._handle_text_state("")
            # Complete opening tag found — entering JSON mode
            self._state = StreamState.TOOL_JSON
            self._json_buffer = ""
            self._buffer = ""
            self._reset_tool_call_state()
            return results

        # If buffer gets too long without >, it's not a valid tag
        if len(self._buffer) > self._flush_threshold:
            text = self._buffer[: self._flush_threshold]
            self._buffer = self._buffer[self._flush_threshold :]
            self._state = StreamState.TEXT
            results.append(StreamOutput(text=text, state=self._state))
            # Re-process remaining buffer in TEXT state
            if self._buffer:
                results.extend(self._handle_text_state(""))

        return results

    def _handle_tool_json_state(self, token: str) -> list[StreamOutput]:
        """Accumulating JSON body inside <tool_call...>...</tool_call...>.

        For incremental streaming, we emit argument deltas token-by-token
        so the gateway can produce per-token SSE chunks with
        delta.tool_calls[i].function.arguments fragments.

        The flow is:
        1. As JSON accumulates, try to extract the "name" field early via
           regex. Once found, emit a tool_call_start output (id + name).
        2. Continue buffering until the "arguments": boundary is found.
           Once found, set _args_emit_offset so subsequent tokens are
           emitted as tool_call_args_delta (only the arguments value).
        3. When the closing </tool_call...> tag is found, emit a final
           tool_call (complete ToolCallResult) with the full arguments.
        """
        results: list[StreamOutput] = []

        # Check for closing tag *before* appending the new token, so we can
        # determine how much of the token is argument text vs. closing tag.
        # Use string-aware search to avoid false positives when TOOL_CALL_CLOSE
        # appears inside a JSON string value (e.g. {"content": "</tool_call"}).
        combined = self._json_buffer + token
        close_idx = self._find_closing_tag_outside_strings(combined)

        if close_idx != -1:
            # ── Closing tag found ──
            json_text = combined[:close_idx].strip()
            after_close = combined[close_idx + len(TOOL_CALL_CLOSE) :]

            # If we haven't emitted start yet, do so now
            if not self._current_tc_start_emitted:
                self._try_parse_name(json_text)
                if self._current_tc_name:
                    self._current_tc_id = self._next_call_id()
                    results.append(
                        StreamOutput(
                            tool_call_start=ToolCallResult(
                                id=self._current_tc_id,
                                name=self._current_tc_name,
                                arguments="",
                            ),
                            state=StreamState.TOOL_JSON,
                        )
                    )
                    self._current_tc_start_emitted = True

            # Emit any remaining args text that hasn't been emitted yet
            # (between what was already emitted and the closing tag)
            if self._current_tc_start_emitted and self._args_emit_offset > 0:
                unemitted_start = max(self._args_emit_offset, self._args_emitted_up_to)
                # Cap at the balanced END of the arguments value so the outer
                # object's closing '}' (sitting between the value and the tag) is
                # not leaked into the streamed arguments.
                _val_end = self._args_value_end(combined, self._args_emit_offset)
                _end = close_idx if _val_end < 0 else min(close_idx, _val_end)
                remaining_args = combined[unemitted_start:_end]
                if remaining_args:
                    results.append(
                        StreamOutput(
                            tool_call_args_delta=remaining_args,
                            state=StreamState.TOOL_JSON,
                        )
                    )

            # Find the > of the closing tag
            gt_idx = after_close.find(">")
            if gt_idx != -1:
                remaining = after_close[gt_idx + 1 :]

                # Parse the complete JSON for the full tool_call output
                tool_call = self._parse_tool_json(json_text)
                if tool_call is None and "<arg_key>" in json_text:
                    # GLM-4.6/4.7 emit <tool_call>name<arg_key>k</arg_key>
                    # <arg_value>v</arg_value></tool_call> — key/value PAIRS, not a JSON
                    # body — so _parse_tool_json (JSON-only) returns None and the whole
                    # tool call was silently DROPPED (the streamer entered TOOL_JSON via
                    # the shared `<tool_call` opener, buffered without leaking, then reset
                    # to TEXT here with nothing emitted). Fall back to the model-aware
                    # GLM parser on the reconstructed markup.
                    try:
                        from .tool_call_parser import parse_tool_calls

                        _glm = parse_tool_calls(
                            f"{TOOL_CALL_OPEN}>{json_text}{TOOL_CALL_CLOSE}>",
                            model_name=self._model_name or "glm",
                        )
                    except Exception:
                        _glm = []
                    if _glm:
                        _c = _glm[0]
                        tool_call = ToolCallResult(
                            id=self._current_tc_id
                            if self._current_tc_start_emitted
                            else self._next_call_id(),
                            name=getattr(_c, "name", "") or "",
                            arguments=getattr(_c, "arguments", "") or "{}",
                        )
                if tool_call:
                    # Use the id/name we already assigned during streaming
                    if self._current_tc_start_emitted:
                        tool_call.id = self._current_tc_id
                    results.append(
                        StreamOutput(
                            tool_call=tool_call,
                            state=StreamState.TOOL_JSON,
                        )
                    )

                self._reset_tool_call_state()
                self._state = StreamState.TEXT
                self._json_buffer = ""
                self._buffer = remaining
                if remaining:
                    results.extend(self._handle_text_state(""))
            else:
                # Partial closing tag (</tool_call without >)
                self._pending_json_text = json_text
                self._buffer = after_close
                self._json_buffer = ""
                self._state = StreamState.TAG_END

            return results

        # ── No closing tag yet — accumulate and emit incremental deltas ──

        # Guard against unbounded growth
        if len(combined) > self._json_buffer_limit:
            text = combined
            self._json_buffer = ""
            self._reset_tool_call_state()
            self._state = StreamState.TEXT
            results.append(StreamOutput(text=text, state=self._state))
            return results

        len(self._json_buffer)
        self._json_buffer = combined

        # Phase 1: Try to extract name from accumulated JSON if not done yet
        if not self._json_name_parsed:
            self._try_parse_name(self._json_buffer)
            if self._current_tc_name:
                self._json_name_parsed = True
                self._current_tc_id = self._next_call_id()
                # Emit tool_call_start so the gateway can send the initial
                # SSE chunk with id + name (arguments="").
                results.append(
                    StreamOutput(
                        tool_call_start=ToolCallResult(
                            id=self._current_tc_id,
                            name=self._current_tc_name,
                            arguments="",
                        ),
                        state=StreamState.TOOL_JSON,
                    )
                )
                self._current_tc_start_emitted = True

        # Phase 2: Look for "arguments": boundary to set emit offset
        if self._current_tc_start_emitted and self._args_emit_offset == 0:
            # Look for the "arguments" key boundary in the accumulated buffer.
            # We want the offset of the value *after* "arguments": (or "parameters":)
            for key in ('"arguments"', '"parameters"'):
                args_match = re.search(
                    key + r"\s*:\s*",
                    self._json_buffer,
                )
                if args_match:
                    self._args_emit_offset = args_match.end()
                    break

        # Phase 3: Emit argument delta for new text beyond offset.
        # We must be careful not to emit a partial closing tag that might
        # be forming at the tail of the buffer (e.g. "</tool_cal" being
        # built char-by-char).  So we scan backwards from the end and
        # hold back any tail that could still become TOOL_CALL_CLOSE.
        if self._current_tc_start_emitted and self._args_emit_offset > 0:
            emit_start = max(self._args_emit_offset, self._args_emitted_up_to)
            if emit_start < len(self._json_buffer):
                raw = self._json_buffer[emit_start:]
                # Find the safe prefix: everything before a potential
                # partial TOOL_CALL_CLOSE at the tail.
                safe_end = len(raw)
                for si in range(
                    len(raw) - 1, max(-1, len(raw) - len(TOOL_CALL_CLOSE) - 1), -1
                ):
                    tail = raw[si:]
                    if TOOL_CALL_CLOSE.startswith(tail):
                        safe_end = si
                        break
                # Never emit past the balanced end of the arguments value — the
                # trailing outer '}' belongs to the wrapper object, not the args.
                _val_end = self._args_value_end(
                    self._json_buffer, self._args_emit_offset
                )
                if _val_end >= 0:
                    safe_end = min(safe_end, max(0, _val_end - emit_start))
                if safe_end > 0:
                    delta = raw[:safe_end]
                    results.append(
                        StreamOutput(
                            tool_call_args_delta=delta,
                            state=StreamState.TOOL_JSON,
                        )
                    )
                    self._args_emitted_up_to = emit_start + safe_end

        return results

    def _handle_tag_end_state(self, token: str) -> list[StreamOutput]:
        """Waiting for > to complete the closing </tool_call...> tag.

        This state is reached when we detect </tool_call but haven't
        seen the closing > yet. _pending_json_text holds the JSON body
        that was accumulated before the partial closing tag was detected.
        """
        results: list[StreamOutput] = []
        self._buffer += token

        close_idx = self._buffer.find(">")
        if close_idx != -1:
            remaining = self._buffer[close_idx + 1 :]
            self._buffer = ""

            # Parse the saved JSON text now that we have the complete closing tag
            json_text = getattr(self, "_pending_json_text", "")
            self._pending_json_text = ""
            self._json_buffer = ""

            tool_call = self._parse_tool_json(json_text) if json_text else None
            if tool_call:
                # Use the id/name we already assigned during streaming
                if self._current_tc_start_emitted:
                    tool_call.id = self._current_tc_id
                results.append(
                    StreamOutput(
                        tool_call=tool_call,
                        state=StreamState.TAG_END,
                    )
                )

            self._reset_tool_call_state()
            self._state = StreamState.TEXT
            if remaining:
                results.extend(self._handle_text_state(remaining))
        elif len(self._buffer) > self._flush_threshold:
            # Not a valid closing tag — emit saved JSON + partial tag as text
            text = (
                getattr(self, "_pending_json_text", "") + TOOL_CALL_CLOSE + self._buffer
            )
            self._pending_json_text = ""
            self._buffer = ""
            self._state = StreamState.TEXT
            self._reset_tool_call_state()
            results.append(StreamOutput(text=text, state=self._state))

        return results

    # ── DeepSeek ✿FUNCTION✿ handler ──

    def _handle_deepseek_function(self, token: str) -> list[StreamOutput]:
        """Handle DeepSeek ✿FUNCTION✿ format tool calls.

        Expected format:
            ✿FUNCTION✿: function_name
            ```json
            {"arg": "value"}
            ```
            ✿RESULT✿

        The buffer accumulates everything.  Once we see the closing
        ``` after the JSON block we extract the function name and args.
        """
        results: list[StreamOutput] = []
        self._buffer += token

        # Look for completed JSON block: ```json\n{...}\n```
        json_fence_start = self._buffer.find("```json")
        if json_fence_start == -1:
            json_fence_start = self._buffer.find("```")
        if json_fence_start == -1:
            # Still accumulating — check flush threshold on text before marker
            marker_idx = self._buffer.find(DEEPSEEK_FUNCTION_MARKER)
            if marker_idx > self._flush_threshold:
                # Flush text before the marker
                text = self._buffer[:marker_idx]
                self._buffer = self._buffer[marker_idx:]
                results.append(StreamOutput(text=text, state=self._state))
            return results

        # Extract function name from between marker and ```json
        marker_end = self._buffer.find(DEEPSEEK_FUNCTION_MARKER) + len(
            DEEPSEEK_FUNCTION_MARKER
        )
        header = self._buffer[marker_end:json_fence_start].strip()
        # Header is like ": function_name" or just "function_name"
        name = header.lstrip(": \n\r\t")

        # Find the JSON body between ```json and closing ```
        fence_after = json_fence_start
        # skip past ```json or just ```
        nl_after_fence = self._buffer.find("\n", fence_after)
        if nl_after_fence == -1:
            return results
        json_start = nl_after_fence + 1

        # Find closing ```
        close_fence = self._buffer.find("```", json_start)
        if close_fence == -1:
            # JSON body still accumulating — enforce buffer limit to prevent
            # unbounded growth when the model hallucinates an opening fence.
            if len(self._buffer) > self._json_buffer_limit:
                logger.warning(
                    "DeepSeek handler buffer exceeded limit (%d bytes), flushing",
                    self._json_buffer_limit,
                )
                text = self._buffer
                self._buffer = ""
                results.append(StreamOutput(text=text, state=self._state))
            return results

        json_body = self._buffer[json_start:close_fence].strip()

        # Parse the JSON
        try:
            args = json.loads(json_body)
            args_str = json.dumps(args, ensure_ascii=False)
        except json.JSONDecodeError:
            # Attempt basic repair: add missing closing braces
            repaired = json_body
            open_curly = repaired.count("{") - repaired.count("}")
            if open_curly > 0:
                repaired += "}" * open_curly
            try:
                json.loads(repaired)
                args_str = repaired
            except (json.JSONDecodeError, ValueError):
                args_str = "{}"

        if name:
            results.append(
                StreamOutput(
                    tool_call=ToolCallResult(
                        id=self._next_call_id(),
                        name=name,
                        arguments=args_str,
                    ),
                    state=self._state,
                )
            )

        # Consume processed portion; keep any remaining text
        remaining = self._buffer[close_fence + 3 :]
        self._buffer = ""

        # Re-process remaining text in TEXT state for subsequent calls
        if remaining:
            # Stay in DeepSeek mode but process remaining through text handling
            self._state = StreamState.TEXT
            for out in self._handle_text_state(remaining):
                results.append(out)
            # If format was already detected, stay in DeepSeek mode
            self._format = StreamFormat.DEEPSEEK_FUNCTION

        return results

    def _parse_tool_json(self, json_text: str) -> ToolCallResult | None:
        """Parse JSON from inside a <tool_call/>...</tool_call/> block.

        Tries to parse the JSON and extract name + arguments.
        Also handles cases where the JSON is incomplete (truncated output).
        """
        if not json_text.strip():
            return None

        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and "name" in data:
                name = data["name"]
                arguments = data.get("arguments") or data.get("parameters") or {}
                if isinstance(arguments, str):
                    # Validate that string arguments are valid JSON
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        # Keep as-is for the caller to handle, don't silently drop
                        args_str = arguments
                        return ToolCallResult(
                            id=self._next_call_id(),
                            name=name,
                            arguments=args_str,
                        )
                args_str = json.dumps(arguments, ensure_ascii=False)
                return ToolCallResult(
                    id=self._next_call_id(),
                    name=name,
                    arguments=args_str,
                )
        except json.JSONDecodeError:
            # Try to extract name/arguments with a looser pattern
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_text)
            if name_match:
                name = name_match.group(1)
                # Extract arguments using brace-counting to handle nested JSON
                args_match = re.search(r'"arguments"\s*:\s*\{', json_text, re.DOTALL)
                if args_match:
                    brace_start = args_match.end() - 1
                    depth = 0
                    args_str = "{}"
                    in_string = False
                    escape_next = False
                    for ci in range(brace_start, len(json_text)):
                        ch = json_text[ci]
                        if escape_next:
                            escape_next = False
                            continue
                        if ch == "\\" and in_string:
                            escape_next = True
                            continue
                        if ch == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                args_str = json_text[brace_start : ci + 1]
                                break
                else:
                    args_str = "{}"
                return ToolCallResult(
                    id=self._next_call_id(),
                    name=name,
                    arguments=args_str,
                )

        return None

    def flush(self) -> list[StreamOutput]:
        """Flush any remaining buffered content.

        Call this at the end of the stream to emit any remaining text
        or incomplete tool call data.
        """
        results: list[StreamOutput] = []

        # BUFFER_ALL format — run the full non-streaming parser on the held buffer.
        if self._format == StreamFormat.BUFFER_ALL:
            _buf = self._buffer
            self._buffer = ""
            try:
                from .tool_call_parser import parse_tool_calls

                _calls = parse_tool_calls(_buf, model_name=self._model_name)
            except Exception:
                _calls = []
            if _calls:
                # The buffer began AT the tool marker, so it is pure tool markup — emit the
                # parsed calls (residual cleaning lives in the gateway layer; not imported
                # here to avoid an engine→gateway dependency).
                for _c in _calls:
                    _name = getattr(_c, "name", None)
                    _args = getattr(_c, "arguments", None)  # already a JSON string
                    results.append(
                        StreamOutput(
                            tool_call=ToolCallResult(
                                id=self._next_call_id(),
                                name=_name or "",
                                arguments=_args or "{}",
                            ),
                            state=StreamState.TEXT,
                        )
                    )
            else:
                # No parseable calls — the marker was a false positive; emit as text.
                results.append(StreamOutput(text=_buf, state=StreamState.TEXT))
            self._reset_tool_call_state()
            self._state = StreamState.TEXT
            self._format = StreamFormat.XML
            return self._apply_tool_choice(results)

        if self._state == StreamState.TOOL_JSON and self._json_buffer.strip():
            # We have accumulated JSON but never saw closing tag
            # Try to parse it anyway (truncated output)
            tool_call = self._parse_tool_json(self._json_buffer.strip())
            if tool_call:
                # Use the id/name we already assigned during streaming
                if self._current_tc_start_emitted:
                    tool_call.id = self._current_tc_id
                results.append(
                    StreamOutput(
                        tool_call=tool_call,
                        state=self._state,
                    )
                )
            else:
                # Can't parse — emit as text
                results.append(
                    StreamOutput(
                        text=self._json_buffer,
                        state=self._state,
                    )
                )
            self._json_buffer = ""
        elif self._state == StreamState.TAG_START:
            # Was waiting for > but stream ended — buffer already contains the partial tag
            text = self._buffer
            results.append(StreamOutput(text=text, state=self._state))
            self._buffer = ""
        elif self._state == StreamState.TAG_END:
            # The stream ended with a partial closing tag (</tool_call without >).
            # Try to parse the pending JSON as a complete tool call before
            # falling back to emitting as text — the closing '>' is cosmetic
            # and the JSON body is likely complete.
            json_text = getattr(self, "_pending_json_text", "")
            self._pending_json_text = ""
            if json_text.strip():
                tool_call = self._parse_tool_json(json_text.strip())
                if tool_call:
                    if self._current_tc_start_emitted:
                        tool_call.id = self._current_tc_id
                    results.append(
                        StreamOutput(
                            tool_call=tool_call,
                            state=self._state,
                        )
                    )
                    self._buffer = ""
                else:
                    # Can't parse — emit accumulated content as text
                    text = json_text + TOOL_CALL_CLOSE + self._buffer
                    results.append(StreamOutput(text=text, state=self._state))
                    self._buffer = ""
            else:
                text = TOOL_CALL_CLOSE + self._buffer
                results.append(StreamOutput(text=text, state=self._state))
                self._buffer = ""

        if self._buffer:
            results.append(StreamOutput(text=self._buffer, state=self._state))
            self._buffer = ""

        # safety net: if the stream surfaced NO tool call (incl. via the results
        # above) but the full text contains markup the non-streaming parser recognizes
        # (e.g. a Mistral [TOOL_CALLS] whose marker was split across an emit boundary so
        # early BUFFER_ALL detection missed it), recover the calls now so the agent loop
        # isn't broken by a silently-dropped tool call.
        _net_emitted = self._emitted_tool_call or any(
            o.tool_call is not None or o.tool_call_start is not None for o in results
        )
        # Only recover when a STRUCTURAL marker is present — never via parse_tool_calls'
        # loose direct-JSON fallback, which would turn plain content like {"name": "Alice"}
        # into a bogus tool call.
        _has_marker = any(
            _m in self._seen_text
            for _m in (TOOL_CALL_OPEN, DEEPSEEK_FUNCTION_MARKER, *_BUFFER_ALL_MARKERS)
        )
        if not _net_emitted and _has_marker and self._seen_text:
            try:
                from .tool_call_parser import parse_tool_calls

                _recovered = parse_tool_calls(
                    self._seen_text, model_name=self._model_name
                )
            except Exception:
                _recovered = []
            for _c in _recovered:
                results.append(
                    StreamOutput(
                        tool_call=ToolCallResult(
                            id=self._next_call_id(),
                            name=getattr(_c, "name", "") or "",
                            arguments=getattr(_c, "arguments", "{}") or "{}",
                        ),
                        state=StreamState.TEXT,
                    )
                )

        self._reset_tool_call_state()
        self._state = StreamState.TEXT
        return self._apply_tool_choice(results)

    def reset(self) -> None:
        """Reset streamer state for reuse."""
        self._state = StreamState.TEXT
        self._format = StreamFormat.XML
        self._buffer = ""
        self._json_buffer = ""
        self._pending_json_text = ""
        self._reset_tool_call_state()
        # reset tool_choice enforcement counters too (per-request reuse).
        self._tc_accepted = 0
        self._tc_suppressing = False
        self._tc_in_call = False
        # reset the recovery-net state.
        self._seen_text = ""
        self._emitted_tool_call = False
