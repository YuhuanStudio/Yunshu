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
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class StreamState(Enum):
    """States for the tool call stream parser."""
    TEXT = auto()        # Normal text output
    TAG_START = auto()   # Potentially inside opening <tool_call...> tag
    TOOL_JSON = auto()   # Inside tool call JSON body
    TAG_END = auto()     # Potentially inside closing </tool_call...> tag


# We need to detect these opening tag variants:
#   <tool_call/>   <tool_call\>   <tool_call >   <tool_call\ >
# And the closing tag:
#   </tool_call/>   </tool_call\>   </tool_call >   </tool_call >
TOOL_CALL_OPEN = "<tool_call"
TOOL_CALL_CLOSE = "</tool_call"


@dataclass
class ToolCallResult:
    """A parsed tool call extracted from the stream."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class StreamOutput:
    """Output from the ToolCallStreamer for one chunk.

    At most one of 'text' or 'tool_call' is set per chunk.
    'text' may be empty string if nothing to emit.
    'tool_call' is set when a complete tool call has been parsed.
    """
    text: str = ""
    tool_call: Optional[ToolCallResult] = None
    state: StreamState = StreamState.TEXT


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
    ) -> None:
        self._flush_threshold = flush_threshold
        self._call_id_prefix = call_id_prefix
        self._call_counter = 0
        self._json_buffer_limit = json_buffer_limit

        self._state = StreamState.TEXT
        self._buffer = ""
        self._json_buffer = ""  # Accumulated JSON inside tool call
        self._pending_json_text = ""  # Saved JSON when closing tag is split across tokens

    @property
    def state(self) -> StreamState:
        return self._state

    def _next_call_id(self) -> str:
        call_id = f"{self._call_id_prefix}{self._call_counter:x}"
        self._call_counter += 1
        return call_id

    def process_token(self, token: str) -> list[StreamOutput]:
        """Process a single streaming token, returning zero or more outputs.

        The token is typically a single token from the LLM (could be a
        partial word, whitespace, etc.).

        Returns a list because buffering may produce multiple outputs:
        - If we were buffering and it turns out to be text, we flush
          the buffer as text and continue.
        - If a complete tool call is detected, we emit it.
        """
        results: list[StreamOutput] = []

        if self._state == StreamState.TEXT:
            results.extend(self._handle_text_state(token))
        elif self._state == StreamState.TAG_START:
            results.extend(self._handle_tag_start_state(token))
        elif self._state == StreamState.TOOL_JSON:
            results.extend(self._handle_tool_json_state(token))
        elif self._state == StreamState.TAG_END:
            results.extend(self._handle_tag_end_state(token))

        return results

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
            after = self._buffer[idx + len(TOOL_CALL_OPEN):]
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
                        results.append(StreamOutput(
                            text=closing_text,
                            state=self._state,
                        ))
                        return results

                    # Move to JSON mode
                    self._state = StreamState.TOOL_JSON
                    self._json_buffer = ""
                    self._buffer = ""

                    # If there's content after the closing >, process it in TOOL_JSON
                    remaining = after[close_idx + 1:]
                    if remaining:
                        results.extend(self._handle_tool_json_state(remaining))

                    return results
            else:
                # Partial tag — might be building up <tool_call...>
                # Check if what we have so far could still become a tag
                partial = self._buffer[idx:]
                if TOOL_CALL_OPEN.startswith(partial) or partial.startswith(TOOL_CALL_OPEN):
                    # Could still become a tag. But if buffer before the
                    # partial is large, we should flush it.
                    before = self._buffer[:idx]
                    if len(before) > 0 and len(partial) >= len(TOOL_CALL_OPEN):
                        # We have a full <tool_call prefix but no > yet
                        # Switch to TAG_START to wait for >
                        self._state = StreamState.TAG_START
                        if before:
                            results.append(StreamOutput(text=before, state=StreamState.TEXT))
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
            for i in range(len(self._buffer) - 1, max(-1, len(self._buffer) - len(TOOL_CALL_OPEN) - 1), -1):
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
            # Complete opening tag found
            self._state = StreamState.TOOL_JSON
            self._json_buffer = ""
            self._buffer = ""
            return results

        # If buffer gets too long without >, it's not a valid tag
        if len(self._buffer) > self._flush_threshold:
            text = self._buffer[:self._flush_threshold]
            self._buffer = self._buffer[self._flush_threshold:]
            self._state = StreamState.TEXT
            results.append(StreamOutput(text=text, state=self._state))
            # Re-process remaining buffer in TEXT state
            if self._buffer:
                results.extend(self._handle_text_state(""))

        return results

    def _handle_tool_json_state(self, token: str) -> list[StreamOutput]:
        """Accumulating JSON body inside <tool_call...>...</tool_call...>.

        We look for the closing </tool_call tag. When found, we parse
        the accumulated JSON and emit a ToolCallResult.
        """
        results: list[StreamOutput] = []
        self._json_buffer += token

        # Guard against unbounded growth — if no closing tag ever arrives,
        # force-emit what we have as text once the buffer exceeds the limit.
        if len(self._json_buffer) > self._json_buffer_limit:
            text = self._json_buffer
            self._json_buffer = ""
            self._state = StreamState.TEXT
            results.append(StreamOutput(text=text, state=self._state))
            return results

        # Check for closing tag
        close_idx = self._json_buffer.find(TOOL_CALL_CLOSE)
        if close_idx != -1:
            # Found closing tag — extract JSON
            json_text = self._json_buffer[:close_idx].strip()
            after_close = self._json_buffer[close_idx + len(TOOL_CALL_CLOSE):]

            # Find the > of the closing tag
            gt_idx = after_close.find(">")
            if gt_idx != -1:
                remaining = after_close[gt_idx + 1:]
                self._json_buffer = ""

                # Parse the JSON
                tool_call = self._parse_tool_json(json_text)
                if tool_call:
                    results.append(StreamOutput(
                        tool_call=tool_call,
                        state=StreamState.TOOL_JSON,
                    ))

                self._state = StreamState.TEXT
                self._buffer = remaining
                if remaining:
                    results.extend(self._handle_text_state(""))
            else:
                # Partial closing tag (</tool_call without >) — switch to
                # TAG_END state so the closing tag completion is tracked
                # separately from the JSON body.  Save the completed JSON
                # text so it can be parsed once the > arrives.
                self._pending_json_text = json_text
                self._buffer = after_close  # partial closing tag remainder
                self._state = StreamState.TAG_END

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
            remaining = self._buffer[close_idx + 1:]
            self._buffer = ""

            # Parse the saved JSON text now that we have the complete closing tag
            json_text = getattr(self, '_pending_json_text', '')
            self._pending_json_text = ''
            self._json_buffer = ''

            tool_call = self._parse_tool_json(json_text) if json_text else None
            if tool_call:
                results.append(StreamOutput(
                    tool_call=tool_call,
                    state=StreamState.TAG_END,
                ))

            self._state = StreamState.TEXT
            if remaining:
                results.extend(self._handle_text_state(remaining))
        elif len(self._buffer) > self._flush_threshold:
            # Not a valid closing tag — emit saved JSON + partial tag as text
            text = getattr(self, '_pending_json_text', '') + TOOL_CALL_CLOSE + self._buffer
            self._pending_json_text = ''
            self._buffer = ""
            self._state = StreamState.TEXT
            results.append(StreamOutput(text=text, state=self._state))

        return results

    def _parse_tool_json(self, json_text: str) -> Optional[ToolCallResult]:
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
                arguments = data.get("arguments", data.get("parameters", {}))
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
                        if ch == '\\' and in_string:
                            escape_next = True
                            continue
                        if ch == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                args_str = json_text[brace_start:ci + 1]
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

        if self._state == StreamState.TOOL_JSON and self._json_buffer.strip():
            # We have accumulated JSON but never saw closing tag
            # Try to parse it anyway (truncated output)
            tool_call = self._parse_tool_json(self._json_buffer.strip())
            if tool_call:
                results.append(StreamOutput(
                    tool_call=tool_call,
                    state=self._state,
                ))
            else:
                # Can't parse — emit as text
                results.append(StreamOutput(
                    text=self._json_buffer,
                    state=self._state,
                ))
            self._json_buffer = ""
        elif self._state == StreamState.TAG_START:
            # Was waiting for > but stream ended — buffer already contains the partial tag
            text = self._buffer
            results.append(StreamOutput(text=text, state=self._state))
            self._buffer = ""
        elif self._state == StreamState.TAG_END:
            # Must include the pending JSON text that was accumulated before
            # the partial closing tag was detected, otherwise the JSON body
            # is silently dropped.
            json_text = getattr(self, '_pending_json_text', '')
            self._pending_json_text = ''
            text = json_text + TOOL_CALL_CLOSE + self._buffer
            results.append(StreamOutput(text=text, state=self._state))
            self._buffer = ""

        if self._buffer:
            results.append(StreamOutput(text=self._buffer, state=self._state))
            self._buffer = ""

        self._state = StreamState.TEXT
        return results

    def reset(self) -> None:
        """Reset streamer state for reuse."""
        self._state = StreamState.TEXT
        self._buffer = ""
        self._json_buffer = ""
        self._pending_json_text = ""
