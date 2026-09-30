"""Incremental tool-call extraction for streamed model output.

Feeds text chunks through the model's tool-call formats (``tool_format``):
text outside calls streams out as soon as it cannot be the start of a call
marker; from a start marker on, text is held until the format's end marker,
then the call is parsed and emitted as a start (id + name) followed by the
complete call. A span no format can read is released as text. Tool markup
never reaches the content stream, and every call is keyed by a start before
its arguments.

Request constraints apply at the source: a named ``tool_choice`` drops calls
to other functions, ``parallel_tool_calls=False`` keeps only the first call,
and arguments are typed with the request's tool schemas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .tool_arguments import coerce_tool_arguments, tool_schemas
from .tool_format import (
    ToolFormat,
    fallback_formats,
    next_start,
    openai_tools,
    parse_block,
)


class StreamState(Enum):
    TEXT = auto()  # streaming content
    TOOL_CALL = auto()  # inside a call span, holding text
    WHOLE = auto()  # the message may itself be a JSON call


@dataclass
class ToolCallResult:
    id: str
    name: str
    arguments: str  # JSON object string


@dataclass
class StreamOutput:
    """One output: ``text`` to emit as content, a ``tool_call_start`` (id +
    name), a ``tool_call_args_delta``, or a complete ``tool_call``."""

    text: str = ""
    tool_call: ToolCallResult | None = None
    state: StreamState = StreamState.TEXT
    tool_call_start: ToolCallResult | None = None
    tool_call_args_delta: str = ""


class ToolCallStreamer:
    def __init__(
        self,
        formats: Sequence[ToolFormat] | None = None,
        *,
        forced_tool_name: str | None = None,
        allow_parallel: bool = True,
        max_calls: int | None = None,
        tools: Any = None,
        call_id_prefix: str = "call_",
    ) -> None:
        self._formats = tuple(formats) if formats else fallback_formats()
        self._starts = tuple(f.start for f in self._formats if f.start and not f.whole)
        self._whole = any(f.whole for f in self._formats)
        self._tools = openai_tools(tools)
        self._schemas = tool_schemas(tools)
        self._forced = forced_tool_name
        self._allow_parallel = allow_parallel
        self._max_calls = max_calls
        self._prefix = call_id_prefix
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._state = StreamState.WHOLE if self._whole else StreamState.TEXT
        self._counter = 0
        self._accepted = 0

    @property
    def state(self) -> StreamState:
        return self._state

    # ── public API ──

    def process_token(self, token: str) -> list[StreamOutput]:
        if not token:
            return []
        self._buf += token
        return self._drain(final=False)

    def flush(self) -> list[StreamOutput]:
        return self._drain(final=True)

    # ── internals ──

    def _drain(self, final: bool) -> list[StreamOutput]:
        out: list[StreamOutput] = []
        if self._state is StreamState.WHOLE:
            head = self._buf.lstrip()
            if not final and "<|python_tag|>".startswith(head):
                return out  # nothing yet, or a partial python tag
            if head.startswith(("{", "<|python_tag|>")) and not final:
                return out  # the message may be a JSON call: hold it
            if head.startswith(("{", "<|python_tag|>")):
                for fmt in self._formats:
                    if not fmt.whole:
                        continue
                    try:
                        calls = fmt.parse(self._buf, self._tools)
                    except Exception:  # noqa: BLE001
                        continue
                    self._buf = ""
                    for call in calls:
                        self._emit_call(call, out)
                    return out
            self._state = StreamState.TEXT
        while self._buf:
            found = next_start(self._buf, self._formats)
            if found is None:
                keep = 0 if final else self._partial_marker_len()
                text = self._buf[: len(self._buf) - keep]
                if text:
                    out.append(StreamOutput(text=text))
                self._buf = self._buf[len(self._buf) - keep :]
                self._state = StreamState.TEXT
                break
            at, _marker, group = found
            if at:
                out.append(StreamOutput(text=self._buf[:at]))
                self._buf = self._buf[at:]
            self._state = StreamState.TOOL_CALL
            parsed = parse_block(self._buf, 0, group, self._tools, final=final)
            if parsed is None:
                break  # end marker not here yet
            calls, end = parsed
            if calls:
                for call in calls:
                    self._emit_call(call, out)
            else:
                out.append(StreamOutput(text=self._buf[:end]))
            self._buf = self._buf[end:]
            self._state = StreamState.TEXT
        return out

    def _partial_marker_len(self) -> int:
        """Length of the longest buffer suffix that could begin a marker."""
        best = 0
        for marker in self._starts:
            for n in range(min(len(marker) - 1, len(self._buf)), best, -1):
                if self._buf.endswith(marker[:n]):
                    best = n
                    break
        return best

    def _emit_call(self, call: dict, out: list[StreamOutput]) -> None:
        name = call["name"]
        if self._forced is not None and name != self._forced:
            return
        if not self._allow_parallel and self._accepted >= 1:
            return
        if self._max_calls is not None and self._accepted >= self._max_calls:
            return
        self._accepted += 1
        call_id = f"{self._prefix}{self._counter:x}"
        self._counter += 1
        args = coerce_tool_arguments(name, call["arguments"], self._schemas)
        out.append(
            StreamOutput(
                tool_call_start=ToolCallResult(id=call_id, name=name, arguments=""),
                state=StreamState.TOOL_CALL,
            )
        )
        out.append(
            StreamOutput(
                tool_call=ToolCallResult(id=call_id, name=name, arguments=args),
                state=StreamState.TOOL_CALL,
            )
        )
