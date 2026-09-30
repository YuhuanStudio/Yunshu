# Upstream (inspired): Blaizzy/mlx-vlm (MIT) mlx_vlm/tools/registry.py @ v0.7.3
# Upstream (inspired): ml-explore/mlx-lm (MIT) mlx_lm/tool_parsers @ v0.31.3
"""Tool-call formats: which one a model speaks, and how to read it.

A model's tool-call format is a property of the model, read from its chat
template the way upstream does (``mlx_vlm.tools.registry``: template markers →
parser module). Each format is a :class:`ToolFormat` — the start/end markers
that delimit one call and a parser for the text between them. Upstream parser
modules (``mlx_vlm.tools.parsers``, then ``mlx_lm.tool_parsers``) back every
format they cover; Yunshu adds only what upstream lacks:

- ``yunshu_json`` — the ``<tool_call>{"name", "arguments"}</tool_call>`` form
  Yunshu's injected tool prompt asks for (models without a template tool
  format, and forced ``tool_choice``). Tolerates ``parameters`` for
  ``arguments`` and string-encoded arguments.
- ``deepseek`` — DeepSeek V3/R1 ``<｜tool▁calls▁begin｜>`` blocks.
- ``json_message`` — a message that is nothing but a JSON call object
  (``{"name", "arguments" | "parameters"}``, optionally after Llama's
  ``<|python_tag|>``): Llama 3.x's native form, and what models without a
  marker format (Qwen3-Omni) emit when asked to call a tool.

:func:`tool_formats` gives an engine's formats in the order to try them (its
native format, then ``yunshu_json``, then ``json_message``);
:func:`parse_tool_output` extracts every
call and returns the text with the call spans removed; ``ToolCallStreamer``
(``tool_call_streamer.py``) does the same incrementally. Arguments are always a
JSON object string, typed with the request's tool schemas.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .tool_arguments import coerce_tool_calls

logger = logging.getLogger(__name__)

# A parsed call: {"name": str, "arguments": JSON object string}.
Call = dict[str, str]


@dataclass(frozen=True)
class ToolFormat:
    """One tool-call format.

    ``start``/``end`` delimit a call; an empty ``end`` means the call runs to
    the end of its line (Mistral). ``whole`` formats have no start marker: the
    entire message is the call (Llama 3.x JSON). ``parse`` gets the text
    between the markers and the request's tools and returns the calls, or
    raises when the text is not a call in this format.
    """

    name: str
    start: str
    end: str
    parse: Callable[[str, Any], list[Call]]
    whole: bool = False


def _call(name: Any, arguments: Any) -> Call:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call without a name")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        stripped = arguments.strip() or "{}"
        parsed = json.loads(stripped)  # raises on non-JSON strings
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments are not an object")
        return {"name": name.strip(), "arguments": stripped}
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments are not an object")
    return {
        "name": name.strip(),
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


def _calls_from(parsed: Any) -> list[Call]:
    items = parsed if isinstance(parsed, list) else [parsed]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("tool call is not an object")
        if "function" in item and isinstance(item["function"], dict):
            item = item["function"]
        args = item.get("arguments", item.get("parameters"))
        calls.append(_call(item.get("name"), args))
    if not calls:
        raise ValueError("no tool call")
    return calls


# ── Upstream-backed formats ─────────────────────────────────────────────────


def _load_module(name: str):
    try:
        from mlx_vlm.tools import load_tool_module

        return load_tool_module(name)
    except (ImportError, ValueError):
        pass
    try:
        return importlib.import_module(f"mlx_lm.tool_parsers.{name}")
    except ImportError:
        return None


def _upstream(name: str) -> ToolFormat | None:
    module = _load_module(name)
    if module is None:
        return None

    def parse(body: str, tools: Any) -> list[Call]:
        body = _unwrap_doubled_braces(body)
        if "<function=" in body:
            return _parse_function_xml(module, body, tools)
        try:
            return _calls_from(module.parse_tool_call(body, tools))
        except Exception:
            loose = _parse_loose_tag(body, tools)
            if loose:
                return loose
            raise

    return ToolFormat(
        name=name,
        start=module.tool_call_start,
        end=module.tool_call_end,
        parse=parse,
    )


# ── Tolerant readers for Qwen's <function=...> XML ──────────────────────────

_FUNCTION_SPAN = re.compile(
    r"<function=.*?(?:</function>|(?=<function=)|\Z)", re.DOTALL
)


def _parse_function_xml(module: Any, body: str, tools: Any) -> list[Call]:
    """Read ``<function=name><parameter=k>v</parameter></function>`` calls.

    Upstream requires the body to end in ``</function>``. Models routinely
    drop it (``</parameter>`` then ``</tool_call>``), put several functions in
    one ``<tool_call>``, or emit a value that does not convert to its declared
    type. Each function span is closed and parsed on its own; a span whose
    typed conversion fails is read again with plain string values (the
    request-schema coercion downstream still types what it can)."""
    calls: list[Call] = []
    for span in _FUNCTION_SPAN.findall(body):
        text = span.rstrip()
        if not text.endswith("</function>"):
            text += "\n</function>"
        try:
            got = module.parse_tool_call(text, tools)
        except Exception:  # noqa: BLE001
            got = module.parse_tool_call(text, None)
        calls.extend(_calls_from(got))
    if not calls:
        raise ValueError("no <function=> call")
    return calls


_LOOSE_TAG = re.compile(r"\s*<(?P<name>[\w.\-]+)>(?P<rest>.*)$", re.DOTALL)
_LOOSE_PARAM = re.compile(
    r"<parameter=(?P<key>[\w.\-]+)[\"'>:=\s]*(?P<val>.*?)(?:</parameter>|(?=<parameter=)|\Z)",
    re.DOTALL,
)


def _parse_loose_tag(body: str, tools: Any) -> list[Call]:
    """``<Read><parameter=file_path": /x"}`` — a tool-named tag with garbled
    parameter syntax. Read only when the tag names a tool of the request, so
    ordinary markup is never taken for a call; stray quote/brace residue
    around a value is trimmed."""
    m = _LOOSE_TAG.match(body)
    known = {
        t["function"]["name"]
        for t in (tools or [])
        if isinstance(t.get("function"), dict)
    }
    if not m or m.group("name") not in known:
        return []
    args: dict[str, str] = {}
    for p in _LOOSE_PARAM.finditer(m.group("rest")):
        val = p.group("val").strip().rstrip("}").strip().strip("\"'").strip()
        args[p.group("key")] = val
    return [_call(m.group("name"), args)] if args else []


# ── Formats upstream lacks ──────────────────────────────────────────────────


def _unwrap_doubled_braces(body: str) -> str:
    """Repair the common small-model malformation of a JSON tool call.

    Small models copy a prompt-template's escaped braces and emit
    ``{{"name": ...}}`` (sometimes with a stray trailing ``)``). Try dropping the
    stray ``)`` and one brace layer; accept only a candidate that parses as JSON
    when the original does not."""
    t = body.strip()
    try:
        json.loads(t)
        return body
    except ValueError:
        pass
    if not t.startswith("{{"):
        return body
    core = t[:-1].rstrip() if t.endswith(")") else t
    for cand in (core[1:-1], core[1:], core):
        try:
            if isinstance(json.loads(cand), dict):
                return cand
        except ValueError:
            continue
    return body


_FUNCTION_ARGS_SLIP = re.compile(
    r'^\{\s*"(?:function|name)"\s*:\s*"(?P<name>[^"]+)"\s*[:,]\s*'
    r'"(?P<key>arguments|parameters)"\s*:\s*(?P<args>.*)$',
    re.DOTALL,
)
_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _json_candidates(body: str):
    """The body, then the repairs of the malformations models produce: a code
    fence, ``{"function": "f": "arguments": {...}}`` (a key/value slip for
    ``{"name": "f", "arguments": {...}}``) and missing closing braces."""
    text = _unwrap_doubled_braces(body).strip()
    fence = _JSON_FENCE.match(text)
    if fence:
        text = fence.group(1).strip()
    yield text
    slip = _FUNCTION_ARGS_SLIP.match(text)
    head = ""
    if slip:
        name, args = json.dumps(slip.group("name")), slip.group("args").rstrip()
        head = f'{{"name": {name}, "arguments": {args}'
        yield head
    for extra in ("}", "}}"):
        yield text + extra
        if slip:
            yield head + extra


def _parse_yunshu_json(body: str, _tools: Any) -> list[Call]:
    decoder = json.JSONDecoder()
    last: Exception = ValueError("empty tool call")
    for cand in _json_candidates(body):
        try:
            # raw_decode: tolerate junk after the object (a stray ``)``).
            obj, _end = decoder.raw_decode(cand)
            return _calls_from(obj)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last


INJECTED_JSON = ToolFormat(
    name="yunshu_json",
    start="<tool_call>",
    end="</tool_call>",
    parse=_parse_yunshu_json,
)

_DS_CALL = re.compile(
    r"<｜tool▁call▁begin｜>(?P<head>.*?)<｜tool▁sep｜>(?P<rest>.*?)<｜tool▁call▁end｜>",
    re.DOTALL,
)
_DS_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_deepseek(body: str, _tools: Any) -> list[Call]:
    calls = []
    for m in _DS_CALL.finditer(body):
        head, rest = m.group("head").strip(), m.group("rest")
        if head in ("", "function"):
            # R1 / V3: ``function<sep>name\n```json\n{...}\n```.``
            name, _, rest = rest.strip().partition("\n")
        else:
            name = head  # V3.1: ``name<sep>{...}``
        fenced = _DS_FENCE.search(rest)
        args = (fenced.group(1) if fenced else rest).strip() or "{}"
        name = name.strip().removeprefix("functions.")
        calls.append(_call(name, args))
    if not calls:
        raise ValueError("no DeepSeek tool call")
    return calls


DEEPSEEK = ToolFormat(
    name="deepseek",
    start="<｜tool▁calls▁begin｜>",
    end="<｜tool▁calls▁end｜>",
    parse=_parse_deepseek,
)


def _parse_json_message(body: str, _tools: Any) -> list[Call]:
    text = body.strip().removeprefix("<|python_tag|>").strip()
    if not text.startswith("{"):
        raise ValueError("not a JSON call message")
    decoder = json.JSONDecoder()
    calls, i = [], 0
    while i < len(text):
        obj, i = decoder.raw_decode(text, i)
        calls.extend(_calls_from(obj))
        while i < len(text) and text[i] in " \n\t;":
            i += 1
    return calls


JSON_MESSAGE = ToolFormat(
    name="json_message",
    start="",
    end="",
    parse=_parse_json_message,
    whole=True,
)

# Chat-template markers for the Yunshu-owned formats (upstream's registry
# decides everything else).
_OWN_TEMPLATE_MARKERS: tuple[tuple[str, ToolFormat], ...] = (
    ("<｜tool▁calls▁begin｜>", DEEPSEEK),
)

_FALLBACK: tuple[ToolFormat, ...] = (INJECTED_JSON, JSON_MESSAGE)


def fallback_formats() -> tuple[ToolFormat, ...]:
    """Formats for a model whose template has no tool-call format."""
    return _FALLBACK


# ── Detection ───────────────────────────────────────────────────────────────


def _template_text(tokenizer: Any) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    try:
        from mlx_vlm.tools.registry import _template_text as upstream_text

        return upstream_text(template)
    except ImportError:  # pragma: no cover - mlx-vlm is a dependency
        return template if isinstance(template, str) else None


def native_format(tokenizer: Any) -> ToolFormat | None:
    """The tool-call format a tokenizer's chat template teaches the model, or
    None when the template has no tool-call format."""
    text = _template_text(tokenizer)
    if not text:
        return None
    for marker, fmt in _OWN_TEMPLATE_MARKERS:
        if marker in text:
            return fmt
    name = None
    try:
        from mlx_vlm.tools.registry import _infer_tool_parser

        name = _infer_tool_parser(text)
    except ImportError:  # pragma: no cover
        pass
    if name is None:
        try:
            from mlx_lm.tokenizer_utils import _infer_tool_parser as lm_infer

            name = lm_infer(text)
        except ImportError:  # pragma: no cover
            name = None
    if name is None or name == "json_tools":
        # json_tools is the same shape Yunshu's injected prompt uses.
        return None
    return _upstream(name)


def formats_for_tokenizer(tokenizer: Any) -> tuple[ToolFormat, ...]:
    """The formats to try, in order: the model's native format, the
    ``yunshu_json`` form the injected tool prompt asks for, and a bare JSON
    call message."""
    native = native_format(tokenizer) if tokenizer is not None else None
    return (native, *_FALLBACK) if native is not None else _FALLBACK


def _engine_tokenizer(engine: Any) -> Any:
    for attr in ("_processor", "processor"):
        proc = getattr(engine, attr, None)
        tok = getattr(proc, "tokenizer", None)
        if tok is not None and getattr(tok, "chat_template", None):
            return tok
        if proc is not None and getattr(proc, "chat_template", None):
            return proc
    for attr in ("_tokenizer", "tokenizer"):
        tok = getattr(engine, attr, None)
        if tok is not None:
            return tok
    return None


def tool_formats(engine: Any) -> tuple[ToolFormat, ...]:
    """The tool-call formats of the model an engine serves (cached on it)."""
    if engine is None:
        return _FALLBACK
    cached = getattr(engine, "_yunshu_tool_formats", None)
    if cached is not None:
        return cached
    try:
        formats = formats_for_tokenizer(_engine_tokenizer(engine))
    except Exception:
        logger.debug("tool format detection failed", exc_info=True)
        formats = _FALLBACK
    with contextlib.suppress(Exception):  # slotted / frozen engines
        engine._yunshu_tool_formats = formats
    return formats


# ── Extraction ──────────────────────────────────────────────────────────────


def openai_tools(tools: Any) -> list[dict] | None:
    """Request tools as OpenAI chat dicts (what upstream parsers read for
    schema-typed values): chat pydantic models, Responses' flat entries and
    Anthropic's ``input_schema`` tools all map to ``{"type": "function",
    "function": {"name", "parameters"}}``."""
    if not tools:
        return None
    out = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        params = fn.get("parameters") or fn.get("input_schema") or {}
        out.append(
            {"type": "function", "function": {"name": name, "parameters": params}}
        )
    return out or None


def _span_end(text: str, fmt: ToolFormat, body_start: int) -> tuple[int, int] | None:
    """(body_end, span_end) of a call whose body starts at ``body_start``, or
    None when its end marker has not arrived yet."""
    if fmt.end:
        idx = text.find(fmt.end, body_start)
        return None if idx < 0 else (idx, idx + len(fmt.end))
    idx = text.find("\n", body_start)
    return None if idx < 0 else (idx, idx + 1)


def next_start(text: str, formats: Sequence[ToolFormat], pos: int = 0):
    """Earliest start marker at or after ``pos``: (index, marker, formats that
    open with it) or None."""
    best = None
    for fmt in formats:
        if fmt.whole or not fmt.start:
            continue
        i = text.find(fmt.start, pos)
        if i < 0:
            continue
        if (
            best is None
            or i < best[0]
            or (i == best[0] and len(fmt.start) > len(best[1]))
        ):
            best = (i, fmt.start)
    if best is None:
        return None
    group = [f for f in formats if f.start == best[1] and not f.whole]
    return best[0], best[1], group


_CALL_LOOK = re.compile(r"\s*(?:<function=|<parameter=|<\w+>|[\[{])")


def parse_block(
    text: str, marker_at: int, group: Sequence[ToolFormat], tools: Any, final: bool
) -> tuple[list[Call], int, bool] | None:
    """Parse the call whose start marker sits at ``marker_at``.

    Returns ``(calls, end, dropped)``: the calls and the index just past the
    call. ``([], end, True)`` means a complete span that looks like a call but
    parses in no format: it is dropped, never shown (tool markup does not
    reach the content stream). ``([], end, False)`` means the marker was not a
    call at all (prose that mentions it): only the marker is skipped and the
    text stays visible. None means the span is still open (and not ``final``).
    At end of output an unterminated span is parsed to the end.
    """
    open_span = False
    unreadable: tuple[str, int] | None = None
    for fmt in group:
        body_start = marker_at + len(fmt.start)
        span = _span_end(text, fmt, body_start)
        if span is None:
            if not final:
                open_span = True
                continue
            span = (len(text), len(text))
        body = text[body_start : span[0]]
        try:
            calls = fmt.parse(body, tools)
        except Exception as exc:  # noqa: BLE001 - any parser failure = not this format
            logger.debug("%s parser rejected a call: %s", fmt.name, exc)
            if unreadable is None:
                unreadable = (body, span[1])
            continue
        return calls, span[1], False
    if open_span:
        return None
    if unreadable is not None and _CALL_LOOK.match(unreadable[0]):
        logger.warning(
            "dropped an unreadable tool call (%d chars): %.200r",
            len(unreadable[0]),
            unreadable[0],
        )
        return [], unreadable[1], True
    # Not a call (prose): skip past the marker, keeping the text visible.
    return [], marker_at + len(group[0].start), False


def _whole_calls(text: str, formats: Sequence[ToolFormat], tools: Any):
    for fmt in formats:
        if fmt.whole:
            try:
                return fmt.parse(text, tools)
            except Exception:  # noqa: BLE001
                continue
    return None


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_tool_output(
    text: str, formats: Sequence[ToolFormat], tools: Any = None
) -> tuple[list[Call], str]:
    """Extract tool calls from a finished model output.

    Returns ``(calls, remaining_text)``: calls in output order, arguments typed
    with ``tools``' schemas; the text with every parsed call span removed.
    """
    if not text:
        return [], text or ""
    tools = openai_tools(tools)
    whole = _whole_calls(text, formats, tools)
    if whole:
        return coerce_tool_calls(whole, tools) or [], ""
    calls: list[Call] = []
    kept: list[str] = []
    pos = 0
    any_dropped = False
    while True:
        found = next_start(text, formats, pos)
        if found is None:
            break
        at, _marker, group = found
        got, end, dropped = parse_block(text, at, group, tools, final=True)
        if got:
            kept.append(text[pos:at])
            calls.extend(got)
        elif dropped:
            kept.append(text[pos:at])
            any_dropped = True
        else:
            kept.append(text[pos:end])
        pos = end
    kept.append(text[pos:])
    if not calls:
        return [], _tidy("".join(kept)) if any_dropped else text
    return coerce_tool_calls(calls, tools) or [], _tidy("".join(kept))
