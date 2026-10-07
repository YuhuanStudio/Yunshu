# Upstream (inspired): guidance-ai/llguidance (MIT) python/llguidance/_struct_tag.py structural tags
# Upstream (inspired): mlx-vlm (MIT) mlx_vlm/structured.py llguidance logits processor
"""Structural-tag constrained decoding for tool calls.

Generation is free until the model emits the tool-call start marker
(``<tool_call>``). From that token on, the call body is constrained to the
exact grammar of the model's call format for *this request's* tools: the
function name is one of the request's tool names, parameters are that tool's
schema keys, typed values follow the schema, and the structure closes
correctly (``</function>`` then ``</tool_call>``). After the closing marker
generation is free again. A forced ``tool_choice`` (``required`` / one tool)
starts constrained instead, after the reasoning block.

The grammar engine is llguidance (a dependency of mlx-vlm): a Rust
Earley/lexer engine that turns a Lark grammar into a per-step token bitmask.
This module owns

- the grammar text for the two call formats Yunshu reads (Qwen3.x XML
  ``<function=..><parameter=..>`` and Hermes-style JSON),
- :class:`ToolCallGuide`, the per-request state machine (waiting for the end of
  reasoning, free, constrained) with snapshot / rollback so a speculative
  verify can mask every position of its window, and
- :class:`ToolCallProcessor`, the logits processor for one-token-at-a-time rows.

A mask never removes a token from a call the model could have written
validly, so a well-formed unconstrained call is reproduced token for token;
the mask only closes the paths to malformed ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Special / added tokens that may appear inside a string value (a file that
# talks about tool calls). They are single tokens, which llguidance treats as
# non-text, so the grammar names them.
_TEXT_SPECIAL_TOKENS = (
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
)

_JSON_TYPES = {"array", "object"}

# Whitespace a forced reply may put before, between and after calls: at most two characters. The
# mask removes EOS until a call has closed, so an unbounded whitespace rule let one sampled "\n"
# snowball into 1500 tokens of blanks and no call (M3 sweep, Qwen2.5-3B, chat stream).
_WS_RULE = "WS: /[ \\n]{1,2}/"


# ── grammar text ────────────────────────────────────────────────────────────


def _q(text: str) -> str:
    """A Lark string literal."""
    return json.dumps(text, ensure_ascii=False)


def _schema_type(schema: Any) -> str | None:
    """The single JSON type a parameter schema declares (None: unknown / union)."""
    if not isinstance(schema, dict):
        return None
    values = schema.get("enum")
    if isinstance(values, list) and values and all(isinstance(v, str) for v in values):
        return "enum"
    if isinstance(schema.get("const"), str):
        return "enum"
    t = schema.get("type")
    if isinstance(t, str):
        return t.lower()
    if isinstance(t, list):
        kinds = [k for k in t if k != "null"]
        if len(kinds) == 1 and isinstance(kinds[0], str):
            return kinds[0].lower()
    return None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    parameters: dict


def normalize_tools(tools: Any) -> list[ToolSpec]:
    """Request tools (OpenAI chat dicts, pydantic models, flat Responses / Anthropic
    entries) as ``ToolSpec`` list."""
    out: list[ToolSpec] = []
    for tool in tools or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        if params is None:
            params = fn.get("input_schema")
        out.append(ToolSpec(name, params if isinstance(params, dict) else {}))
    return out


class _GrammarBuilder:
    def __init__(self, validate_json, text_specials: list[int]):
        self.validate_json = validate_json
        self.text_specials = text_specials
        self.rules: list[str] = []
        self.fallbacks = 0

    def value(self, prop: Any, defs: dict | None) -> str:
        """Lark expression for a parameter value and its ``</parameter>`` close (the
        newline before the close belongs to the value's terminator)."""
        kind = _schema_type(prop)
        nullable = (
            isinstance(prop, dict)
            and isinstance(prop.get("type"), list)
            and "null" in prop["type"]
        )
        null = ' | "null"' if nullable else ""
        close = '"\\n</parameter>"'
        if kind == "integer":
            return f"(INT{null}) {close}"
        if kind == "number":
            return f"(NUMBER{null}) {close}"
        if kind == "boolean":
            # the template renders history booleans as Python or JSON literals
            return f'("true" | "false" | "True" | "False"{null}) {close}'
        if kind == "enum":
            values = prop["enum"] if "enum" in prop else [prop["const"]]
            return "(" + " | ".join(_q(v) for v in values) + f") {close}"
        if kind in _JSON_TYPES:
            schema = dict(prop)
            if defs:
                schema.setdefault("$defs", defs)
            if self.validate_json(schema):
                return f"(%json {json.dumps(schema, ensure_ascii=False)}{null}) {close}"
            self.fallbacks += 1
        return "text"


def build_xml_grammar(
    tools: list[ToolSpec],
    *,
    start_id: int,
    end_id: int,
    text_specials: list[int],
    validate_json,
    forced: bool = False,
    parallel: bool = True,
    only: str | None = None,
) -> str:
    """Lark grammar for Qwen3.x XML tool calls.

    Auto mode (``forced`` false) covers the body after the start marker; forced
    mode covers the whole reply: ``<tool_call>`` then the body, repeated.
    """
    b = _GrammarBuilder(validate_json, text_specials)
    chosen = [t for t in tools if only is None or t.name == only]
    if not chosen:
        raise ValueError("no tool matches the forced tool_choice")
    spec_alt = " | ".join(f"<[{i}]>" for i in text_specials) or "<[0]>"
    lines: list[str] = []
    calls: list[str] = []
    for ti, tool in enumerate(chosen):
        params = tool.parameters or {}
        props = params.get("properties")
        defs = params.get("$defs") or params.get("definitions")
        alts: list[str] = []
        if isinstance(props, dict) and props:
            for ki, (key, prop) in enumerate(props.items()):
                rule = f"p{ti}_{ki}"
                lines.append(
                    f"{rule}: {_q('<parameter=' + key + '>' + chr(10))} "
                    f'{b.value(prop, defs)} "\\n"?'
                )
                alts.append(rule)
        else:
            alts.append("pany")
        lines.append(f"params{ti}: ({' | '.join(alts)})*")
        head = _q("<function=" + tool.name + ">")
        lines.append(
            f'call{ti}: {head} "\\n"? params{ti} "</function>" "\\n"? <[{end_id}]>'
        )
        calls.append(f"call{ti}")
    text_rules = [
        "text: (TXT? spec)* VALT",
        f"spec: {spec_alt}",
        # Text up to the first newline + </parameter>; a single lexeme so the lexer
        # cannot end the text inside a partial close.
        r"VALT: /(.|\n)*\n<\/parameter>/ & ~/(.|\n)*\n<\/parameter>(.|\n)+/",
        r"TXT: /(.|\n)+/ & ~/(.|\n)*\n<\/parameter>(.|\n)*/",
        r"INT: /-?(0|[1-9][0-9]*)/",
        r"NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/",
        r'pany: "<parameter=" ANYKEY ">\n" text "\n"?',
        r"ANYKEY: /[^>\n]+/",
    ]
    call_alt = " | ".join(calls)
    if forced:
        one = f'<[{start_id}]> "\\n"? call'
        head = [
            f"start: WS? tcall{' (WS? tcall)*' if parallel else ''} WS?",
            f"tcall: {one}",
            _WS_RULE,
        ]
    else:
        head = ['start: "\\n"? call']
    return "\n".join([*head, f"call: {call_alt}", *lines, *text_rules]) + "\n"


def _inline_refs(schema: Any, _depth: int = 0) -> dict | None:
    """``schema`` with its local ``$ref``s (``#/$defs/X``, ``#/definitions/X``) replaced by the
    definitions they name, so a tool whose parameters use shared definitions still compiles.
    None when a reference is not local or recurses (a call grammar cannot be unbounded)."""
    defs: dict = {}
    if isinstance(schema, dict):
        for key in ("$defs", "definitions"):
            if isinstance(schema.get(key), dict):
                defs.update(schema[key])

    def walk(node: Any, seen: tuple) -> Any:
        if isinstance(node, list):
            items = [walk(n, seen) for n in node]
            return None if any(o is None for o in items) else items
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if ref is not None:
            name = ref.rsplit("/", 1)[-1] if isinstance(ref, str) else None
            if (
                not isinstance(ref, str)
                or not ref.startswith(("#/$defs/", "#/definitions/"))
                or name not in defs
                or name in seen
                or len(seen) > 8
            ):
                return None
            rest = {k: v for k, v in node.items() if k != "$ref"}
            target = walk(defs[name], (*seen, name))
            if target is None:
                return None
            if not isinstance(target, dict):
                return target
            merged = {**target, **walk(rest, seen)} if rest else target
            return merged
        copy: dict = {}
        for k, v in node.items():
            if k in ("$defs", "definitions"):
                continue
            w = walk(v, seen)
            if w is None and v is not None:
                return None
            copy[k] = w
        return copy

    res = walk(schema, ())
    return res if isinstance(res, dict) else None


def build_json_grammar(
    tools: list[ToolSpec],
    *,
    start_id: int,
    end_id: int,
    forced: bool = False,
    parallel: bool = True,
    only: str | None = None,
) -> str | None:
    """Lark grammar for Hermes-style ``{"name": .., "arguments": {..}}`` calls; None
    when a tool schema uses ``$ref`` (its base cannot be kept per tool)."""
    branches = []
    for tool in tools:
        if only is not None and tool.name != only:
            continue
        params = _inline_refs(tool.parameters or {"type": "object"})
        if params is None:
            return None
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": tool.name},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    if not branches:
        raise ValueError("no tool matches the forced tool_choice")
    schema = branches[0] if len(branches) == 1 else {"anyOf": branches}
    call = f'"\\n"? %json {json.dumps(schema, ensure_ascii=False)} "\\n"? <[{end_id}]>'
    if forced:
        return "\n".join(
            [
                f"start: WS? tcall{' (WS? tcall)*' if parallel else ''} WS?",
                f"tcall: <[{start_id}]> call",
                f"call: {call}",
                _WS_RULE,
            ]
        )
    return f"start: call\ncall: {call}\n"


def detect_style(tokenizer: Any) -> str | None:
    """``"xml"`` (Qwen3.x ``<function=..>``) or ``"json"`` (``<tool_call>{..}</tool_call>``)
    from what the chat template teaches, None when the template has no tool call format."""
    try:
        from .tool_format import _template_text

        text = _template_text(tokenizer)
    except Exception:
        text = None
    if not isinstance(text, str) or "<tool_call>" not in text:
        return None
    if "<function=" in text:
        return "xml"
    if "<arg_key>" in text:
        return None
    if '"name"' in text or "'name'" in text:
        return "json"
    return None


# ── compiled grammar ────────────────────────────────────────────────────────

_TOKENIZERS: dict[tuple[int, int], tuple[Any, Any]] = {}
_TOKENIZER_LOCK = threading.Lock()


def _hf_tokenizer(tokenizer: Any) -> Any:
    for attr in ("_tokenizer", "tokenizer"):
        inner = getattr(tokenizer, attr, None)
        if inner is not None and hasattr(inner, "backend_tokenizer"):
            return inner
    return tokenizer


def llg_tokenizer(tokenizer: Any, vocab_size: int):
    """llguidance tokenizer for ``tokenizer`` with masks ``vocab_size`` wide (cached;
    building one walks the vocabulary, about a second)."""
    hf = _hf_tokenizer(tokenizer)
    key = (id(hf), int(vocab_size))
    with _TOKENIZER_LOCK:
        hit = _TOKENIZERS.get(key)
        if hit is not None:
            return hit[1]
    import llguidance.hf

    built = llguidance.hf.from_tokenizer(hf, n_vocab=int(vocab_size))
    with _TOKENIZER_LOCK:
        _TOKENIZERS[key] = (hf, built)  # keeps ``hf`` alive so its id stays unique
    return built


class ToolGrammar:
    """A tool list compiled for one tokenizer and ``tool_choice``: build a
    :class:`ToolCallGuide` per request with :meth:`guide`."""

    def __init__(
        self,
        lark: str,
        llt: Any,
        *,
        start_id: int,
        end_id: int,
        think_end_id: int | None,
        forced: bool,
        style: str,
        eos_ids: tuple[int, ...] = (),
    ):
        from llguidance import LLMatcher

        self.lark = lark
        self.llt = llt
        self.start_id = start_id
        self.end_id = end_id
        self.think_end_id = think_end_id
        self.forced = forced
        self.style = style
        self.eos_ids = tuple(eos_ids)
        self.words = (llt.vocab_size + 31) // 32
        self._template = LLMatcher(llt, LLMatcher.grammar_from_lark(lark))
        err = self._template.get_error()
        if err:
            raise ValueError(f"tool grammar failed to compile: {err[:300]}")

    def new_matcher(self):
        return self._template.deep_copy()

    def guide(self, *, thinking_open: bool = False) -> ToolCallGuide:
        return ToolCallGuide(self, thinking_open=thinking_open)


def _valid_json_schema(llt: Any):
    from llguidance import LLMatcher

    cache: dict[str, bool] = {}

    def ok(schema: dict) -> bool:
        key = json.dumps(schema, sort_keys=True, ensure_ascii=False)
        hit = cache.get(key)
        if hit is None:
            try:
                grammar = LLMatcher.grammar_from_json_schema(key)
                hit = not LLMatcher.validate_grammar(grammar, llt)
            except Exception:
                hit = False
            cache[key] = hit
        return hit

    return ok


def _token_id(tokenizer: Any, token: str) -> int | None:
    hf = _hf_tokenizer(tokenizer)
    try:
        tid = hf.convert_tokens_to_ids(token)
    except Exception:
        return None
    if tid is None or (
        getattr(hf, "unk_token_id", None) is not None and tid == hf.unk_token_id
    ):
        return None
    return int(tid)


def _eos_ids(tokenizer: Any) -> tuple[int, ...]:
    """Every token that ends generation for ``tokenizer`` (its eos ids plus the chat / text
    end markers), so a forced reply cannot end inside its reasoning."""
    ids: set[int] = set()
    for tok in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        if tok is None:
            continue
        many = getattr(tok, "eos_token_ids", None)
        if isinstance(many, (set, list, tuple)):
            ids.update(int(i) for i in many if isinstance(i, int))
        one = getattr(tok, "eos_token_id", None)
        if isinstance(one, int):
            ids.add(one)
    for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
        i = _token_id(tokenizer, name)
        if i is not None:
            ids.add(i)
    return tuple(sorted(ids))


def is_forced(choice: Any) -> bool:
    """True when ``tool_choice`` forces a call (required / any / a named function). A forced
    choice is a guarantee in both the OpenAI and the Anthropic API, so it is always
    constrained by the tool grammar; ``YUNSHU_TOOL_GRAMMAR`` governs only auto."""
    c = normalize_tool_choice(choice)
    return c == "required" or isinstance(c, dict)


def normalize_tool_choice(choice: Any) -> Any:
    """OpenAI / Anthropic ``tool_choice`` as None (auto), "none", "required" or
    ``{"name": X}`` (that tool)."""
    if choice is None:
        return None
    if hasattr(choice, "model_dump"):
        choice = choice.model_dump()
    if isinstance(choice, str):
        return {"required": "required", "any": "required", "none": "none"}.get(choice)
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "none":
            return "none"
        if kind == "any":
            return "required"
        if kind == "auto":
            return None
        fn = choice.get("function")
        name = choice.get("name") or (fn.get("name") if isinstance(fn, dict) else None)
        return {"name": name} if name else None
    return None


def compile_tool_grammar(
    tools: Any,
    tokenizer: Any,
    vocab_size: int,
    *,
    tool_choice: Any = None,
    parallel: bool = True,
) -> ToolGrammar | None:
    """Compile ``tools`` for ``tokenizer``. ``tool_choice``: None / "auto" (free until
    the marker), "required" (a call is forced), or ``{"name": X}`` / a tool name
    string (that call is forced). Returns None when this model / tool set cannot be
    constrained (unknown format, marker is not a single token, invalid schema); the
    caller then decodes unconstrained."""
    specs = normalize_tools(tools)
    if not specs:
        return None
    style = detect_style(tokenizer)
    if style is None:
        from .tool_family_grammar import build_native_grammar
        from .tool_format import native_format

        fmt = native_format(tokenizer)
        choice = normalize_tool_choice(tool_choice)
        if fmt is None or not is_forced(tool_choice):
            return None
        only = choice.get("name") if isinstance(choice, dict) else None
        try:
            llt = llg_tokenizer(tokenizer, vocab_size)
            lark = build_native_grammar(
                specs, fmt, tokenizer, only=only, parallel=parallel
            )
            return ToolGrammar(
                lark,
                llt,
                start_id=-1,
                end_id=-1,
                think_end_id=_token_id(tokenizer, "</think>"),
                forced=True,
                style=fmt.name,
                eos_ids=_eos_ids(tokenizer),
            )
        except Exception:
            logger.warning("native tool grammar unavailable", exc_info=True)
            return None
    start_id = _token_id(tokenizer, "<tool_call>")
    end_id = _token_id(tokenizer, "</tool_call>")
    if start_id is None or end_id is None:
        return None
    only = None
    forced = False
    if tool_choice == "required":
        forced = True
    elif isinstance(tool_choice, dict) and tool_choice.get("name"):
        forced, only = True, str(tool_choice["name"])
    if only is not None and all(t.name != only for t in specs):
        return None
    try:
        llt = llg_tokenizer(tokenizer, vocab_size)
        if style == "xml":
            specials = [
                i
                for i in (_token_id(tokenizer, t) for t in _TEXT_SPECIAL_TOKENS)
                if i is not None
            ]
            lark = build_xml_grammar(
                specs,
                start_id=start_id,
                end_id=end_id,
                text_specials=specials,
                validate_json=_valid_json_schema(llt),
                forced=forced,
                parallel=parallel,
                only=only,
            )
        else:
            lark = build_json_grammar(
                specs,
                start_id=start_id,
                end_id=end_id,
                forced=forced,
                parallel=parallel,
                only=only,
            )
            if lark is None:
                return None
        return ToolGrammar(
            lark,
            llt,
            start_id=start_id,
            end_id=end_id,
            think_end_id=_token_id(tokenizer, "</think>"),
            forced=forced,
            style=style,
            eos_ids=_eos_ids(tokenizer),
        )
    except Exception:
        logger.warning(
            "tool-call grammar unavailable; decoding unconstrained", exc_info=True
        )
        return None


def grammar_key(tools: Any, tool_choice: Any, parallel: bool) -> str:
    payload = json.dumps(
        [
            [(t.name, t.parameters) for t in normalize_tools(tools)],
            tool_choice,
            bool(parallel),
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


# ── per-request state ───────────────────────────────────────────────────────

WAIT, FREE, BODY = "wait", "free", "body"


class ToolCallGuide:
    """Where a request is in its reply — inside reasoning (``WAIT``), free text
    (``FREE``) or a constrained tool call (``BODY``) — and the token mask that
    follows from it.

    ``feed`` advances over one generated token; ``checkpoint`` / ``restore``
    undo any run of ``feed`` calls (a speculative verify masks every position of its
    window along the draft path, then rewinds). A token the grammar rejects
    cannot happen when the mask was applied; if it does (a mask that was not
    applied) the guide goes free instead of wedging the request."""

    def __init__(self, grammar: ToolGrammar, *, thinking_open: bool = False):
        self.grammar = grammar
        self.matcher = None
        self.consumed = 0
        self.broken = False
        # engagement counters (logged when a call closes)
        self.calls = 0
        self.masked = 0
        self.lane_rounds = 0
        self._planning = 0
        if thinking_open and grammar.think_end_id is not None:
            self.phase = WAIT
        elif grammar.forced:
            self.phase = BODY
            self.matcher = grammar.new_matcher()
        else:
            self.phase = FREE
        self._row = np.empty((1, grammar.words), dtype=np.int32)

    # ── state ──
    @property
    def constrained(self) -> bool:
        # a forced reply is also masked while it reasons: it must not end before the call
        return self.phase == BODY or (self.phase == WAIT and self.grammar.forced)

    def arms(self, token: int) -> bool:
        """True when ``token``, fed now, switches an unconstrained state into a
        constrained one (the positions after it must be masked)."""
        g = self.grammar
        if self.phase == FREE:
            return token == g.start_id
        if self.phase == WAIT:
            return g.forced and token == g.think_end_id
        return False

    def feed(self, token: int) -> bool:
        g = self.grammar
        token = int(token)
        if self.phase == WAIT:
            if token == g.think_end_id:
                if g.forced:
                    self.phase = BODY
                    self.matcher = g.new_matcher()
                    self.consumed = 0
                else:
                    self.phase = FREE
            return True
        if self.phase == FREE:
            if token == g.start_id:
                self.phase = BODY
                self.matcher = g.new_matcher()
                self.consumed = 0
                if g.forced:  # the forced grammar names the marker itself
                    return self._consume(token)
            return True
        return self._consume(token)

    def _consume(self, token: int) -> bool:
        m = self.matcher
        if not m.consume_token(token):
            logger.warning(
                "tool-call grammar rejected token %d (%s); decoding this reply unconstrained",
                token,
                (m.get_error() or "")[:120],
            )
            self.broken = True
            self.phase = FREE
            return False
        self.consumed += 1
        if m.is_stopped():
            self.phase = FREE
            self.calls += 1
            if not self._planning:
                logger.info(
                    "tool call closed under grammar (%d tokens masked so far in %d calls, "
                    "%d masked lane rounds)",
                    self.masked,
                    self.calls,
                    self.lane_rounds,
                )
        return True

    def checkpoint(self) -> tuple:
        return (
            self.phase,
            self.matcher,
            self.consumed,
            self.broken,
            self.calls,
        )

    def restore(self, cp: tuple) -> None:
        phase, matcher, consumed, broken, calls = cp
        if matcher is not None and matcher is self.matcher and self.consumed > consumed:
            matcher.rollback(self.consumed - consumed)
        self.phase, self.matcher, self.consumed, self.broken = (
            phase,
            matcher,
            consumed,
            broken,
        )
        self.calls = calls

    def advance(self, tokens: list[int]) -> int:
        """Feed ``tokens`` until one arms a constrained state (that token is
        included); returns how many were consumed. The positions after an arming
        token were sampled without its mask, so a speculative round keeps only
        the tokens up to it."""
        for i, tok in enumerate(tokens):
            arms = self.arms(tok)
            self.feed(tok)
            if arms:
                return i + 1
        return len(tokens)

    # ── masks ──
    def fill(self, out: np.ndarray) -> bool:
        """Write the next-token mask into ``out`` (int32 words); False when the next
        token is unconstrained."""
        if self.phase == WAIT and self.grammar.forced:
            # reasoning is free text, but it may not end (EOS) before the forced call: a
            # model that stops inside <think> would otherwise answer with no call at all
            out[:] = -1
            bits = out.view(np.uint32)
            for t in self.grammar.eos_ids:
                if t < self.grammar.words * 32:
                    bits[t >> 5] &= np.uint32(~(1 << (t & 31)) & 0xFFFFFFFF)
            self.masked += 1
            return True
        if self.phase != BODY or self.matcher is None:
            return False
        import llguidance.numpy as lnp

        self.masked += 1
        try:
            lnp.fill_next_token_bitmask(self.matcher, out.reshape(1, -1), 0)
        except Exception:
            logger.warning(
                "tool-call mask failed; decoding unconstrained", exc_info=True
            )
            self.broken = True
            self.phase = FREE
            return False
        return True

    def mask(self) -> np.ndarray | None:
        """The next-token mask row ([words] int32, valid until the next call), or None."""
        return self._row[0] if self.fill(self._row[0]) else None

    def plan(self, drafts: list[int], n_pos: int) -> np.ndarray | None:
        """Masks for the ``n_pos`` positions of a verify window whose input is
        ``[last emitted, *drafts]`` ([n_pos, words] int32; unconstrained positions
        are all ones), or None when no position is constrained. The window is
        walked along the draft path and the guide rewound; positions after a draft
        the grammar rejects, or that arms a constraint, are left unconstrained
        (the round discards them)."""
        if not self.constrained:
            return None
        out = np.full((n_pos, self.grammar.words), -1, dtype=np.int32)
        cp = self.checkpoint()
        self._planning += 1
        try:
            for i in range(n_pos):
                constrained = self.fill(out[i])
                if i >= len(drafts):
                    break
                tok = int(drafts[i])
                if constrained and not (out[i, tok >> 5] >> (tok & 31)) & 1:
                    # The mask forbids this draft, so the round rejects it here.
                    # Feeding it would leave the matcher in an error state.
                    break
                if self.arms(tok) or not self.feed(tok) or self.broken:
                    break
        finally:
            self.restore(cp)
            self._planning -= 1
        return out


class ToolCallProcessor:
    """Logits processor over a :class:`ToolCallGuide` for rows that decode one token at
    a time (``__call__`` for the first token, ``process_last_token`` after)."""

    def __init__(self, guide: ToolCallGuide):
        self.guide = guide

    def _apply(self, logits):
        row = self.guide.mask()
        if row is None:
            return logits
        return apply_bitmask(logits, row)

    def __call__(self, tokens, logits):
        return self._apply(logits)

    def process_last_token(self, token, logits):
        self.guide.feed(int(token))
        return self._apply(logits)


class TokenStreamToolCallProcessor(ToolCallProcessor):
    """The same processor for mlx-lm's ``generate_step`` convention: it is called
    with every token seen so far (the prompt's last token, then each sampled one),
    not just the newest."""

    def __init__(self, guide: ToolCallGuide):
        super().__init__(guide)
        self._seen: int | None = None

    def __call__(self, tokens, logits):
        n = int(tokens.shape[0])
        if self._seen is None:
            self._seen = n  # the prompt's tail: nothing generated yet
        else:
            for i in range(self._seen, n):
                self.guide.feed(int(tokens[i]))
            self._seen = n
        return self._apply(logits)


def prompt_opens_thinking(ids: list[int], grammar: ToolGrammar, tokenizer: Any) -> bool:
    """True when the prompt ends inside an open ``<think>`` block (generation starts
    in the reasoning)."""
    start = _token_id(tokenizer, "<think>")
    end = grammar.think_end_id
    if start is None or end is None:
        return False
    tail = list(ids[-64:])
    if start not in tail:
        return False
    last = len(tail) - 1 - tail[::-1].index(start)
    return end not in tail[last + 1 :]


def apply_bitmask(logits, mask: np.ndarray):
    """``logits`` [..., V] with every token whose bit is 0 in ``mask`` ([words] or
    [rows, words] int32) set to -inf."""
    import mlx.core as mx
    from llguidance.mlx import apply_token_bitmask

    shape = logits.shape
    flat = logits.reshape(-1, shape[-1])
    m = mask.reshape(-1, mask.shape[-1])
    if m.shape[0] == 1 and flat.shape[0] > 1:
        m = np.broadcast_to(m, (flat.shape[0], m.shape[1])).copy()
    out = apply_token_bitmask(flat, m)
    return mx.reshape(out, shape)
