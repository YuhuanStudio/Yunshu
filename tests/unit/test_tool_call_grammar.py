"""Structural-tag constrained decoding for tool calls: the grammar accepts every
call the model's chat template renders, rejects the malformed shapes seen in agent
traffic, and the guide's snapshot / rollback matches token-by-token masking.

Needs the Qwen3.5 (XML calls) and Qwen2.5 (JSON calls) tokenizers; skipped when
they are not on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("llguidance")
pytest.importorskip("transformers")

from yunshu_engine import tool_call_grammar as tcg  # noqa: E402

from .model_paths import model_dir  # noqa: E402

XML_TOKENIZER = model_dir("Qwen3.5-0.8B-MLX-bf16")
JSON_TOKENIZER = model_dir("Qwen2.5-3B-Instruct-4bit")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "run a command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "background": {"type": "boolean"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "mode": {"type": "string", "enum": ["one", "all"]},
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "done": {"type": "boolean"},
                            },
                            "required": ["content"],
                        },
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "Ping", "parameters": {"type": "object"}},
    },
]


@pytest.fixture(scope="module")
def xml_tok():
    if not XML_TOKENIZER.exists():
        pytest.skip("Qwen3.5 tokenizer not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(XML_TOKENIZER))


@pytest.fixture(scope="module")
def xml_grammar(xml_tok):
    grammar = tcg.compile_tool_grammar(TOOLS, xml_tok, len(xml_tok) + 243)
    assert grammar is not None and grammar.style == "xml"
    return grammar


def render(tok, name: str, args: dict) -> str:
    """The text the chat template renders for one assistant tool call."""
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": args}}
            ],
        },
    ]
    text = tok.apply_chat_template(msgs, tokenize=False)
    start = text.index("<tool_call>") + len("<tool_call>")
    return text[start : text.rindex("</tool_call>") + len("</tool_call>")]


def allowed(row: np.ndarray | None, token: int) -> bool:
    if row is None:
        return True
    return bool((int(row[token // 32]) >> (token % 32)) & 1)


def run_tokens(guide, ids: list[int]) -> int | None:
    """Feed ids, checking each against the mask; the index of the first token the
    mask rejects, else None."""
    for i, token in enumerate(ids):
        if not allowed(guide.mask(), token):
            return i
        guide.feed(token)
    return None


def call_guide(grammar):
    guide = grammar.guide()
    assert guide.feed(grammar.start_id)
    assert guide.constrained
    return guide


@pytest.mark.parametrize(
    "name,args",
    [
        ("Bash", {"command": "ls -la | head\nwc -l"}),
        ("Bash", {"command": "sleep 1", "timeout": 30, "background": False}),
        ("Bash", {"command": ""}),
        ("Bash", {"command": "echo 'a</b>'\n\n"}),
        ("Edit", {"file_path": "/a/b.py", "old_string": "x\n", "new_string": "y\n\n"}),
        ("Edit", {"file_path": "a.md", "mode": "all"}),
        (
            "Edit",
            {
                "file_path": "a",
                "todos": [{"content": "one", "done": True}, {"content": "two"}],
            },
        ),
        ("Ping", {}),
        ("Ping", {"anything": "goes"}),
        ("Bash", {"command": "grep '<tool_call>' a.py && echo '</tool_call>'"}),
    ],
)
def test_template_rendered_calls_are_accepted(xml_tok, xml_grammar, name, args):
    body = render(xml_tok, name, args)
    ids = xml_tok.encode(body, add_special_tokens=False)
    guide = call_guide(xml_grammar)
    assert run_tokens(guide, ids) is None
    assert guide.phase == tcg.FREE  # the closing marker hands generation back


def rejects_after(xml_tok, grammar, prefix: str, tail: str) -> bool:
    """True when the grammar rejects ``tail`` (as one continuation) after ``prefix``."""
    guide = call_guide(grammar)
    assert run_tokens(guide, xml_tok.encode(prefix, add_special_tokens=False)) is None
    tail_ids = xml_tok.encode(tail, add_special_tokens=False)
    return run_tokens(guide, tail_ids) is not None


def test_malformed_calls_are_blocked(xml_tok, xml_grammar):
    ok_prefix = "\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n"
    # JSON where the XML call belongs
    assert rejects_after(xml_tok, xml_grammar, "", '{"name": "Bash"')
    assert rejects_after(xml_tok, xml_grammar, "\n", '{"function": "Bash"')
    # the call must close with </function> before </tool_call>
    assert rejects_after(xml_tok, xml_grammar, ok_prefix, "</tool_call>")
    # names outside the request's tools
    assert rejects_after(xml_tok, xml_grammar, "\n<function=", "Read>")
    assert rejects_after(xml_tok, xml_grammar, "\n", "<Read>")
    # keys outside the tool's schema
    assert rejects_after(
        xml_tok, xml_grammar, "\n<function=Bash>\n", "<parameter=path>"
    )
    assert rejects_after(
        xml_tok, xml_grammar, "\n<function=Bash>\n", "<parameter=Edit>"
    )
    # typed values
    assert rejects_after(
        xml_tok, xml_grammar, "\n<function=Bash>\n<parameter=timeout>\n", "soon"
    )
    assert rejects_after(
        xml_tok, xml_grammar, "\n<function=Edit>\n<parameter=mode>\n", "some"
    )
    # the tool's own parameter keys are fine
    assert not rejects_after(
        xml_tok, xml_grammar, "\n<function=Bash>\n", "<parameter=timeout>\n5"
    )


@pytest.mark.parametrize("kind", ["integer", "number", "boolean", "array", "object"])
@pytest.mark.parametrize("nullable", [False, True])
def test_xml_null_requires_nullable_schema(xml_tok, kind, nullable):
    prop = {"type": [kind, "null"] if nullable else kind}
    tools = [
        {
            "name": "Probe",
            "parameters": {
                "type": "object",
                "properties": {"value": prop},
                "required": ["value"],
            },
        }
    ]
    grammar = tcg.compile_tool_grammar(tools, xml_tok, len(xml_tok) + 243)
    assert grammar is not None
    body = "\n<function=Probe>\n<parameter=value>\nnull\n</parameter>\n</function>\n</tool_call>"
    guide = call_guide(grammar)
    rejected = run_tokens(guide, xml_tok.encode(body, add_special_tokens=False))
    assert (rejected is None) is nullable


def test_only_the_call_can_end_with_the_end_marker_and_eos_needs_a_finished_call(
    xml_tok, xml_grammar
):
    guide = call_guide(xml_grammar)
    eos = xml_tok.eos_token_id
    assert not allowed(guide.mask(), eos)  # a call cannot be cut short by EOS
    ids = xml_tok.encode(render(xml_tok, "Ping", {}), add_special_tokens=False)
    assert run_tokens(guide, ids) is None
    assert guide.mask() is None  # free again: nothing is masked


def test_free_until_the_marker_and_back(xml_grammar):
    guide = xml_grammar.guide()
    assert guide.phase == tcg.FREE and guide.mask() is None
    assert guide.feed(1234) and guide.phase == tcg.FREE
    assert guide.feed(xml_grammar.start_id) and guide.phase == tcg.BODY
    assert guide.mask() is not None


def test_thinking_defers_the_marker(xml_grammar):
    guide = xml_grammar.guide(thinking_open=True)
    assert guide.phase == tcg.WAIT and guide.mask() is None
    # a marker inside the reasoning is text, not a call
    guide.feed(xml_grammar.start_id)
    assert guide.phase == tcg.WAIT
    guide.feed(xml_grammar.think_end_id)
    assert guide.phase == tcg.FREE
    guide.feed(xml_grammar.start_id)
    assert guide.phase == tcg.BODY


def test_forced_choice_starts_constrained_after_reasoning(xml_tok):
    grammar = tcg.compile_tool_grammar(
        TOOLS, xml_tok, len(xml_tok) + 243, tool_choice="required"
    )
    assert grammar.forced
    guide = grammar.guide()
    assert guide.constrained
    row = guide.mask()
    assert allowed(row, grammar.start_id)
    assert not allowed(row, xml_tok.eos_token_id)
    assert not allowed(row, xml_tok.encode("Sure", add_special_tokens=False)[0])
    think = grammar.guide(thinking_open=True)
    assert think.phase == tcg.WAIT
    think.feed(grammar.think_end_id)
    assert think.constrained


def test_forced_tool_only_offers_that_tool(xml_tok):
    grammar = tcg.compile_tool_grammar(
        TOOLS, xml_tok, len(xml_tok) + 243, tool_choice={"name": "Edit"}
    )
    guide = grammar.guide()
    ids = xml_tok.encode(
        render(xml_tok, "Edit", {"file_path": "a"}), add_special_tokens=False
    )
    assert run_tokens(guide, [grammar.start_id, *ids]) is None
    bad = grammar.guide()
    ids = xml_tok.encode(
        render(xml_tok, "Bash", {"command": "ls"}), add_special_tokens=False
    )
    assert run_tokens(bad, [grammar.start_id, *ids]) is not None


def test_plan_matches_token_by_token_masks_and_rewinds(xml_tok, xml_grammar):
    ids = xml_tok.encode(
        render(xml_tok, "Bash", {"command": "ls", "timeout": 5}),
        add_special_tokens=False,
    )
    guide = call_guide(xml_grammar)
    before = guide.checkpoint()
    window = 7
    plan = guide.plan(ids[: window - 1], window)
    assert guide.checkpoint()[0] == before[0] and guide.consumed == before[2]
    reference = call_guide(xml_grammar)
    for i in range(window):
        row = reference.mask()
        assert np.array_equal(plan[i], row)
        reference.feed(ids[i])
    # a draft the grammar rejects leaves the later positions unconstrained
    bad = guide.plan([xml_tok.encode("{", add_special_tokens=False)[0], 5, 5, 5], 5)
    assert not np.all(bad[0] == -1) and np.all(bad[2:] == -1)
    assert guide.consumed == before[2]


def test_advance_keeps_tokens_up_to_the_arming_one(xml_grammar):
    guide = xml_grammar.guide()
    tokens = [5, 6, xml_grammar.start_id, 7, 8]
    assert guide.advance(tokens) == 3
    assert guide.phase == tcg.BODY
    quiet = xml_grammar.guide()
    assert quiet.advance([5, 6, 7]) == 3 and quiet.phase == tcg.FREE


def test_rejected_token_never_wedges_the_request(xml_grammar):
    guide = call_guide(xml_grammar)
    assert guide.feed(5) is False  # not a valid start of a call
    assert guide.phase == tcg.FREE and guide.broken


def test_processor_masks_logits(xml_tok, xml_grammar):
    mx = pytest.importorskip("mlx.core")
    guide = call_guide(xml_grammar)
    proc = tcg.ToolCallProcessor(guide)
    vocab = xml_grammar.llt.vocab_size
    logits = mx.zeros((1, vocab))
    out = proc(mx.array([1]), logits)
    row = guide.mask()
    mask = np.array(mx.isfinite(out))[0]
    expected = np.array([allowed(row, i) for i in range(vocab)])
    assert np.array_equal(mask, expected)
    # free stretches leave the logits alone
    free = tcg.ToolCallProcessor(xml_grammar.guide())
    assert free(mx.array([1]), logits) is logits


def test_normalize_tool_choice():
    n = tcg.normalize_tool_choice
    assert n(None) is None and n("auto") is None and n({"type": "auto"}) is None
    assert n("required") == "required" and n({"type": "any"}) == "required"
    assert n("none") == "none" and n({"type": "none"}) == "none"
    assert n({"type": "tool", "name": "Bash"}) == {"name": "Bash"}
    assert n({"type": "function", "function": {"name": "Bash"}}) == {"name": "Bash"}


def test_tools_of_every_shape_normalize():
    flat = [{"name": "a", "input_schema": {"type": "object"}}]
    assert [t.name for t in tcg.normalize_tools(flat)] == ["a"]
    assert [t.name for t in tcg.normalize_tools(TOOLS)] == ["Bash", "Edit", "Ping"]
    assert tcg.grammar_key(TOOLS, None, True) != tcg.grammar_key(
        TOOLS, "required", True
    )


def test_no_grammar_without_a_call_format(tmp_path):
    class Plain:
        chat_template = "{{ messages }}"

    assert tcg.compile_tool_grammar(TOOLS, Plain(), 1000) is None
    assert tcg.compile_tool_grammar([], Plain(), 1000) is None


@pytest.fixture(scope="module")
def json_tok():
    if not JSON_TOKENIZER.exists():
        pytest.skip("Qwen2.5 tokenizer not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(JSON_TOKENIZER))


def test_json_style_calls(json_tok):
    grammar = tcg.compile_tool_grammar(TOOLS, json_tok, len(json_tok) + 10)
    assert grammar is not None and grammar.style == "json"
    good = (
        '\n{"name": "Bash", "arguments": {"command": "ls", "timeout": 5}}\n</tool_call>'
    )
    guide = grammar.guide()
    assert guide.feed(grammar.start_id)
    ids = json_tok.encode(good, add_special_tokens=False)
    assert run_tokens(guide, ids) is None
    assert guide.phase == tcg.FREE
    for bad in (
        '\n{"name": "Read", "arguments": {}}',
        '\n{"name": "Bash", "arguments": {"command": 5}',
        "\n<function=Bash>",
    ):
        g = grammar.guide()
        g.feed(grammar.start_id)
        assert run_tokens(g, json_tok.encode(bad, add_special_tokens=False)) is not None
    assert json.loads(good.split("\n")[1])["name"] == "Bash"


def _ws_tokens(tok):
    return [
        tok.encode(s, add_special_tokens=False)[0]
        for s in ("\n", "\n\n", " ")
        if len(tok.encode(s, add_special_tokens=False)) == 1
    ]


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("which", ["xml", "json"])
def test_forced_reply_cannot_loop_on_whitespace(request, which, parallel):
    """A forced reply may not stall in blanks: EOS is masked until a call closes, so unbounded
    whitespace let one sampled newline run to max_tokens with no call (M3 sweep, chat stream)."""
    tok = request.getfixturevalue(f"{which}_tok")
    grammar = tcg.compile_tool_grammar(
        TOOLS, tok, len(tok) + 243, tool_choice="required", parallel=parallel
    )
    assert grammar is not None
    ws = _ws_tokens(tok)
    assert ws
    guide = grammar.guide()
    # two characters of blank are allowed, a third never is, and the call marker always is
    for _ in range(2):
        assert allowed(guide.mask(), grammar.start_id)
        w = ws[0]
        assert allowed(guide.mask(), w)
        guide.feed(w)
        if not allowed(guide.mask(), w):
            break
    for t in ws:
        assert not allowed(guide.mask(), t)
    assert allowed(guide.mask(), grammar.start_id)
    assert not allowed(guide.mask(), tok.eos_token_id)


def test_json_style_inlines_local_refs():
    shared = {
        "type": "object",
        "properties": {"p": {"$ref": "#/$defs/Pt"}},
        "required": ["p"],
        "$defs": {"Pt": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    }
    out = tcg._inline_refs(shared)
    assert out["properties"]["p"] == {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }
    assert "$defs" not in out
    loop = {
        "properties": {"n": {"$ref": "#/$defs/N"}},
        "$defs": {"N": {"properties": {"n": {"$ref": "#/$defs/N"}}}},
    }
    assert tcg._inline_refs(loop) is None
    assert tcg._inline_refs({"properties": {"a": {"$ref": "http://x/y"}}}) is None
    spec = tcg.ToolSpec("T", shared)
    assert tcg.build_json_grammar([spec], start_id=1, end_id=2) is not None


def test_forced_reply_cannot_end_inside_its_reasoning(request):
    """A forced reply that thinks first may not stop (EOS) before </think>: Qwen3.5-0.8B ended its
    reasoning with end-of-turn and the reply carried no call (M3, Responses stream, 1 in 80)."""
    tok = request.getfixturevalue("xml_tok")
    grammar = tcg.compile_tool_grammar(
        TOOLS, tok, len(tok) + 243, tool_choice="required"
    )
    assert grammar is not None and grammar.think_end_id is not None
    assert tok.eos_token_id in grammar.eos_ids
    guide = grammar.guide(thinking_open=True)
    assert guide.phase == tcg.WAIT and guide.constrained
    row = guide.mask()
    assert row is not None
    for eos in grammar.eos_ids:
        assert not allowed(row, eos)
    word = tok.encode("Sure", add_special_tokens=False)[0]
    assert allowed(row, word) and allowed(row, grammar.think_end_id)
    assert guide.plan([word, grammar.think_end_id], 3) is not None
    guide.feed(grammar.think_end_id)
    assert guide.phase == tcg.BODY
    # not forced: reasoning stays unmasked
    auto = tcg.compile_tool_grammar(TOOLS, tok, len(tok) + 243)
    free = auto.guide(thinking_open=True)
    assert not free.constrained and free.mask() is None
