"""The speculative lane under a tool-call guide emits exactly what token-by-token
masked decoding emits, whatever the drafts are.

The lane's rounds run on a toy target (fake verify: the logits at generated index
``j`` are fixed random rows biased toward a script) with a byte-level toy
tokenizer, so the whole path (drafts, per-position masks, cutting a round at the
tool-call marker, rollback bookkeeping) runs without a model.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("llguidance")
tokenizers = pytest.importorskip("tokenizers")

from yunshu_engine import mtp_lane  # noqa: E402
from yunshu_engine import tool_call_grammar as tcg  # noqa: E402

SPECIALS = [
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
    "<eos>",
]


def byte_level_alphabet() -> dict[str, int]:
    """GPT-2's byte -> printable char map, so each byte is one token."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): i for i, c in enumerate(cs[i] for i in range(len(bs)))}


@pytest.fixture(scope="module")
def toy():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = byte_level_alphabet()
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    from tokenizers import AddedToken

    tok.add_special_tokens(
        [AddedToken(s, special=False, normalized=False) for s in SPECIALS[:-1]]
    )
    tok.add_special_tokens([AddedToken(SPECIALS[-1], special=True, normalized=False)])
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    return hf


def ids_of(hf, text: str) -> list[int]:
    out: list[int] = []
    i = 0
    marks = {s: hf.convert_tokens_to_ids(s) for s in SPECIALS}
    while i < len(text):
        for s, tid in marks.items():
            if text.startswith(s, i):
                out.append(tid)
                i += len(s)
                break
        else:
            out.extend(hf.encode(text[i], add_special_tokens=False))
            i += 1
    return out


def make_grammar(hf, forced=False):
    vocab = len(hf) + 4
    llt = tcg.llg_tokenizer(hf, vocab)
    tools = tcg.normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "ab",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "k": {"type": "string"},
                            "n": {"type": "integer"},
                        },
                    },
                },
            }
        ]
    )
    start = hf.convert_tokens_to_ids("<tool_call>")
    end = hf.convert_tokens_to_ids("</tool_call>")
    lark = tcg.build_xml_grammar(
        tools,
        start_id=start,
        end_id=end,
        text_specials=[hf.convert_tokens_to_ids(s) for s in SPECIALS[:6]],
        validate_json=lambda schema: True,
        forced=forced,
    )
    return tcg.ToolGrammar(
        lark,
        llt,
        start_id=start,
        end_id=end,
        think_end_id=hf.convert_tokens_to_ids("</think>"),
        forced=forced,
        style="xml",
    )


class ToyTarget:
    """Logits after generated index j: fixed noise + a bias toward ``script[j + 1]``."""

    def __init__(self, script: list[int], vocab: int, eos: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rows = rng.normal(size=(len(script) + 64, vocab)).astype(np.float32)
        for j in range(len(self.rows)):
            nxt = script[j + 1] if j + 1 < len(script) else eos
            self.rows[j, nxt] += 20.0
        self.vocab = vocab

    def logits(self, j: int) -> np.ndarray:
        return self.rows[j]


def reference(target: ToyTarget, guide, first: int, eos: int, limit: int) -> list[int]:
    """Plain token-by-token masked greedy decode."""
    out = [first]
    guide.feed(first)
    while len(out) < limit and out[-1] != eos:
        logits = mx.array(target.logits(len(out) - 1))[None]
        row = guide.mask()
        if row is not None:
            logits = tcg.apply_bitmask(logits, row)
        tok = int(mx.argmax(logits, axis=-1).item())
        out.append(tok)
        guide.feed(tok)
    return out


class FakeVerify:
    def __init__(self, hidden, state, bs):
        self.hidden = hidden
        self.target_tokens = None
        self.rollback_state = None
        self._state, self._bs = state, bs

    def commit(self, lm, cache, accepted, bs):
        self._state["j"] += accepted + 1

    def abort(self):
        pass


class FakeHead:
    """Draft head: ``readout(hidden at generated index k)`` guesses the token at k+2
    (the target's own choice with probability ``accuracy``, else a random token)."""

    def __init__(self, target: ToyTarget, accuracy: float, seed: int):
        self.t, self.acc, self.rng = target, accuracy, random.Random(seed)
        self._cache = []
        self.accept_lens = []
        self._next_position = 0
        self._seed_token = self._seed_hidden = None
        self.supports_greedy_draft_argmax = True

    def reset(self, model):
        pass

    def _guess(self, k: int) -> int:
        if self.rng.random() < self.acc:
            return int(np.argmax(self.t.logits(k + 1)))
        return self.rng.randrange(self.t.vocab)

    def _greedy_token(self, hidden):
        idx = np.array(mx.argmax(hidden, axis=-1)).reshape(-1)
        return mx.array([[self._guess(int(k)) for k in idx]], dtype=mx.int32)

    def _forward_token(self, tok, h, dtype):
        k = int(mx.argmax(h, axis=-1).item())
        return onehot(k + 1, self.t.vocab)

    def _forward_tokens(self, target, hidden, dtype):
        return hidden + 0  # row i already is the hidden after target[i]


def onehot(k: int, n: int) -> mx.array:
    v = np.zeros((1, 1, n), np.float32)
    v[0, 0, min(k, n - 1)] = 1.0
    return mx.array(v)


def run_lane(monkeypatch, target, guide, first, eos, limit, accuracy, seed, block):
    import mlx_vlm.speculative.mtp as mtp

    vocab = target.vocab
    state = {"j": 0}
    lm = type("LM", (), {})()

    def project(hidden):
        idx = np.array(mx.argmax(hidden, axis=-1)).reshape(-1)
        return mx.array(np.stack([target.logits(int(k)) for k in idx]))[None]

    lm.speculative_logits_from_hidden = project

    def fake_verify(lm_, verify_input, cache, sampler, *, sample_target_tokens=True):
        bs = int(verify_input.shape[1])
        # hidden row i = one-hot of the generated index whose logits it carries
        rows = np.zeros((1, bs, vocab), np.float32)
        for i in range(bs):
            rows[0, i, state["j"] + i] = 1.0
        hidden = mx.array(rows)
        res = FakeVerify(hidden, state, bs)
        if sample_target_tokens:
            res.target_tokens = mx.argmax(project(hidden), axis=-1)
        return res

    monkeypatch.setattr(mtp, "_mtp_verify_target", fake_verify)
    head = FakeHead(target, accuracy, seed)
    hidden = onehot(0, vocab)
    out = [first]
    for toks, _meta in mtp_lane.rounds(
        lm,
        head,
        [],
        hidden,
        prompt_tokens=None,
        first_bonus=first,
        max_tokens=limit,
        sampler=None,
        draft_block_size=block,
        token_dtype=mx.int32,
        stop_check=None,
        eos_token_ids={eos},
        guide=guide,
    ):
        out.extend(toks)
    return out


CALL = "\n<function=ab>\n<parameter=k>\nhello\n</parameter>\n<parameter=n>\n42\n</parameter>\n</function>\n<tool_call_end>"


def script_ids(hf, text: str) -> list[int]:
    return ids_of(hf, text.replace("<tool_call_end>", "</tool_call>"))


@pytest.mark.parametrize("accuracy", [1.0, 0.7, 0.0])
@pytest.mark.parametrize("block", [4, 7])
def test_valid_call_is_reproduced_and_matches_masked_decode(
    toy, monkeypatch, accuracy, block
):
    hf = toy
    grammar = make_grammar(hf)
    eos = hf.eos_token_id
    start = grammar.start_id
    script = ids_of(hf, "Sure. ") + [start] + script_ids(hf, CALL) + [eos]
    target = ToyTarget(script, len(hf) + 4, eos)
    ref = reference(target, grammar.guide(), script[0], eos, 200)
    assert ref == script  # the grammar leaves a valid call alone
    got = run_lane(
        monkeypatch, target, grammar.guide(), script[0], eos, 200, accuracy, 3, block
    )
    assert got == ref


@pytest.mark.parametrize("accuracy", [1.0, 0.6, 0.0])
@pytest.mark.parametrize("block", [4, 7])
def test_malformed_call_is_steered_identically(toy, monkeypatch, accuracy, block):
    hf = toy
    grammar = make_grammar(hf)
    eos = hf.eos_token_id
    start = grammar.start_id
    bad = (
        ids_of(hf, "Let me. ")
        + [start]
        + ids_of(hf, '\n{"name": "ab", "arguments": {}}')
        + [eos]
    )
    target = ToyTarget(bad, len(hf) + 4, eos, seed=5)
    ref = reference(target, grammar.guide(), bad[0], eos, 120)
    assert ref != bad  # the JSON attempt was blocked
    got = run_lane(
        monkeypatch, target, grammar.guide(), bad[0], eos, 120, accuracy, 9, block
    )
    assert got == ref
    text = hf.decode(ref[ref.index(start) + 1 :])
    assert text.lstrip().startswith("<function=ab>")


def test_two_calls_and_text_between(toy, monkeypatch):
    hf = toy
    grammar = make_grammar(hf)
    eos = hf.eos_token_id
    start = grammar.start_id
    call = script_ids(hf, CALL)
    script = (
        ids_of(hf, "a ")
        + [start]
        + call
        + ids_of(hf, "\nnow ")
        + [start]
        + call
        + [eos]
    )
    target = ToyTarget(script, len(hf) + 4, eos, seed=2)
    ref = reference(target, grammar.guide(), script[0], eos, 400)
    assert ref == script
    for accuracy in (1.0, 0.5):
        got = run_lane(
            monkeypatch, target, grammar.guide(), script[0], eos, 400, accuracy, 4, 5
        )
        assert got == ref


def test_marker_as_first_token_and_reasoning_gate(toy, monkeypatch):
    hf = toy
    grammar = make_grammar(hf)
    eos = hf.eos_token_id
    start = grammar.start_id
    script = [start] + script_ids(hf, CALL) + [eos]
    target = ToyTarget(script, len(hf) + 4, eos)
    got = run_lane(monkeypatch, target, grammar.guide(), start, eos, 200, 0.8, 1, 5)
    assert got == script
    # inside reasoning a marker is text: nothing is masked until </think>
    think_end = grammar.think_end_id
    text = [start] + ids_of(hf, "{junk}") + [think_end] + ids_of(hf, "done") + [eos]
    target = ToyTarget(text, len(hf) + 4, eos, seed=8)
    ref = reference(target, grammar.guide(thinking_open=True), text[0], eos, 100)
    assert ref == text
    got = run_lane(
        monkeypatch,
        target,
        grammar.guide(thinking_open=True),
        text[0],
        eos,
        100,
        0.5,
        2,
        4,
    )
    assert got == ref


def test_lane_masks_are_engaged_and_rounds_are_cut_at_the_marker(toy, monkeypatch):
    hf = toy
    grammar = make_grammar(hf)
    eos = hf.eos_token_id
    start = grammar.start_id
    script = ids_of(hf, "xy ") + [start] + script_ids(hf, CALL) + [eos]
    target = ToyTarget(script, len(hf) + 4, eos)
    guide = grammar.guide()
    got = run_lane(monkeypatch, target, guide, script[0], eos, 200, 1.0, 1, 7)
    assert got == script
    assert guide.lane_rounds > 0  # the constrained stretch went through masked verify
    assert guide.calls == 1 and guide.phase == tcg.FREE


def test_can_guide_needs_the_lane_installed():
    class Head:
        supports_greedy_draft_argmax = True

    assert not mtp_lane.can_guide(object())
    mtp_lane._STATE["installed"], was = True, mtp_lane._STATE["installed"]
    try:
        assert mtp_lane.can_guide(Head())
        assert not mtp_lane.can_guide(object())
    finally:
        mtp_lane._STATE["installed"] = was
