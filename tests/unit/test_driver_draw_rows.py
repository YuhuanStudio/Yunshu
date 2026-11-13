"""RoundDriver._draw on an n-row window (an MTP chain: guide plan, processors, logprobs report)
gives every row what n plain single-row steps give (spec on == spec off), bit for bit."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("llguidance")

sys.path.insert(0, str(Path(__file__).parent))
from test_driver_window_processors import PROMPT, SETS, V  # noqa: E402

from yunshu_engine import constrained_spec as cs  # noqa: E402
from yunshu_engine.round_driver.driver import (  # noqa: E402
    RoundDriver,
    _Item,
    _PerRow,
    _Row,
)


class Automaton:
    """Allows the tokens t with t % 3 == state % 3; every fed token advances the state."""

    words = (V + 31) // 32
    constrained = True

    def __init__(self):
        self.state = 0

    def mask(self):
        bits = np.zeros(self.words * 32, dtype=np.uint8)
        bits[[t for t in range(V) if t % 3 == self.state % 3]] = 1
        return np.packbits(bits, bitorder="little").view(np.int32)

    def checkpoint(self):
        return self.state

    def restore(self, saved):
        self.state = saved

    def feed(self, token):
        self.state += 1
        return True

    def arms(self, token):
        return False

    plan = cs.ConstraintGuide.plan


def _driver():
    d = object.__new__(RoundDriver)
    d.dflash = None
    d.lm = None
    return d


def _row(make, guide, logprobs, gen, generic):
    procs = make()
    row = _Row(
        req=SimpleNamespace(
            guide=guide,
            processors=procs,
            logprobs=logprobs,
            top_logprobs=0,
            sampling=None,
            ids=list(PROMPT),
        ),
        cache=None,
        mtp_cache=None,
        pending=gen[-1] if gen else None,
        generated=len(gen),
        context=list(PROMPT) + list(gen),
        generic=generic,
    )
    if generic:  # a drafting row keeps the penalty to count what its windows commit
        for p in procs:
            if hasattr(p, "window_row"):
                row.penalty = p
                p._counts(V)
                if len(gen) > 1:
                    p.commit_tokens(gen[:-1])
    return row


def _plain_steps(make, use_guide, logprobs, gen, drafts, logits):
    """n plain single-row steps (what spec off does): returns tokens and reported logprobs."""
    guide = Automaton() if use_guide else None
    row = _row(make, guide, logprobs, [], False)
    drv = _driver()
    toks, lps = [], []
    seq = list(gen) + list(drafts)
    for k in range(len(gen) + len(drafts) + 1):
        final = k >= len(gen)
        lg = (
            logits[k - len(gen) : k - len(gen) + 1]
            if final
            else mx.zeros((1, V), dtype=logits.dtype)
        )
        kind = "p" if k == 0 else "d"
        item = _Item(kind, row, None, len(PROMPT) - 1, len(PROMPT), 0, 1)
        ((_, tok, extra),) = drv._draw([item], None, target_logits=lg)
        if final:
            toks.append(int(tok.tolist()[0]))
            lps.append(None if extra is None else float(extra[0].item()))
        if k < len(seq):
            t = seq[k]
            row.context.append(t)
            row.generated += 1
            row.pending = t
            if guide is not None:
                guide.feed(t)
    return toks, lps


@pytest.mark.parametrize("logprobs", [False, True])
@pytest.mark.parametrize("use_guide", [False, True])
@pytest.mark.parametrize("name", ["none", *sorted(SETS)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_window_draw_equals_plain_steps(name, use_guide, logprobs, dtype):
    make = (lambda: []) if name == "none" else SETS[name]
    if name == "none" and not (use_guide or logprobs):
        pytest.skip("nothing to apply")
    mx.random.seed(7)
    gen = [30, 31, 32]
    s0 = len(gen)
    drafts = []
    for j in range(4):  # legal at row j: t % 3 == (s0 + j) % 3
        drafts.append(next(t for t in range(40, 80) if t % 3 == (s0 + j) % 3))
    if not use_guide:
        drafts = [40, 41, 42, 43]
    n = len(drafts) + 1
    logits = (mx.random.normal((n, V)) * 3).astype(dtype)

    want_tok, want_lp = _plain_steps(make, use_guide, logprobs, gen, drafts, logits)

    guide = Automaton() if use_guide else None
    for t in gen:
        if guide is not None:
            guide.feed(t)
    row = _row(make, guide, logprobs, gen, generic=name != "none")
    window = mx.array([gen[-1], *drafts], dtype=mx.int32)
    item = _Item("d", row, None, 0, 0, 0, n, tokens=window)
    ((_, tok, extra),) = _driver()._draw([item], None, target_logits=logits)
    assert [int(t) for t in tok.tolist()] == want_tok
    if logprobs:
        assert isinstance(extra, _PerRow)
        got = [float(e[0].item()) for e in extra]
        assert got == want_lp
    else:
        assert extra is None


@pytest.mark.parametrize("name", sorted(SETS))
def test_window_draw_computes_greedy_once(monkeypatch, name):
    """A greedy window row takes one argmax after its last logits change (was one per change)."""
    real = mx.argmax
    calls = []
    monkeypatch.setattr(mx, "argmax", lambda *a, **k: calls.append(1) or real(*a, **k))
    drafts = [next(t for t in range(40, 80) if t % 3 == (3 + j) % 3) for j in range(4)]
    gen = [30, 31, 32]
    logits = mx.random.normal((len(drafts) + 1, V)) * 3
    guide = Automaton()
    for t in gen:
        guide.feed(t)
    row = _row(SETS[name], guide, False, gen, generic=True)
    window = mx.array([gen[-1], *drafts], dtype=mx.int32)
    item = _Item("d", row, None, 0, 0, 0, len(drafts) + 1, tokens=window)
    calls.clear()
    _driver()._draw([item], None, target_logits=logits)
    assert len(calls) == 1
