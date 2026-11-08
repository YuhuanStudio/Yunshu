"""Processors applied one window row at a time (round_driver.driver.window_processed) give each
row bit for bit what the serial step that draws it gives (spec on == spec off)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.sample_utils import make_logits_processors  # noqa: E402

from yunshu_engine import vlm_batch_runner as vbr  # noqa: E402
from yunshu_engine.round_driver.driver import (  # noqa: E402
    RoundDriver,
    window_processed,
)
from yunshu_engine.text_logit_processors import GeneratedTokenPenalty  # noqa: E402

V = 64
PROMPT = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
]
EOS = [2]


def _rep():
    return make_logits_processors(repetition_context_size=20, repetition_penalty=1.3)


def _mask(**kw):
    return vbr.TokenMaskProcessor(eos_ids=EOS, **kw)


SETS = {
    "repetition": lambda: _rep(),
    "min_tokens": lambda: [_mask(min_tokens=6)],
    "top_n_sigma": lambda: [_mask(top_n_sigma=0.8)],
    "ban+min_tokens": lambda: [_mask(suppress=[1, 3], min_tokens=6)],
    "bias+rep": lambda: [vbr.LogitBiasProcessor({4: 2.0, 9: -3.0}), *_rep()],
    "penalty+rep": lambda: [GeneratedTokenPenalty(0.3, 0.7), *_rep()],
    "all": lambda: [
        vbr.LogitBiasProcessor({4: 2.0}),
        *_rep(),
        GeneratedTokenPenalty(0.2, 0.9),
        _mask(suppress=[1], min_tokens=6, top_n_sigma=1.2),
    ],
}


def _serial(make, gen, logits_row):
    """The serial pipeline (RoundDriver._processed, as a plain row runs it) for the draw whose
    context is PROMPT + gen; earlier draws run on zero logits so stateful processors count."""
    procs = make()
    ctx = list(PROMPT)
    out = None
    for k in range(len(gen) + 1):
        row = SimpleNamespace(req=SimpleNamespace(processors=procs), context=list(ctx))
        lg = logits_row if k == len(gen) else mx.zeros((1, V), dtype=logits_row.dtype)
        out = RoundDriver._processed(None, row, lg, first=(k == 0))
        if k < len(gen):
            ctx.append(gen[k])
    return out


def _window(make, gen, tokens, parents, logits):
    procs = make()
    for p in procs:
        if hasattr(p, "window_row"):
            p._counts(V)
            if len(gen) > 1:
                p.commit_tokens(gen[:-1])  # committed tokens before the window root
    return window_processed(procs, logits, PROMPT + gen, len(gen), tokens, parents)


def _path(tokens, parents, j):
    out, at = [], j
    while at >= 0:
        out.append(tokens[at])
        at = parents[at]
    return out[::-1]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("gen_len", [1, 3, 5])
@pytest.mark.parametrize("name", sorted(SETS))
def test_chain_window_rows_equal_serial(name, gen_len, dtype):
    make = SETS[name]
    mx.random.seed(11)
    gen = [30 + i for i in range(gen_len)]
    drafts = [40, 41, 42, 43]
    tokens = [gen[-1], *drafts]
    logits = (mx.random.normal((len(tokens), V)) * 3).astype(dtype)
    got = _window(make, gen, tokens, None, logits)
    for j in range(len(tokens)):
        want = _serial(make, gen + drafts[:j], logits[j : j + 1])
        assert mx.array_equal(got[j : j + 1], want), (name, gen_len, j)


@pytest.mark.parametrize("name", sorted(SETS))
def test_tree_window_rows_equal_serial_along_their_path(name):
    make = SETS[name]
    mx.random.seed(5)
    gen = [30, 31, 32]
    tokens = [gen[-1], 40, 41, 42, 43, 44]
    parents = [-1, 0, 0, 1, 1, 2]
    logits = mx.random.normal((len(tokens), V)) * 3
    got = _window(make, gen, tokens, parents, logits)
    for j in range(len(tokens)):
        path = _path(tokens, parents, j)
        want = _serial(make, gen + path[1:], logits[j : j + 1])
        assert mx.array_equal(got[j : j + 1], want), (name, j)


@pytest.mark.parametrize("name", sorted(SETS))
def test_first_token_row_equals_serial(name):
    make = SETS[name]
    mx.random.seed(3)
    logits = mx.random.normal((1, V)) * 3
    procs = make()
    got = window_processed(procs, logits, PROMPT, 0)
    want = _serial(make, [], logits)
    assert mx.array_equal(got, want)


def test_min_tokens_flips_inside_a_window():
    # min_tokens 6, 3 tokens generated: rows at depth 0..2 still ban EOS, depth 3+ allow it
    make = SETS["min_tokens"]
    gen = [30, 31, 32]
    tokens = [32, 40, 41, 42, 43]
    logits = mx.zeros((5, V))
    got = window_processed(make(), logits, PROMPT + gen, 3, tokens, None)
    banned = [bool(mx.isinf(got[j, EOS[0]]).item()) for j in range(5)]
    assert banned == [True, True, True, False, False]


@pytest.mark.parametrize("name", sorted(SETS))
def test_tree_report_rows_equal_the_serial_report(name):
    """DFlash tree rows: processors in the logits dtype, then the fp32 report
    (keyed_sampling.tree_target_rows with logprobs) == the plain row's report."""
    from yunshu_engine import keyed_sampling

    make = SETS[name]
    mx.random.seed(9)
    gen = [30, 31, 32]
    tokens = [gen[-1], 40, 41, 42, 43, 44]
    parents = [-1, 0, 0, 1, 1, 2]
    logits = (mx.random.normal((len(tokens), V)) * 3).astype(mx.bfloat16)
    drawn = _window(make, gen, tokens, parents, logits)[None]
    target, probs = keyed_sampling.tree_target_rows(
        drawn, None, len(gen), None, SimpleNamespace(logprobs=True), None, None
    )
    for j in range(len(tokens)):
        path = _path(tokens, parents, j)
        x = _serial(make, gen + path[1:], logits[j : j + 1]).astype(mx.float32)
        lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
        assert mx.array_equal(probs[j : j + 1], lp), (name, j)
        assert int(target[j].item()) == int(mx.argmax(lp, axis=-1)[0].item())
