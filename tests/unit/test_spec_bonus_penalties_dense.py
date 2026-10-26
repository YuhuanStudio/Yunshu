"""The speculative bonus-token penalties (n-gram / speculative decoder) apply one dense vector per
penalty, not 2-3 tiny MLX ops per distinct generated token (port of 4634f98 to engine_sampling),
with the same arithmetic as the old per-id loops."""

from __future__ import annotations

import mlx.core as mx
import pytest

from yunshu_engine.engine_sampling import _apply_spec_bonus_penalties

V = 64
HISTORY = [3, 5, 5, 9, 9, 9, 12] + [20 + (i % 7) for i in range(40)]
N_PROMPT = 2


def _reference(logits, hist, n_prompt, fp, pp, bias):
    out = logits
    gen = hist[n_prompt:]
    counts = {}
    for t in gen:
        counts[t] = counts.get(t, 0) + 1
    for tid, cnt in counts.items():
        if fp != 0.0:
            out[..., tid] = out[..., tid] - fp * cnt
        if pp != 0.0 and cnt > 0:
            out[..., tid] = out[..., tid] - pp
    for tid, b in (bias or {}).items():
        if 0 <= tid < out.shape[-1]:
            out[..., tid] = out[..., tid] + b
    return out


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", [(V,), (1, V)])
@pytest.mark.parametrize(
    "fp,pp,bias",
    [(0.5, 0.0, None), (0.0, -0.7, None), (0.4, 0.3, {3: 1.5, 99: 2.0, -1: 1.0})],
)
def test_dense_matches_the_per_id_loops(dtype, shape, fp, pp, bias):
    base = (mx.random.normal(shape) * 3).astype(dtype)
    mx.eval(base)
    want = _reference(mx.array(base), HISTORY, N_PROMPT, fp, pp, bias)
    got = _apply_spec_bonus_penalties(
        mx.array(base),
        HISTORY,
        N_PROMPT,
        frequency_penalty=fp,
        presence_penalty=pp,
        logit_bias=bias,
    )
    assert got.shape == base.shape and got.dtype == dtype
    assert mx.array_equal(got, want).item()


def test_no_penalty_is_untouched():
    base = mx.random.normal((V,))
    got = _apply_spec_bonus_penalties(mx.array(base), HISTORY, N_PROMPT)
    assert mx.array_equal(got, base).item()
