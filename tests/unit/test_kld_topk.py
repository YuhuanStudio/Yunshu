"""The kld capture's two-stage top-K returns exactly what the full-vocabulary sort returned (ids, values,
tie order), so every KLD / top-1 / top-5 number is unchanged."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.verify.kld import capture, metrics  # noqa: E402

V = 248320  # Qwen3.5 / Flash-Next vocabulary: not a multiple of the group (padding path)


def _full_sort_top_k(logits, k=capture.K):
    """The capture's top-K before the two-stage version (argpartition over the whole vocabulary)."""
    vocab = logits.shape[-1]
    idx = mx.argpartition(logits, kth=vocab - k, axis=-1)[:, vocab - k :]
    vals = mx.take_along_axis(logits, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(
        vals, order, axis=-1
    )


def _lexsort_top_k(logits, k=capture.K):
    """The definition: the last k under ascending (value, id), then value descending, ties id ascending."""
    ids, vals = [], []
    for row in np.array(logits):
        top = np.lexsort((np.arange(row.size), row))[-k:]
        order = np.argsort(-row[top], kind="stable")
        ids.append(top[order])
        vals.append(row[top][order])
    return np.array(ids), np.array(vals)


def _logits(seed, kind):
    mx.random.seed(seed)
    x = mx.random.normal((16, V)) * 3.0
    if kind == "bf16":  # the head's own dtype: many exact ties among 248K values
        x = x.astype(mx.bfloat16)
    elif kind == "ties":  # integer logits: ties at every rank, including the k-th
        x = mx.round(x)
    elif kind == "boundary":  # 40 ids tie at the top: id alone picks the 32 kept
        x = mx.concatenate([x[:, :-40], mx.full((16, 40), 50.0)], axis=1)
    x = x.astype(mx.float32)
    mx.eval(x)
    return x


@pytest.mark.parametrize("kind", ["fp32", "bf16", "ties", "boundary"])
@pytest.mark.parametrize("group", [capture.GROUP, 1024, 4096])
def test_two_stage_top_k_is_the_full_sort(kind, group):
    x = _logits(7, kind)
    old_ids, old_vals = _full_sort_top_k(x)
    new_ids, new_vals = capture.top_k(x, group=group)
    mx.eval(old_ids, old_vals, new_ids, new_vals)
    assert np.array_equal(np.array(old_ids), np.array(new_ids))
    assert np.array_equal(np.array(old_vals), np.array(new_vals))
    want_ids, want_vals = _lexsort_top_k(x)
    assert np.array_equal(np.array(new_ids), want_ids)
    assert np.array_equal(np.array(new_vals), want_vals)


def test_kld_metrics_unchanged():
    """Records from the old and new top-K give the same metric summary, bit for bit."""

    def window(x, select):
        lse = mx.logsumexp(x, axis=-1, keepdims=True)
        ids, vals = select(x)
        lp = vals - lse
        mx.eval(ids, lp)
        n = x.shape[0]
        return metrics.Window(
            np.array(ids).astype(np.int32),
            np.array(lp).astype(np.float32),
            np.full(n, -1.0, dtype=np.float32),
            np.zeros(n, dtype=np.int32),
        )

    ref = {"w": window(_logits(1, "bf16"), _full_sort_top_k)}
    arm_x = _logits(2, "bf16") * 0.3 + _logits(1, "bf16")
    old = metrics.summarize(ref, {"w": window(arm_x, _full_sort_top_k)})
    new = metrics.summarize(
        {"w": window(_logits(1, "bf16"), capture.top_k)},
        {"w": window(arm_x, capture.top_k)},
    )
    assert old == new
    assert old["kld_mean"] > 0


def test_flashnext_head_calls_the_head_in_small_row_chunks():
    """The Flash-Next head sees at most HEAD_ROWS rows per call (130 rows: a ragged last chunk) and the
    concatenated logits are those of one call over every row."""
    from types import SimpleNamespace

    mx.random.seed(3)
    w = mx.random.normal((64, 40))
    seen = []

    def logits(lm, h):
        seen.append(int(h.shape[0]))
        return h @ w.T

    hidden = mx.random.normal((1, 130, 40))
    got = capture.flashnext_head(SimpleNamespace(logits=logits), None, hidden)
    want = (hidden[0] @ w.T)[None]
    assert got.shape == (1, 130, 64)
    assert max(seen) == capture.HEAD_ROWS and sum(seen) == 130
    assert np.allclose(np.array(got), np.array(want), atol=1e-5)
