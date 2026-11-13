"""Flash-Next packed rows: the QSA projections shared over a pack (``flashnext.qsa_rows``) and the MTP head's
one LM-head pass give each row the bits of its own call."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
ql = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from mlx_vlm.speculative.cache_state import start_speculative_cache  # noqa: E402

from yunshu_engine.round_driver import flashnext  # noqa: E402

PROMPTS = [
    [1 + (5 * i) % 60 for i in range(18)],
    [3 + (7 * i) % 58 for i in range(23)],
    [2 + (3 * i) % 61 for i in range(30)],
]


@pytest.fixture(scope="module")
def lm():
    from scripts.research.phaseb_verify import native_tiny

    return native_tiny()


def _caches(lm):
    out = []
    for ids in PROMPTS:
        c = lm.make_cache()
        lm(mx.array([ids]), cache=c)
        mx.eval([e.state for e in c if hasattr(e, "state")])
        out.append(c)
    return out


def _attention_index(lm):
    return next(i for i, layer in enumerate(lm.model.layers) if not layer.is_linear)


def _run(lm, caches, mixed, spans, packed):
    fwd = ql._QWEN4_BATCH_INVARIANT_FORWARD
    model = lm.model
    index = _attention_index(lm)
    attn = model.layers[index].self_attn
    txns = [
        start_speculative_cache(c, e - s)
        for c, (s, e) in zip(caches, spans, strict=True)
    ]
    try:
        positions, masks = [], []
        for c, (s, e) in zip(caches, spans, strict=True):
            off = int(c[model.fa_idx].offset)
            positions.append((off + mx.arange(e - s, dtype=mx.int64))[None])
            masks.append(
                ql._create_qwen3_5_attention_mask(mixed[:, s:e], c[model.fa_idx])
            )
        rows = [c[index] for c in caches]
        if packed:
            parts = flashnext.qsa_rows(fwd, attn, mixed, spans, rows, positions, masks)
        else:
            parts = [
                fwd._qsa_attention(attn, mixed[:, s:e], c, p, m)
                for (s, e), c, p, m in zip(spans, rows, positions, masks, strict=True)
            ]
        out = mx.concatenate(parts, axis=1)
        mx.eval(out)
        return out
    finally:
        for t in txns:
            t.abort()


@pytest.mark.parametrize("widths", [[2, 2], [2, 2, 2], [2, 1, 2]])
def test_qsa_rows_equal_each_row_alone(lm, widths, monkeypatch):
    caches = _caches(lm)[: len(widths)]
    spans, at = [], 0
    for w in widths:
        spans.append((at, at + w))
        at += w
    dims = lm.model.args.hidden_size
    mixed = (mx.random.normal((1, at, dims), key=mx.random.key(9)) * 0.5).astype(
        mx.bfloat16
    )
    solo = _run(lm, caches, mixed, spans, packed=False)
    cls = type(ql._QWEN4_BATCH_INVARIANT_FORWARD)
    seen = []
    original = cls._linears

    def spy(self, lins, x):
        seen.append(tuple(x.shape[:2]))
        return original(self, lins, x)

    monkeypatch.setattr(cls, "_linears", spy)
    packed = _run(lm, caches, mixed, spans, packed=True)
    assert mx.array_equal(solo, packed).item()
    if len(set(widths)) == 1:
        # the q/k/v projection ran once, with the rows as the batch axis
        assert seen == [(len(widths), widths[0])]
    else:
        assert all(rows == 1 for rows, _ in seen)  # mixed widths keep the per-row path


def test_absorb_window_reads_every_row_in_one_head_pass(monkeypatch):
    calls = []

    class Head(flashnext.FlashNextHead):
        def __init__(self):
            self.caches = {}
            self._fused = False  # the per-row head layer (no fused kernels here)

        def _sync(self, cache, keep):
            pass

        def _step(self, cache, tokens, hidden, delta=0, positions=None):
            return hidden[..., :4] * 1.0, hidden

        def _mix_logits(self, mixed):
            calls.append(tuple(mixed.shape))
            return mx.concatenate([mixed, -mixed], axis=-1)

    head = Head()
    rows = []
    for _ in range(3):
        r = type("R", (), {})()
        r.hn = 0
        r.req = type("Q", (), {"rope_delta": 0})()
        head.caches[id(r)] = []
        rows.append(r)
    hidden = mx.random.normal((3, 2, 8), key=mx.random.key(1))
    tokens = mx.zeros((3, 2), dtype=mx.int32)
    states, greedy = head.absorb_window(rows, tokens, hidden)
    assert calls == [(6, 4)]
    want = mx.stack(
        [
            mx.argmax(
                mx.concatenate([hidden[b, :, :4], -hidden[b, :, :4]], axis=-1), axis=-1
            )
            for b in range(3)
        ]
    )
    assert greedy.shape == (3, 2)
    assert mx.array_equal(greedy, want.astype(mx.int32)).item()
    assert states.shape == (3, 2, 8)
