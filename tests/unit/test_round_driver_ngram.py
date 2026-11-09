"""Round driver, generic layer: the model-free n-gram / suffix drafter (T3).

The tiny random LM of ``test_round_driver_generic`` (float32, CPU, no near ties) stands in for a
family without a packed layer. Greedy decoding of a random LM falls into short cycles, which is
exactly where prompt lookup wins. The guarantee pinned here is the old verifier's: kept tokens are the
model's top-1 over the window, so on this tie-free model they equal the plain loop; cache offsets
match after every rejected tail (trim); counters report the rounds; rows the drafter cannot serve
(sampled, guided, logprobs, concurrent rows, caches that cannot trim) decode plain.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")
mx = pytest.importorskip("mlx.core")

from test_round_driver_generic import (  # noqa: E402
    LOG,
    TinyLM,
    _params,
    cpu_and_clear_log,  # noqa: F401  (autouse fixture)
    make,
    prompt,
    prompt_kwargs,
    reference,
    run,
    tokens,
)

from yunshu_engine.round_driver import driver as drv  # noqa: E402
from yunshu_engine.round_driver.ngram import MAX_COOL, NgramDrafter  # noqa: E402


class Handle:
    def __init__(self):
        self.stats = SimpleNamespace(
            spec_drafted=0, spec_accepted=0, spec_rounds=0, spec_mode=None
        )


def handle():
    return Handle()


def req(lm, ids, n=40, h=None, ngram=True, **kw):
    return drv.Request(
        ids,
        n,
        handle=h if h is not None else handle(),
        prompt_kwargs=prompt_kwargs(lm, ids),
        draft=False,
        ngram=ngram,
        **kw,
    )


def windows():
    return [c for c in LOG if 1 < c["T"] <= 6 and c["keep"] is None and c["n"] is None]


def test_ngram_tokens_equal_the_plain_loop_and_windows_run():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    out = run(make(lm), [req(lm, ids, 48, h)])
    assert tokens(out, h) == reference(lm, ids, 48)
    assert windows(), "no verify window ran: the drafter never engaged"
    st = h.stats
    assert st.spec_mode == "ngram"
    assert st.spec_rounds > 0 and st.spec_drafted >= st.spec_accepted > 0


def test_off_without_the_request_flag():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    out = run(make(lm), [req(lm, ids, 40, h, ngram=False)])
    assert tokens(out, h) == reference(lm, ids, 40)
    assert not windows() and h.stats.spec_rounds == 0


def test_trim_leaves_the_cache_where_plain_decode_would():
    lm = TinyLM()
    ids = prompt(10)
    d = make(lm)
    h = handle()
    run(d, [req(lm, ids, 30, h)])
    last = [c for c in LOG if c["T"] == 1 and c["keep"] is None]
    # every window fed only kept positions: the offset of the next call is prompt + fed tokens
    for a, b in zip(LOG, LOG[1:], strict=False):
        if a["keep"] is None and a["n"] is None and a["offset"] >= len(ids):
            assert b["offset"] <= a["offset"] + a["T"]
    assert last or windows()


def test_suffix_proposer_kind():
    lm = TinyLM()
    ids = prompt(12)
    d = make(lm)
    h = handle()
    r = req(lm, ids, 40, h)
    d.add(r)
    row = d.rows[0]
    assert row.ngram is not None and row.ngram.kind in ("ngram", "suffix")
    drafter = NgramDrafter("suffix")
    assert drafter.propose([1, 2, 3, 4, 1, 2, 3], 5)[:1] == [4]


@pytest.mark.parametrize(
    "extra",
    [
        {"sampling": _params(seed=3)},
        {"logprobs": True},
        {"prompt_logprobs": 1},
    ],
)
def test_rows_the_drafter_cannot_serve_decode_plain(extra):
    lm = TinyLM()
    ids = prompt(12)
    d = make(lm)
    d.add(req(lm, ids, 6, **extra))
    assert d.rows[0].ngram is None


def test_concurrent_rows_decode_plain_and_equal_solo():
    lm = TinyLM()
    prompts = [prompt(12, 1), prompt(15, 2)]
    hs = [handle(), handle()]
    out = run(make(lm), [req(lm, p, 30, h) for p, h in zip(prompts, hs, strict=True)])
    for p, h in zip(prompts, hs, strict=True):
        assert tokens(out, h) == reference(lm, p, 30)
    # a window only ever ran with one row in the batch
    assert all(c["B"] == 1 for c in windows())


def test_a_cache_that_cannot_trim_decodes_plain(monkeypatch):
    lm = TinyLM()
    ids = prompt(12)
    d = make(lm)
    monkeypatch.setattr(d.batch, "can_trim", lambda extra=0: False)
    h = handle()
    out = run(d, [req(lm, ids, 30, h)])
    assert tokens(out, h) == reference(lm, ids, 30)
    assert not windows()


def test_length_limit_cuts_the_window():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    out = run(make(lm), [req(lm, ids, 7, h)])
    assert tokens(out, h) == reference(lm, ids, 7)
    assert out[h][-1].finish == "length"


def test_ngram_replies_have_their_own_apc_namespace():
    d = make(TinyLM())
    assert d.apc_salt(5, False, ngram=True) != d.apc_salt(5, False)


def test_backoff_after_misses_and_reset_on_a_hit():
    n = NgramDrafter("ngram")
    ctx = [1, 2, 3, 1, 2, 3, 1, 2]
    assert n.propose(ctx, 5) == [3, 1, 2]
    n.observe(3, 0)
    assert n.propose(ctx, 5) == []  # cooling: one step out
    assert n.cool == 0
    for _ in range(10):
        n.cool = 0
        n.observe(3, 0)
    assert n.cool == MAX_COOL
    n.observe(3, 1)
    assert n.misses == 0
    assert n.propose(ctx, 0) == []


def test_a_rotating_cache_that_would_wrap_cannot_trim():
    from mlx_vlm.models import cache as vc

    from yunshu_engine.round_driver.generic import _trimmable

    c = vc.RotatingKVCache(max_size=8, keep=0)
    k = mx.zeros((1, 1, 6, 4))
    c.update_and_fetch(k, k)
    assert _trimmable(c, 2) and not _trimmable(c, 3)  # 6 + 3 > 8: the window would wrap
    assert _trimmable(vc.KVCache(), 99)


def test_long_prompts_still_draft():
    # the proposers' own 32K max_model_len must not silence drafts on a longer context
    from yunshu_engine.round_driver.ngram import make_proposer

    for kind in ("ngram", "suffix"):
        ctx = list(range(100, 40000)) + [7, 8, 9, 10, 5, 7, 8, 9]
        assert make_proposer(kind).propose(ctx)[:1] == [10]
