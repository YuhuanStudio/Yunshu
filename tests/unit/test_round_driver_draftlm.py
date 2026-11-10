"""Round driver, generic layer: the draft-LM and Gemma-4 assistant drafters (T4).

The tiny random LM of ``test_round_driver_generic`` (float32, CPU, no near ties) is both target and
draft. A draft with the target's own weights is accepted whole; a draft with other weights is mostly
rejected; either way the kept tokens equal the plain loop (the verifier's top-1 over the window).
The assistant is faked by an oracle that reads the true continuation, so what the test pins is the
plumbing the real Gemma-4 assistant needs: the target's normed hidden state at the last cached
position, the row's own target cache, and the position offset.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from test_round_driver_generic import (  # noqa: E402
    LOG,
    TinyLM,
    cpu_and_clear_log,  # noqa: F401  (autouse fixture)
    make,
    prompt,
    prompt_kwargs,
    reference,
    run,
    tokens,
)

from yunshu_engine.round_driver import driver as drv  # noqa: E402
from yunshu_engine.round_driver import generic  # noqa: E402
from yunshu_engine.round_driver.draftlm import (  # noqa: E402
    DraftLM,
    DraftLMDrafter,
    RowDrafters,
)


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


class FakeDraft(DraftLM):
    """A draft whose model is a TinyLM (mlx-lm call: tokens + cache -> logits)."""

    def __init__(self, lm):
        self.lm = lm
        self.model = lambda toks, cache: lm(toks, cache=cache).logits
        self.path = "fake"

    def new_cache(self):
        return generic.new_cache(self.lm)


def windows():
    return [c for c in LOG if 1 < c["T"] <= 6 and c["keep"] is None and c["n"] is None]


def test_draft_lm_equal_weights_accepts_every_draft_and_matches_plain():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    d = make(lm, row_drafters=RowDrafters(draft_lm=FakeDraft(TinyLM())))
    out = run(d, [req(lm, ids, 48, h)])
    assert tokens(out, h) == reference(lm, ids, 48)
    assert h.stats.spec_mode == "draft-lm"
    assert windows(), "the draft LM never engaged"
    assert h.stats.spec_rounds > 0
    assert h.stats.spec_accepted == h.stats.spec_drafted > 0


def test_draft_lm_with_other_weights_still_equals_plain():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    d = make(lm, row_drafters=RowDrafters(draft_lm=FakeDraft(TinyLM(seed=5))))
    out = run(d, [req(lm, ids, 48, h)])
    assert tokens(out, h) == reference(lm, ids, 48)
    assert h.stats.spec_mode == "draft-lm"
    assert h.stats.spec_drafted > h.stats.spec_accepted


def test_draft_cache_follows_the_row_context():
    lm = TinyLM()
    drafter = DraftLMDrafter(FakeDraft(TinyLM()))
    drafter.steps = 3
    ctx = prompt(10)
    first = drafter.propose(ctx, 4)
    assert len(first) == 4 and drafter.fed == len(ctx)
    assert drafter.cache[0].offset == len(ctx)  # the chain's extra tokens are trimmed
    # the row accepted two drafts and has a new pending token: only those are fed again
    ctx2 = [*ctx, *first[:2], 7]
    second = drafter.propose(ctx2, 3)
    assert drafter.fed == len(ctx2) and drafter.cache[0].offset == len(ctx2)
    ref = generic.new_cache(lm)
    want = lm(mx.array(ctx2)[None], cache=ref).logits[0, -1]
    assert second[0] == int(mx.argmax(want))
    # a shorter context resyncs instead of feeding a negative span
    third = drafter.propose(ctx, 2)
    assert len(third) == 2 and drafter.fed == len(ctx)


def test_row_without_the_flag_decodes_plain():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    d = make(lm, row_drafters=RowDrafters(draft_lm=FakeDraft(TinyLM())))
    out = run(d, [req(lm, ids, 40, h, ngram=False)])
    assert tokens(out, h) == reference(lm, ids, 40)
    assert not windows() and h.stats.spec_mode is None


def test_concurrent_rows_decode_plain_and_equal_solo():
    lm = TinyLM()
    a, b = prompt(12), prompt(9, salt=3)
    ha, hb = handle(), handle()
    d = make(lm, row_drafters=RowDrafters(draft_lm=FakeDraft(TinyLM())))
    out = run(d, [req(lm, a, 30, ha), req(lm, b, 30, hb)])
    assert tokens(out, ha) == reference(lm, a, 30)
    assert tokens(out, hb) == reference(lm, b, 30)


class Oracle:
    """Stands in for ``Gemma4AssistantProposer``: drafts from the true continuation, records its inputs."""

    sliding_kv_layer = 0
    full_kv_layer = 1

    def __init__(self, lm, ids, n):
        self.ids = ids
        self.future = reference(lm, ids, n + 8)
        self.calls = []

    def propose_chain(self, last, hidden, kv, offset, k=1):
        j = offset + 1 - len(self.ids)
        assert self.future[j] == last, "the pending token is the one after the cache"
        self.calls.append((offset, hidden, {i: tuple(kv[i][0].shape) for i in kv}))
        return self.future[j + 1 : j + 1 + k]


def test_assistant_gets_hidden_state_cache_and_offset():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    oracle = Oracle(lm, ids, 40)
    d = make(lm, row_drafters=RowDrafters(assistant=oracle))
    out = run(d, [req(lm, ids, 40, h)])
    assert tokens(out, h) == reference(lm, ids, 40)
    assert h.stats.spec_mode == "gemma4-assistant"
    assert oracle.calls and windows()
    assert h.stats.spec_accepted == h.stats.spec_drafted > 0
    full = [*ids, *oracle.future]
    cache = generic.new_cache(lm)
    hs = lm(mx.array(full)[None], cache=cache, return_hidden=True).hidden_states[-1]
    want = lm.speculative_draft_hidden(hs)[0]
    for offset, hidden, shapes in oracle.calls:
        assert mx.allclose(hidden.reshape(-1), want[offset], atol=1e-4)
        assert shapes[0][2] == shapes[1][2] == offset + 1  # K/V of positions 0..offset


def test_assistant_with_a_bad_draft_still_equals_plain():
    lm = TinyLM()
    ids = prompt(12)
    h = handle()
    oracle = Oracle(lm, ids, 40)
    real = oracle.propose_chain

    def wrong(last, hidden, kv, offset, k=1):
        got = real(last, hidden, kv, offset, k)
        return [(t + 1) % 64 for t in got[:2]] + got[2:]

    oracle.propose_chain = wrong
    d = make(lm, row_drafters=RowDrafters(assistant=oracle))
    out = run(d, [req(lm, ids, 40, h)])
    assert tokens(out, h) == reference(lm, ids, 40)
    assert h.stats.spec_drafted > h.stats.spec_accepted


def test_draft_with_a_cache_that_would_wrap_proposes_nothing():
    from mlx_lm.models.cache import RotatingKVCache

    class Rot(FakeDraft):
        def new_cache(self):
            return [RotatingKVCache(max_size=16) for _ in range(2)]

    drafter = DraftLMDrafter(Rot(TinyLM()))
    drafter.steps = 3
    assert drafter.propose(prompt(20), 4) == []
