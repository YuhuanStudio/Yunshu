"""Round driver, generic layer: the model's own forward, prefill per row, decode rows batched.

A tiny random attention LM on the CPU device stands in for a family without a packed layer. It
speaks the upstream mlx-vlm language-model call (``inputs_embeds``, ``n_to_process``,
``logits_to_keep``, ``rope_deltas``) and keeps mRoPE-style state on itself. The tests pin what the
layer promises: the stock single-row prefill recipe, decode rows sharing ONE upstream batched call
(batch caches, left padding) whose tokens equal the solo ones here (float32, no near ties; on the GPU
the batched arithmetic is the stock one, not row-invariant), per-row model state, every request kind a plain row serves (sampling,
logprobs, processors, stop, length, cancel), and the routing of every family to the driver.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")

from yunshu_engine.round_driver import driver as drv  # noqa: E402
from yunshu_engine.round_driver import generic  # noqa: E402

VOCAB, DIM, HEADS, DEPTH = 64, 32, 2, 2
LOG: list[dict] = []


@pytest.fixture(autouse=True)
def cpu_and_clear_log():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    LOG.clear()
    yield
    mx.set_default_device(previous)


def _first(x):
    """An int, or the first element of an array."""
    return int(mx.array(x).reshape(-1)[0])


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.hd = DIM // HEADS
        self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.o = nn.Linear(DIM, DIM, bias=False)
        self.ff = nn.Linear(DIM, DIM, bias=False)
        self.rope = nn.RoPE(self.hd)

    def __call__(self, x, cache):
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv(mx.fast.rms_norm(x, None, 1e-6)), 3, axis=-1)
        q, k, v = (
            t.reshape(B, T, HEADS, self.hd).transpose(0, 2, 1, 3) for t in (q, k, v)
        )
        offset = cache.offset  # an int, or [B] row positions of a batch cache
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        k, v = cache.update_and_fetch(k, v)
        mask = None
        if hasattr(
            cache, "left_padding"
        ):  # batch cache, one token a row: hide the left padding
            ok = mx.arange(k.shape[2])[None, :] >= cache.left_padding[:, None]
            mask = mx.where(ok, 0.0, -1e9)[:, None, None, :]
        elif T > 1:
            rows = mx.arange(T)[:, None] + offset
            cols = mx.arange(offset + T)[None, :]
            mask = mx.where(cols <= rows, 0.0, -1e9)
        a = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.hd**-0.5, mask=mask
        )
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, DIM))
        return x + self.ff(mx.fast.rms_norm(x, None, 1e-6))


class TinyLM(nn.Module):
    """The upstream language-model call on random weights (no ``make_cache``: the default KVCache)."""

    stateful = False

    def __init__(self, seed: int = 0):
        super().__init__()
        mx.random.seed(seed)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.blocks = [_Block() for _ in range(DEPTH)]
        if self.stateful:
            self._rope_deltas = None
            self._position_ids = None
        mx.eval(self.parameters())

    @property
    def layers(self):
        return self.blocks

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        cache=None,
        n_to_process=None,
        logits_to_keep=None,
        rope_deltas=None,
        **kwargs,
    ):
        T = inputs.shape[1]
        LOG.append(
            dict(
                T=T,
                n=n_to_process,
                keep=logits_to_keep,
                B=inputs.shape[0],
                offset=_first(cache[0].offset),
                rope=None if rope_deltas is None else rope_deltas.reshape(-1).tolist(),
                state=_first(self._rope_deltas)
                if self.stateful and self._rope_deltas is not None
                else None,
                extra=sorted(kwargs),
            )
        )
        shift = 0.0
        if self.stateful:
            # mRoPE-like: the offsets come from the call's kwarg, else (a later prompt span) from the
            # model's own state, set by the first call of a row; the maths depends on them
            if rope_deltas is not None:
                self._rope_deltas = rope_deltas
            elif _first(cache[0].offset) == 0:
                self._rope_deltas = mx.array([[T + 100 * int(inputs[0, 0])]])
            if self._rope_deltas is not None:
                shift = self._rope_deltas.reshape(-1, 1, 1).astype(mx.float32) * 1e-3
        x = self.embed(inputs) if inputs_embeds is None else inputs_embeds
        x = x + shift
        for block, c in zip(self.blocks, cache, strict=True):
            x = block(x, c)
        x = mx.fast.rms_norm(x, None, 1e-6)
        if logits_to_keep is not None:
            x = x[:, -logits_to_keep:]
        return _Out(self.embed.as_linear(x))


class StatefulLM(TinyLM):
    stateful = True


def prompt_kwargs(lm, ids, **extra):
    return {"inputs_embeds": lm.embed(mx.array(ids, dtype=mx.int32))[None], **extra}


def request(lm, ids, n=8, handle=None, **kw):
    return drv.Request(
        ids,
        n,
        handle=handle if handle is not None else tuple(ids[:2]),
        prompt_kwargs=prompt_kwargs(lm, ids),
        draft=False,
        **kw,
    )


def make(lm, chunk=8, **kw):
    kw.setdefault("stop_tokens", set())
    return drv.RoundDriver(lm, generic_layer=True, chunk=chunk, **kw)


def run(d, reqs, *, join_at=None, drop=None):
    """Drive ``d`` to idle; ``reqs`` start together or each at its step in ``join_at``."""
    join_at = join_at or {}
    out = {}
    pending = list(reqs)
    for r in list(pending):
        if r.handle not in join_at:
            d.add(r)
            pending.remove(r)
    steps = 0
    while d.busy() or pending:
        for r in list(pending):
            if join_at.get(r.handle, 0) <= steps:
                d.add(r)
                pending.remove(r)
        for e in d.step():
            out.setdefault(e.handle, []).append(e)
        steps += 1
        if drop and steps == drop[1]:
            d.remove(drop[0])
        assert steps < 500
    return out


def tokens(out, handle):
    return [e.token for e in out[handle]]


def prompt(n, salt=1):
    return [4 + (salt * 7 + 5 * i) % (VOCAB - 4) for i in range(n)]


def reference(lm, ids, n):
    """Greedy continuation by plain calls: full prompt once, then one token a call."""
    cache = generic.new_cache(lm)
    out = lm(mx.array(ids)[None], cache=cache).logits[0, -1]
    got = []
    for _ in range(n):
        t = int(mx.argmax(out - mx.logsumexp(out)))
        got.append(t)
        out = lm(mx.array([[t]]), cache=cache).logits[0, -1]
    return got


# ── the stock single-row call recipe ─────────────────────────────────────────────────────────
def test_prefill_calls_are_the_stock_recipe():
    lm = TinyLM()
    d = make(lm, chunk=8)
    ids = prompt(20)
    run(d, [request(lm, ids, n=2)])
    prefill = [c for c in LOG if c["T"] > 1 or c["keep"] is not None]
    # spans 0-8, 8-16, 16-20: chunks with n_to_process, the last prompt token alone with one logits row
    assert [(c["T"], c["n"], c["keep"], c["offset"]) for c in prefill] == [
        (8, 8, None, 0),
        (8, 8, None, 8),
        (3, 3, None, 16),
        (1, None, 1, 19),
    ]
    decode = LOG[len(prefill) :]
    assert all(c["T"] == 1 and c["keep"] is None and c["n"] is None for c in decode)


def test_a_one_token_prompt_runs_only_the_last_token_call():
    lm = TinyLM()
    run(make(lm), [request(lm, [9], n=1)])
    assert [(c["T"], c["keep"]) for c in LOG] == [(1, 1)]


def test_sequence_aligned_kwargs_are_cut_with_the_span():
    lm = TinyLM()
    ids = prompt(11)
    pkw = prompt_kwargs(
        lm, ids, per_layer_inputs=mx.zeros((1, 11, 4)), flag=mx.ones((1,))
    )
    feats = generic.PromptFeatures(pkw, 11)
    assert feats.aware == ["per_layer_inputs"]
    assert feats.span(3, 7)["per_layer_inputs"].shape == (1, 4, 4)
    assert feats.span(3, 7)["flag"].shape == (1,)  # not sequence aligned: passed whole


def test_private_and_empty_prompt_features_are_not_forwarded():
    lm = TinyLM()
    pkw = prompt_kwargs(lm, prompt(5), _apc_semantic_hash=7, attention_mask_4d=None)
    feats = generic.PromptFeatures(pkw, 5)
    assert feats.kwargs == {}


def test_prompt_features_refuse_a_mismatched_or_missing_embedding():
    lm = TinyLM()
    with pytest.raises(ValueError, match="embedding rows"):
        generic.PromptFeatures(prompt_kwargs(lm, prompt(5)), 6)
    with pytest.raises(ValueError, match="inputs_embeds"):
        generic.PromptFeatures({"rope_deltas": mx.zeros((1, 1))}, 5)


# ── identity: alone == beside others ─────────────────────────────────────────────────────────
PROMPTS = [prompt(5, 1), prompt(23, 2), prompt(9, 3)]


def solo(lm, ids, n=8, **kw):
    return tokens(run(make(lm), [request(lm, ids, n, handle="a", **kw)]), "a")


def test_greedy_rows_follow_the_plain_loop():
    lm = TinyLM()
    for ids in PROMPTS:
        assert solo(lm, ids, 10) == reference(lm, ids, 10)


def test_solo_equals_concurrent_and_staggered():
    lm = TinyLM()
    alone = [solo(lm, ids, 12) for ids in PROMPTS]
    reqs = [request(lm, ids, 12, handle=i) for i, ids in enumerate(PROMPTS)]
    together = run(make(lm), reqs)
    assert [tokens(together, i) for i in range(3)] == alone
    reqs = [request(lm, ids, 12, handle=i) for i, ids in enumerate(PROMPTS)]
    staggered = run(make(lm), reqs, join_at={0: 0, 1: 3, 2: 7})
    assert [tokens(staggered, i) for i in range(3)] == alone


def test_sampled_rows_with_a_seed_are_the_same_alone_and_beside_others():
    lm = TinyLM()
    params = _params(seed=11)
    alone = [
        tokens(run(make(lm), [request(lm, ids, 10, handle="a", sampling=params)]), "a")
        for ids in PROMPTS
    ]
    reqs = [
        request(lm, ids, 10, handle=i, sampling=_params(seed=11))
        for i, ids in enumerate(PROMPTS)
    ]
    together = run(make(lm), reqs)
    assert [tokens(together, i) for i in range(3)] == alone
    assert alone[0] != solo(lm, PROMPTS[0], 10)  # it really sampled


def _params(seed=None):
    from yunshu_engine.vlm_batch_runner import RowParams

    return RowParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=seed)


# ── per-row model state ──────────────────────────────────────────────────────────────────────
def test_each_row_decodes_with_its_own_rope_deltas_not_the_models_last():
    lm = StatefulLM()
    a, b = prompt(5, 1), prompt(9, 2)
    run(make(lm), [request(lm, a, 6, handle="a"), request(lm, b, 6, handle="b")])
    decode = [c for c in LOG if c["T"] == 1 and c["keep"] is None]
    assert all(
        c["rope"] is not None for c in decode
    )  # the captured offsets, as the stock generator
    # the model recorded T + 100 * first token on each row's first call: one value per row, in batch order
    both = [c for c in decode if c["B"] == 2]
    assert both and both[0]["rope"] == [4 + 100 * a[0], 8 + 100 * b[0]]


def test_rope_state_does_not_leak_between_interleaved_prefills():
    long, short = prompt(40, 1), prompt(6, 2)
    # S decodes while L prefills one span a step: S's calls overwrite the model's state between L's
    # spans, and the later spans (no rope kwarg) read it. Each row swaps its own state in and out.
    lm = StatefulLM()
    reqs = [request(lm, short, 30, handle="S"), request(lm, long, 6, handle="L")]
    both = run(make(lm), reqs, join_at={"L": 2})
    assert tokens(both, "L") == solo_stateful(long, 6)
    assert tokens(both, "S") == solo_stateful(short, 30)
    # the setting really interleaved: L's spans were cut while S decoded
    assert sum(1 for c in LOG if c["T"] == 8 and c["n"] == 8) >= 4


def solo_stateful(ids, n):
    lm = StatefulLM()
    return tokens(run(make(lm), [request(lm, ids, n, handle="x")]), "x")


# ── request kinds ────────────────────────────────────────────────────────────────────────────
def test_length_and_stop_finish_reasons():
    lm = TinyLM()
    out = run(make(lm), [request(lm, PROMPTS[0], 4, handle="a")])
    assert [e.finish for e in out["a"]] == [None, None, None, "length"]
    # the stop token ends the row at it: pick a prompt whose reference reaches EOS or force it
    ref = reference(lm, PROMPTS[1], 40)
    target = ref[3]
    d = make(lm, stop_tokens={target})
    out = run(d, [request(lm, PROMPTS[1], 40, handle="s")])
    got = tokens(out, "s")
    assert got[-1] == target and out["s"][-1].finish == "stop" and len(got) <= 4


def test_logprobs_are_reported_per_token_with_top_k():
    lm = TinyLM()
    out = run(
        make(lm),
        [request(lm, PROMPTS[0], 5, handle="a", logprobs=True, top_logprobs=3)],
    )
    for e in out["a"]:
        assert e.logprob is not None and e.logprob["logprob"] <= 0.0
        assert len(e.logprob["top_logprobs"]) == 3
        assert (
            e.logprob["top_logprobs"][0]["logprob"]
            >= e.logprob["top_logprobs"][-1]["logprob"]
        )


def test_logit_bias_and_token_ban_apply_on_the_generic_rows():
    from yunshu_engine.vlm_batch_runner import LogitBiasProcessor, TokenMaskProcessor

    lm = TinyLM()
    base = solo(lm, PROMPTS[0], 6)
    biased = solo(lm, PROMPTS[0], 6, processors=[LogitBiasProcessor({17: 100.0})])
    assert biased == [17] * 6
    banned = solo(
        lm, PROMPTS[0], 6, processors=[TokenMaskProcessor(suppress=[base[0]])]
    )
    assert base[0] not in banned[:1]


def test_cancel_mid_decode_leaves_the_other_row_unchanged():
    lm = TinyLM()
    alone = solo(lm, PROMPTS[1], 12)
    d = make(lm)
    out = run(
        d,
        [
            request(lm, PROMPTS[1], 12, handle="keep"),
            request(lm, PROMPTS[2], 60, handle="drop"),
        ],
        drop=("drop", 6),
    )
    assert tokens(out, "keep") == alone
    assert not d.rows and not d.batch.rows


def test_every_step_evaluates_the_caches_it_advanced():
    lm = TinyLM()
    d = make(lm, chunk=8)
    d.add(request(lm, prompt(30), 4, handle="a"))
    d.step()  # the whole idle prompt (spans of 8) and its first token
    row = d.rows[0]
    assert row.pending is not None and row.cache[0].offset == 30
    assert row.cache[0].keys is not None


# ── construction guards ──────────────────────────────────────────────────────────────────────
def test_a_generic_driver_takes_no_drafter_and_no_int8_kv():
    lm = TinyLM()
    with pytest.raises(ValueError, match="no drafter"):
        drv.RoundDriver(lm, generic_layer=True, drafter=object())
    with pytest.raises(ValueError, match="YUNSHU_KV_PRECISION"):
        drv.RoundDriver(lm, generic_layer=True, kv_format="int8")


def test_a_generic_row_needs_its_prompt_features():
    lm = TinyLM()
    d = make(lm)
    with pytest.raises(ValueError, match="prompt_kwargs"):
        d.add(drv.Request(prompt(4), 2, handle="a"))


def test_the_generic_driver_never_drafts():
    lm = TinyLM()
    d = make(lm)
    assert d.can_draft is False
    assert (
        d.will_draft(sampling=None, processors=[], logprobs=False, draft=True) is False
    )
    assert d.max_draft == 0


# ── routing: every family takes the driver ──────────────────────────────────────────────────
def test_generic_request_features_of_a_text_prompt_come_from_the_embedding_call():
    """The runner feeds a text prompt what the stock generator feeds it: ``get_input_embeddings``."""
    from types import SimpleNamespace

    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    lm = TinyLM()
    calls = []

    class Feat:
        def __init__(self, ids):
            self.ids = ids

        def to_dict(self):
            return {"inputs_embeds": lm.embed(self.ids), "position_ids": None}

    def get_input_embeddings(ids, pixels, mask=None):
        calls.append((ids.shape, pixels, mask))
        return Feat(ids)

    runner = object.__new__(VLMBatchRunner)
    runner.model = SimpleNamespace(get_input_embeddings=get_input_embeddings)
    job = SimpleNamespace(ids=prompt(7), prompt_kwargs=None)
    got = runner._generic_request(job)
    assert calls == [((1, 7), None, None)]
    assert got["media"] is False and got["prompt_kwargs"]["inputs_embeds"].shape == (
        1,
        7,
        DIM,
    )
    media = {
        "inputs_embeds": mx.zeros((1, 7, DIM)),
        "position_ids": mx.zeros((3, 1, 7)),
    }
    job = SimpleNamespace(ids=prompt(7), prompt_kwargs=media)
    got = runner._generic_request(job)
    assert got["media"] is True and got["prompt_kwargs"] is media


def test_the_engine_builds_a_generic_driver_for_every_family_without_a_packed_layer():
    import inspect

    from yunshu_engine import vlm_engine

    src = inspect.getsource(vlm_engine.VLMEngine._build_batch_runner)
    # one driver for every model: the packed layers when wanted, else the generic one (paravirtual too)
    assert src.count("generic_layer=True") == 2
    assert "generic_layer = not use_driver" in src


def test_the_generic_apc_grid_is_the_prefill_step():
    from yunshu_engine.vlm_engine import apc_prefill_stride

    assert apc_prefill_stride(False, False, False, 2048, generic=True) == 2048
    assert apc_prefill_stride(False, False, False, 2048) == 0


# ── batched decode ───────────────────────────────────────────────────────────────────────────
def test_decode_rows_share_one_batched_call_and_a_lone_row_keeps_its_own_cache():
    lm = TinyLM()
    run(
        make(lm),
        [request(lm, PROMPTS[0], 6, handle=0), request(lm, PROMPTS[1], 6, handle=1)],
    )
    decode = [c for c in LOG if c["T"] == 1 and c["keep"] is None]
    assert decode and all(
        c["B"] == 2 for c in decode
    )  # one call per step for both rows
    LOG.clear()
    run(make(lm), [request(lm, PROMPTS[0], 6, handle=0)])
    assert all(c["B"] == 1 for c in LOG)


def test_the_batch_shrinks_when_a_row_leaves_and_the_other_keeps_its_tokens():
    lm = TinyLM()
    short = solo(lm, PROMPTS[0], 12)
    out = run(
        make(lm),
        [
            request(lm, PROMPTS[0], 12, handle="a"),
            request(lm, PROMPTS[1], 3, handle="b"),
        ],
    )
    assert tokens(out, "a") == short
    sizes = [c["B"] for c in LOG if c["T"] == 1 and c["keep"] is None]
    assert sizes[0] == 2 and sizes[-1] == 1


def _prefilled(lm, ids):
    from types import SimpleNamespace

    row = SimpleNamespace(
        req=SimpleNamespace(ids=ids),
        features=generic.PromptFeatures(prompt_kwargs(lm, ids), len(ids)),
        state=generic.RowState(),
        cache=generic.new_cache(lm),
        slot=None,
        n=len(ids),
    )
    generic.prefill(lm, row, 0, len(ids))
    mx.eval(*generic.cache_arrays(row.cache))
    return row


def test_a_batch_cache_gives_a_rows_own_cache_back_and_leave_filters_the_others():
    lm = TinyLM()
    a, b = _prefilled(lm, prompt(7, 1)), _prefilled(lm, prompt(12, 2))
    want = _prefilled(lm, prompt(12, 2))
    batch = generic.GenericBatch(lm)
    batch.join([a, b])
    assert a.cache is None and b.cache is None  # the batch cache owns the K/V
    assert batch.cache[0].keys.shape[0] == 2
    batch.restore(b)
    assert b.cache[0].offset == 12
    assert mx.allclose(
        b.cache[0].keys[..., :12, :], want.cache[0].keys[..., :12, :]
    ).item()
    batch.leave([a])
    assert batch.rows == [b] and batch.cache[0].keys.shape[0] == 1
    batch.leave([b])
    assert batch.cache is None and batch.rows == []
