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
from yunshu_engine.round_driver.layer import PackedLayer  # noqa: E402

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
            causal = mx.arange(k.shape[2])[None, :] <= (
                mx.arange(T)[:, None] + k.shape[2] - T
            )
            mask = mx.where(ok[:, None, :] & causal[None], 0.0, -1e9)[:, None]
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
        pre = x
        x = mx.fast.rms_norm(x, None, 1e-6)
        if logits_to_keep is not None:
            x = x[:, -logits_to_keep:]
        out = _Out(self.embed.as_linear(x))
        if kwargs.get(
            "return_hidden"
        ):  # mlx-vlm: the last decoder output, before the final norm
            out.hidden_states = [pre]
        return out

    def speculative_draft_hidden(self, hidden):
        return mx.fast.rms_norm(hidden, None, 1e-6)


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
    firsts = {tuple(c["rope"]) for c in decode}
    assert (4 + 100 * a[0],) in firsts and (8 + 100 * b[0],) in firsts


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
def test_decode_rows_each_make_their_own_call_on_their_own_cache():
    lm = TinyLM()
    run(
        make(lm),
        [request(lm, PROMPTS[0], 6, handle=0), request(lm, PROMPTS[1], 6, handle=1)],
    )
    decode = [c for c in LOG if c["T"] == 1 and c["keep"] is None]
    assert decode and all(c["B"] == 1 for c in decode)
    LOG.clear()
    run(make(lm), [request(lm, PROMPTS[0], 6, handle=0)])
    assert all(c["B"] == 1 for c in LOG)


def test_a_row_leaving_leaves_the_other_with_its_tokens():
    lm = TinyLM()
    short = solo(lm, PROMPTS[0], 12)
    LOG.clear()
    out = run(
        make(lm),
        [
            request(lm, PROMPTS[0], 12, handle="a"),
            request(lm, PROMPTS[1], 3, handle="b"),
        ],
    )
    assert tokens(out, "a") == short
    assert {c["B"] for c in LOG} == {1}


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


def test_rows_keep_their_own_caches_and_leave_drops_only_that_row():
    lm = TinyLM()
    a, b = _prefilled(lm, prompt(7, 1)), _prefilled(lm, prompt(12, 2))
    want = _prefilled(lm, prompt(12, 2))
    batch = generic.GenericBatch(lm)
    batch.join([a, b])
    assert a.cache is not None and b.cache is not None
    assert batch.cache[0].keys.shape[0] == 1
    batch.cut_out(b)
    assert b.cache[0].offset == 12
    assert mx.allclose(
        b.cache[0].keys[..., :12, :], want.cache[0].keys[..., :12, :]
    ).item()
    batch.leave([a])
    assert batch.rows == [b] and a.slot is None
    batch.leave([b])
    assert batch.cache is None and batch.rows == []


class _DyingGuide:
    """A guide whose grammar has no continuation after ``alive`` tokens."""

    constrained = True

    def __init__(self, alive):
        self.alive, self.fed, self.fault = alive, 0, None

    def mask(self):
        from yunshu_engine.constraint_eos import GrammarDeadEndError

        if self.fed >= self.alive:
            raise GrammarDeadEndError("Grammar constraint has no valid next token")
        return None

    def arms(self, token):
        return False

    def feed(self, token):
        self.fed += 1


def test_a_grammar_dead_end_marks_only_that_row_and_the_step_goes_on():
    lm = TinyLM()
    alone = solo(lm, PROMPTS[1], 8)
    guide = _DyingGuide(alive=2)
    d = make(lm)
    reqs = [
        request(lm, PROMPTS[0], 8, handle="bad", guide=guide),
        request(lm, PROMPTS[1], 8, handle="good"),
    ]
    out = run(d, reqs)  # must not raise
    assert isinstance(guide.fault, Exception)
    assert tokens(out, "good") == alone


@pytest.mark.parametrize("chunk", [3, 8, 512])
@pytest.mark.parametrize("k", [0, 5])
def test_prompt_logprobs_plain_forward(chunk, k):
    from yunshu_engine.text_logprobs import top_logprobs_device

    lm = TinyLM()
    ids = prompt(71)
    logits = lm(mx.array([ids]), cache=generic.new_cache(lm)).logits[0]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    expected_ids, expected_vals = top_logprobs_device(lp[:-1], max(1, k))
    out = run(make(lm, chunk=chunk), [request(lm, ids, n=2, prompt_logprobs=k)])
    events = out[tuple(ids[:2])]
    scores = events[0].prompt_logprobs
    assert scores[0] is None
    assert len(scores) == len(ids)
    for i, entry in enumerate(scores[1:]):
        assert entry["token_id"] == ids[i + 1]
        assert entry["logprob"] == pytest.approx(float(lp[i, ids[i + 1]]), abs=2e-6)
        if k:
            assert [t["token_id"] for t in entry["top_logprobs"]] == expected_ids[
                i
            ].tolist()
            assert [t["logprob"] for t in entry["top_logprobs"]] == pytest.approx(
                expected_vals[i].tolist(), abs=2e-6
            )
        else:
            assert "top_logprobs" not in entry
    assert all(e.prompt_logprobs is None for e in events[1:])
    assert max(c["T"] for c in LOG[1:]) <= 32


def test_prompt_scoring_skips_apc_restore():
    from types import SimpleNamespace

    d = make(TinyLM())
    d.apc = object()
    # No coordinator may be constructed: scoring must recompute even a cached prompt.
    d._new_coordinator = lambda *_: pytest.fail("scoring restored an APC prefix")
    row = SimpleNamespace(req=drv.Request([1, 2], 1, prompt_logprobs=0))
    d._restore(row)


def test_prompt_scoring_ties_and_chunk_identity():
    from types import SimpleNamespace

    from yunshu_engine.round_driver.prompt_logprobs import score_span

    ids = [1, 2, 3, 0, 1]
    logits = mx.zeros((4, 5))
    rows = [
        SimpleNamespace(
            req=drv.Request(ids, 1, prompt_logprobs=3), prompt_scores=[None]
        )
        for _ in range(2)
    ]
    score_span(rows[0], 0, logits)
    score_span(rows[1], 0, logits[:2])
    score_span(rows[1], 2, logits[2:])
    assert rows[0].prompt_scores == rows[1].prompt_scores
    assert [t["token_id"] for t in rows[0].prompt_scores[1]["top_logprobs"]] == [
        0,
        1,
        2,
    ]


@pytest.mark.parametrize("k", [None, 0, 3])
def test_single_prompt_token(k):
    lm = TinyLM()
    out = run(make(lm), [request(lm, [3], n=2, prompt_logprobs=k)])
    events = out[(3,)]
    assert events[0].prompt_logprobs == (None if k is None else [None])
    assert events[1].prompt_logprobs is None


def test_unrequested_prompt_scores_do_no_work(monkeypatch):
    from yunshu_engine.round_driver import prompt_logprobs

    monkeypatch.setattr(
        prompt_logprobs, "score_span", lambda *_: pytest.fail("unrequested scoring")
    )
    lm = TinyLM()
    ids = prompt(71)
    events = run(make(lm), [request(lm, ids)])[tuple(ids[:2])]
    assert all(e.prompt_logprobs is None for e in events)


def test_prompt_scoring_chunked_equals_full_forward_bits():
    """A lookup LM isolates scoring bits from attention/GEMM batch-shape rounding."""

    class LookupLM(TinyLM):
        def __call__(self, inputs, **kwargs):
            table = (
                mx.arange(VOCAB * VOCAB, dtype=mx.float32).reshape(VOCAB, VOCAB) * 0.125
            )
            out = table[inputs]
            if kwargs.get("logits_to_keep") is not None:
                out = out[:, -kwargs["logits_to_keep"] :]
            return _Out(out)

    lm = LookupLM()
    ids = prompt(71)
    logits = lm(mx.array([ids])).logits[0]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    scores = []
    for chunk in (3, 512):
        events = run(make(lm, chunk=chunk), [request(lm, ids, n=1, prompt_logprobs=3)])[
            tuple(ids[:2])
        ]
        scores.append(events[0].prompt_logprobs)
    assert scores[0] == scores[1]
    assert [e["logprob"] for e in scores[0][1:]] == [
        float(lp[i, ids[i + 1]]) for i in range(len(ids) - 1)
    ]


@pytest.mark.parametrize("delta,media", [(None, False), (-2, True)])
def test_first_resumed_span_keeps_absolute_rope_positions(delta, media):
    """Restoring K/V must not make a fresh row recompute suffix RoPE from zero."""
    from types import SimpleNamespace

    observed = []

    class LM:
        _rope_deltas = None
        _position_ids = None

        def __call__(self, ids, **kwargs):
            # Qwen's upstream forward recomputes positions from the supplied
            # suffix if _rope_deltas is None, even with a nonempty KV cache.
            if kwargs.get("rope_deltas") is not None:
                self._rope_deltas = kwargs["rope_deltas"]
            offset = (
                0 if self._rope_deltas is None else 3 + int(self._rope_deltas.item())
            )
            observed.extend(range(offset, offset + ids.shape[-1]))
            return mx.zeros((1, ids.shape[-1], 8))

    kwargs = {"inputs_embeds": mx.zeros((1, 6, 8))}
    if delta is not None:
        kwargs["rope_deltas"] = mx.array([[delta]])
    row = SimpleNamespace(
        req=SimpleNamespace(ids=list(range(6)), media=media),
        cache=[],
        features=generic.PromptFeatures(kwargs, 6),
        state=generic.RowState(),
    )
    generic.prefill(LM(), row, 3, 5)
    assert observed == [3 + (delta or 0), 4 + (delta or 0)]


class _ForcedGuide:
    """A token grammar with singleton masks, including checkpointed lookahead."""

    from yunshu_engine.constrained_spec import ConstraintGuide

    plan = ConstraintGuide.plan
    words = (VOCAB + 31) // 32
    constrained = True

    def __init__(self, tokens, vocab=VOCAB):
        self.words = (vocab + 31) // 32
        self.tokens = tokens
        self.at = 0
        self.fault = None

    def arms(self, token):
        return False

    def checkpoint(self):
        return self.at

    def restore(self, cp):
        self.at = cp

    def mask(self):
        import numpy as np

        mask = np.zeros(self.words, dtype=np.int32)
        token = self.tokens[min(self.at, len(self.tokens) - 1)]
        mask[token // 32] = 1 << (token % 32)
        return mask

    def feed(self, token):
        assert token == self.tokens[min(self.at, len(self.tokens) - 1)]
        self.at += 1
        return True


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_forced_run_matches_serial_tokens(monkeypatch, enabled, batched):
    from yunshu_engine import settings

    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *a, **kw: (
            enabled if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
        ),
    )
    lm = TinyLM()
    scripts = (
        [list(range(7, 22)), list(range(11, 18))] if batched else [list(range(7, 22))]
    )
    guides = [_ForcedGuide(s) for s in scripts]
    reqs = [
        request(lm, [1, 2, 3 + i], len(s), handle=i, guide=g)
        for i, (s, g) in enumerate(zip(scripts, guides, strict=True))
    ]
    out = run(make(lm), reqs)
    for i, script in enumerate(scripts):
        assert [e.token for e in out[i]] == script
        assert guides[i].at == len(script)
        assert out[i][-1].finish == "length"
    decode = [c for c in LOG if c["keep"] is None and c["n"] is None]
    assert any(c["T"] > 1 for c in decode) == enabled


def test_forced_plan_and_cancel_restore_guide(monkeypatch):
    from yunshu_engine import settings

    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *a, **kw: (
            True if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
        ),
    )
    lm = TinyLM()
    guide = _ForcedGuide(list(range(7, 25)))
    d = make(lm)
    handle = object()
    d.add(request(lm, [1, 2, 3], 18, handle=handle, guide=guide))
    while not d.batch.rows:
        d.step()
    row = d.batch.rows[0]
    at = guide.at
    assert len(d._forced_window(row, 8)) > 1
    assert guide.at == at
    d.remove(handle)
    assert guide.at == at
    assert not d.busy()


@pytest.mark.parametrize("used", [[3, 1], [2, 1]])
def test_generic_windows_keep_each_rows_cache_and_causal_mask(used):
    import copy
    from types import SimpleNamespace

    lm = TinyLM()
    rows, refs = [], []
    for prompt in ([1, 2, 3], [4, 5]):
        cache = generic.new_cache(lm)
        lm(mx.array([prompt]), cache=cache)
        refs.append(copy.deepcopy(cache))
        rows.append(
            SimpleNamespace(cache=cache, state=generic.RowState(), n=len(prompt))
        )
    batch = generic.GenericBatch(lm)
    batch.join(rows)
    windows = [[7, 8, 9], [10]]
    got = batch.forward([3, 1], mx.array([7, 8, 9, 10]))
    expected = []
    for window, cache, keep in zip(windows, refs, used, strict=True):
        check = copy.deepcopy(cache)
        expected.extend(lm(mx.array([[t]]), cache=check).logits[0, 0] for t in window)
        for token in window[:keep]:
            lm(mx.array([[token]]), cache=cache)
    assert mx.allclose(got, mx.stack(expected), atol=1e-5).item()
    batch.commit(used)
    nxt = batch.forward([1, 1], mx.array([11, 12]))
    ref = mx.concatenate(
        [
            lm(mx.array([[t]]), cache=c).logits[:, 0]
            for t, c in zip([11, 12], refs, strict=True)
        ]
    )
    assert mx.allclose(nxt, ref, atol=1e-5).item()


@pytest.mark.parametrize("boundary", ["stop", "arm", "fault"])
def test_forced_planning_stops_at_boundaries_and_restores(boundary):
    from yunshu_engine.constrained_spec import forced_tokens

    guide = _ForcedGuide(list(range(7, 22)))
    if boundary == "arm":
        guide.arms = lambda t: t == 9
    elif boundary == "fault":
        mask = guide.mask

        def failing_mask():
            if guide.at == 3:
                raise RuntimeError("cancelled during lookahead")
            return mask()

        guide.mask = failing_mask
    if boundary == "fault":
        with pytest.raises(RuntimeError, match="cancelled"):
            forced_tokens(guide, 10)
    else:
        assert forced_tokens(guide, 10, {9} if boundary == "stop" else ()) == [7, 8, 9]
    assert guide.at == 0


def test_forced_plain_rows_apply_processors_at_each_position(monkeypatch):
    from yunshu_engine import settings
    from yunshu_engine.vlm_batch_runner import LogitBiasProcessor

    # A static logit bias is windowable but does not make an ordinary row draft.

    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *a, **kw: (
            True if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
        ),
    )
    lm = TinyLM()
    script = list(range(7, 22))
    processor = LogitBiasProcessor({7: 1.0})
    out = run(
        make(lm),
        [
            request(
                lm,
                [1, 2, 3],
                len(script),
                handle=0,
                guide=_ForcedGuide(script),
                processors=[processor],
            )
        ],
    )
    assert [e.token for e in out[0]] == script
    assert any(c["T"] > 1 and c["n"] is None for c in LOG)


def test_single_forced_token_verifies_the_following_choice(monkeypatch):
    from yunshu_engine import settings

    class BranchGuide(_ForcedGuide):
        def mask(self):
            mask = super().mask()
            # Alternate singleton and real branch positions. A forced-token window
            # must read the model's next choice through the guide's planned mask.
            if self.at % 2:
                token = self.tokens[min(self.at, len(self.tokens) - 1)] + 20
                mask[token // 32] |= 1 << (token % 32)
            return mask

        def feed(self, token):
            assert (int(self.mask()[token // 32]) >> (token % 32)) & 1
            self.at += 1
            return True

    original = settings.get
    results, calls = [], []
    for enabled in (False, True):
        monkeypatch.setattr(
            settings,
            "get",
            lambda key, *a, **kw: (
                enabled if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
            ),
        )
        LOG.clear()
        lm = TinyLM()
        guide = BranchGuide(list(range(7, 25)))
        out = run(make(lm), [request(lm, [1, 2, 3], 18, handle=0, guide=guide)])
        results.append([e.token for e in out[0]])
        calls.append(len(LOG))
        assert guide.at == 18
    assert results[0] == results[1]
    assert calls[1] < calls[0]


@pytest.mark.parametrize("boundary", ["stop", "arm"])
def test_forced_windows_emit_boundary_as_final_draw(monkeypatch, boundary):
    from yunshu_engine import settings

    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *a, **kw: (
            True if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
        ),
    )
    lm = TinyLM()
    guide = _ForcedGuide(list(range(7, 25)))
    if boundary == "arm":
        guide.arms = lambda t: t == 12
    d = make(lm, stop_tokens={12} if boundary == "stop" else set())
    handle = object()
    d.add(request(lm, [1, 2, 3], 18, handle=handle, guide=guide))
    emitted = []
    while d.busy():
        events = d.step()
        values = [e.token for e in events]
        if 12 in values:
            assert values[-1] == 12
        emitted.extend(values)
    assert emitted == list(range(7, 13 if boundary == "stop" else 25))
    assert guide.at == len(emitted)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sigma", [0.0, 2.0])
def test_runner_hands_plain_rows_a_live_guide_for_jump_forward(
    monkeypatch, enabled, sigma
):
    from types import SimpleNamespace

    from yunshu_engine import settings
    from yunshu_engine.constrained_spec import ConstraintGuide
    from yunshu_engine.vlm_batch_runner import TokenMaskProcessor, VLMBatchRunner

    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *a, **kw: (
            enabled if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
        ),
    )
    runner = VLMBatchRunner.__new__(VLMBatchRunner)
    runner.driver = SimpleNamespace(can_draft=False)
    guide = ConstraintGuide.__new__(ConstraintGuide)
    job = SimpleNamespace(
        guide=guide,
        processors=[TokenMaskProcessor(top_n_sigma=sigma)] if sigma else [],
        logprobs=False,
    )
    assert runner._driver_guide(job) is (enabled and not sigma)


def test_forced_lookahead_fault_does_not_fault_the_live_guide(monkeypatch):
    from yunshu_engine import settings
    from yunshu_engine.constraint_eos import GrammarDeadEndError

    class DeadLater(_ForcedGuide):
        def mask(self):
            if self.at >= 3:
                raise GrammarDeadEndError("future dead end")
            return super().mask()

    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *a, **kw: (
            True if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
        ),
    )
    lm = TinyLM()
    guide = DeadLater(list(range(7, 25)))
    d = make(lm)
    d.add(request(lm, [1, 2, 3], 18, guide=guide))
    while not d.batch.rows:
        d.step()
    row = d.batch.rows[0]
    at = guide.at
    assert d._forced_window(row, 8) == []
    assert guide.at == at and guide.fault is None


def test_generic_jump_forward_caches_do_not_enter_default_namespace(monkeypatch):
    from types import SimpleNamespace

    from yunshu_engine import settings

    lm = TinyLM()
    d = make(lm)
    packed = SimpleNamespace(layer=PackedLayer(lm), dflash=None)
    original = settings.get
    salts = []
    for enabled in (False, True, False):
        monkeypatch.setattr(
            settings,
            "get",
            lambda key, *a, **kw: (
                enabled if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
            ),
        )
        salts.append(
            (d.apc_salt(17, False), drv.RoundDriver.apc_salt(packed, 17, False))
        )
    assert salts[0] == salts[2]
    assert salts[0][0] != salts[1][0]
    assert salts[0][1] == salts[1][1]


def test_generic_resume_cache_namespace_excludes_old_packed_namespace():
    lm = TinyLM()
    d = make(lm)
    generic_salt = d.apc_salt(19, False)
    d.layer = PackedLayer(
        lm
    )  # the previous shared namespace, with identical arithmetic ids
    assert d.apc_salt(19, False) != generic_salt


def test_rows_and_requests_compare_by_identity_not_by_media_arrays():
    """2026-11-10 OCR A/B: with two media rows live, `r in self.rows` (RoundDriver._finish_join)
    ran the dataclass field-wise __eq__ over prompt_kwargs holding mx arrays and raised
    "Only length-1 arrays can be converted to Python scalars" -> HTTP 500 for concurrent
    image / audio / video requests."""
    import mlx.core as mx

    def req():
        return drv.Request(
            [1, 2, 3],
            4,
            prompt_kwargs={"pixel_values": mx.ones((2, 3))},
            embeds=mx.ones((3, 4)),
            media=True,
        )

    a, b = req(), req()
    assert a in [b, a] and b not in [a]
    rows = [drv._Row(r, None, None) for r in (a, b)]
    assert rows[1] in rows[::-1] and drv._Row(a, None, None) not in rows


class BatchSensitiveLM(TinyLM):
    """A model whose arithmetic depends on how many rows share the call (as a stock M=rows matmul or a
    padded-batch attention does): the hidden state is rounded differently by batch size."""

    def __call__(self, inputs, inputs_embeds=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.embed(inputs)
        b = inputs.shape[0]
        inputs_embeds = inputs_embeds * (1.0 + 0.3 * (b - 1))
        return super().__call__(inputs, inputs_embeds=inputs_embeds, **kw)


def test_decode_rows_never_share_an_upstream_call():
    lm = BatchSensitiveLM()
    alone = [solo(lm, ids, 12) for ids in PROMPTS]
    LOG.clear()
    reqs = [request(lm, ids, 12, handle=i) for i, ids in enumerate(PROMPTS)]
    together = run(make(lm), reqs)
    assert [tokens(together, i) for i in range(3)] == alone
    assert {e["B"] for e in LOG} == {1}
