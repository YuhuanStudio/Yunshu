"""Round driver: cost-aware allocation, and row invariance on a tiny random
4-bit Qwen3.5 (lane projections): a greedy row's tokens are the same alone,
batched with other rows, joining mid-flight, with MTP drafts or without, and
with drafts that are all accepted (oracle) or all rejected (random head)."""

import pytest

from yunshu_engine.round_driver.allocate import CostCurve, allocate, chain

mx = pytest.importorskip("mlx.core")


def test_chain_probabilities():
    assert chain([0.8, 0.5], 3) == pytest.approx([0.8, 0.4, 0.2])


def test_allocate_flat_cost_drafts_deep_steep_cost_does_not():
    probs = [chain([0.8] * 7, 7), chain([0.8] * 7, 7)]
    flat = allocate(2, probs, lambda rows: 10.0, 16)
    assert flat == [7, 7]
    steep = allocate(2, probs, lambda rows: 10.0 * rows, 16)
    assert steep == [0, 0]
    # bandwidth-bound up to 8 rows, then linear: drafts stop around there
    knee = allocate(2, probs, lambda rows: 10.0 if rows <= 8 else 10.0 * rows / 8, 16)
    assert sum(knee) + 2 <= 9 and sum(knee) >= 5


def test_allocate_prefers_likely_rows():
    probs = [chain([0.9] * 3, 3), chain([0.2] * 3, 3)]
    got = allocate(2, probs, lambda rows: 10.0 if rows <= 4 else 1e9, 8)
    assert got == [2, 0]


def test_cost_curve_interpolates_and_extrapolates():
    c = CostCurve({1: 10.0, 9: 18.0})
    assert c(5) == pytest.approx(14.0)
    assert c(17) == pytest.approx(26.0)
    assert CostCurve()(3) == 3.0


# ── invariance on a tiny model ──────────────────────────────────────────────


def _tiny():
    nn = pytest.importorskip("mlx.nn")
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )

    from yunshu_engine.kernels import lane_linear

    cfg = dict(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=512,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=256,
        tie_word_embeddings=False,
    )
    mx.random.seed(3)
    lm = LanguageModel(TextConfig(**cfg))
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=4)
    assert lane_linear.convert(lm)["skipped"] == []
    drafter = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(text_config={**cfg, "mtp_num_hidden_layers": 1})
    )
    drafter.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters(), drafter.parameters())
    return lm, drafter


@pytest.fixture(scope="module")
def tiny():
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    return _tiny()


PROMPTS = [
    list(range(3, 8)),
    [(7 * i + 11) % 500 for i in range(700)],  # two prefill chunks
    [(13 * i + 5) % 500 for i in range(40)],
]
N = 24


class _Budget:
    """Forces tokens 1, 2 after the 6th generated token (like the thinking
    budget's "\\n</think>")."""

    def __init__(self):
        self.n, self.queue, self.forced = 0, [], None

    def __call__(self, tok):
        self.n += 1
        if self.n == 6:
            self.queue = [1, 2]
        self.forced = self.queue.pop(0) if self.queue else None

    def pop_forced_token_id(self):
        f, self.forced = self.forced, None
        return f


def _run(
    lm, drafter, prompts, *, stagger=False, oracle=None, budget=False, wrong=False
):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(lm, drafter=drafter, stop_tokens=set())
    if oracle is not None:
        # drafts that are always right: the reference continuation
        def draft(rows, heads, depths, first=None):
            out = []
            for row, depth in zip(rows, depths, strict=True):
                ref = oracle[row.req.handle]
                got = list(ref[row.generated : row.generated + depth])
                if wrong and got:
                    # right up to a position that varies by row and step
                    j = (row.generated + row.req.handle) % len(got)
                    got[j] = (got[j] + 1) % 500
                out.append(mx.array(got, dtype=mx.int32))
            return out

        d.head.draft = draft
    out = {i: [] for i in range(len(prompts))}
    todo = list(range(len(prompts)))

    def add(i):
        d.add(Request(prompts[i], N, handle=i, budget=_Budget() if budget else None))

    if not stagger:
        for i in todo:
            add(i)
        todo = []
    steps = 0
    while d.busy() or todo:
        if todo and steps % 2 == 0:
            add(todo.pop(0))
        for e in d.step():
            out[e.handle].append(e.token)
        steps += 1
    return [out[i] for i in range(len(prompts))], d


def test_greedy_rows_invariant(tiny):
    lm, drafter = tiny
    ref = [_run(lm, None, [p])[0][0] for p in PROMPTS]
    assert all(len(r) == N for r in ref)
    assert _run(lm, None, PROMPTS)[0] == ref
    assert _run(lm, drafter, PROMPTS)[0] == ref
    assert _run(lm, drafter, PROMPTS, stagger=True)[0] == ref
    got, d = _run(lm, drafter, PROMPTS, oracle=dict(enumerate(ref)))
    assert got == ref
    # oracle drafts land: far fewer steps than tokens
    assert d.accepted > N and d.steps < N + 6


def test_partially_accepted_windows_invariant(tiny):
    """Rows keep different prefixes of windows of different lengths: KV
    lengths, GDN state and conv window continue from the kept position."""
    lm, drafter = tiny
    ref = [_run(lm, None, [p])[0][0] for p in PROMPTS]
    for stagger in (False, True):
        got, d = _run(
            lm,
            drafter,
            PROMPTS,
            stagger=stagger,
            oracle=dict(enumerate(ref)),
            wrong=True,
        )
        assert got == ref
        assert 0 < d.accepted < d.drafted


def test_slots_grow_and_recycle(tiny):
    """Ten rows join over time (slots double, keys cross the buffer capacity,
    early finishers free slots for later joins): each row's tokens equal its
    solo run."""
    lm, drafter = tiny
    prompts = [
        [(11 * i + 3 * j + 1) % 500 for j in range(n)]
        for i, n in enumerate([30, 500, 12, 505, 60, 20, 300, 8, 480, 45])
    ]
    ref = [_run(lm, None, [p])[0][0] for p in prompts]
    got, _ = _run(lm, drafter, prompts, stagger=True)
    assert got == ref


def test_thinking_budget_forcing_invariant(tiny):
    lm, drafter = tiny
    ref = [_run(lm, None, [p], budget=True)[0][0] for p in PROMPTS]
    assert all(r[6:8] == [1, 2] for r in ref)
    assert _run(lm, drafter, PROMPTS, stagger=True, budget=True)[0] == ref


def test_every_step_evaluates_the_caches_it_advanced(tiny, monkeypatch):
    """A prompt chunk that emits no token still ends its step evaluated: no
    lazy graph carries over to the next chunk (a long prompt otherwise builds
    one graph over all its chunks and every KV buffer version)."""
    from yunshu_engine.round_driver import driver as drv

    lm, drafter = tiny
    seen: list[set] = []
    real_eval = mx.eval

    def spy(*arrays):
        flat = []
        for a in arrays:
            flat.extend(a if isinstance(a, (list, tuple)) else [a])
        seen[-1].update(id(a) for a in flat)
        return real_eval(*arrays)

    monkeypatch.setattr(drv.mx, "eval", spy)
    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    long_prompt = [
        (3 * i + 1) % 500 for i in range(drv.IDLE_BUDGET + 3 * drv.CHUNK + 7)
    ]
    d.add(drv.Request(long_prompt, 2, handle=0))
    row = d.rows[0]
    prefill_steps = 0
    while row.pending is None:
        seen.append(set())
        d.step()
        prefill_steps += 1
        if row.pending is None:  # mid-prompt: nothing emitted this step
            for buf in drv.cache_buffers(row.cache) + drv.cache_buffers(row.mtp_cache):
                assert id(buf) in seen[-1]
    assert prefill_steps >= 2


def test_prefill_chunk_sets_the_span_a_decoding_row_waits_behind(tiny, monkeypatch):
    """While a row decodes, a prefill step is one span of the configured chunk; a
    smaller chunk means more, shorter prefill steps (decode steps in between)."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    steps: list[list[int]] = []
    real_forward = drv.lane_forward

    def spy(model, seg, **kw):
        steps.append([seg.length])
        return real_forward(model, seg, **kw)

    monkeypatch.setattr(drv, "lane_forward", spy)

    def prefill_spans(chunk):
        d = drv.RoundDriver(lm, stop_tokens=set(), chunk=chunk)
        d.add(drv.Request([1, 2, 3], 200, handle="decoding"))
        while d.rows[0].pending is None:
            d.step()
        steps.clear()
        d.add(drv.Request([(7 * i + 1) % 500 for i in range(300)], 1, handle="prompt"))
        while len(d.rows) > 1 or d.rows[0].req.handle == "prompt":
            d.step()
            if d.rows[0].req.handle == "prompt" and d.rows[0].pending is not None:
                break
        return [sum(x) for x in steps if x]

    small, large = prefill_spans(32), prefill_spans(128)
    assert max(small) <= 32 and max(large) <= 128
    assert len(small) > len(large)


def test_each_prefill_segment_runs_the_lane_forward(tiny, monkeypatch):
    """A step holding two segments of one row (a full atom, then the partial atom
    at the prompt's end) runs the lane forward once per segment; it must never
    fall back to the packed multi-segment forward (stock matmul: slower and not
    the arithmetic the lane runs)."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    lengths: list[int] = []
    real = drv.lane_forward

    def spy(model, seg, **kw):
        lengths.append(seg.length)
        return real(model, seg, **kw)

    monkeypatch.setattr(drv, "lane_forward", spy)
    d = drv.RoundDriver(lm, stop_tokens=set(), chunk=32)
    d.add(drv.Request([(7 * i + 1) % 500 for i in range(40)], 1, handle=0))
    d.step()
    assert lengths == [32, 8]


def test_prefill_atoms_merge_up_to_checkpoints_and_the_tail():
    """A prefill segment is a run of atoms (a restore off the grid included) that
    stops at a checkpoint and before the prompt's tail, which runs apart."""
    from types import SimpleNamespace

    from yunshu_engine.round_driver import driver as drv

    d = drv.RoundDriver.__new__(drv.RoundDriver)
    d.chunk = 512
    d.long_from, d.long_chunk = 1 << 60, 2048
    ids = list(range(512 * 6 + 40))
    row = SimpleNamespace(req=SimpleNamespace(ids=ids), ckpts=[])
    assert d._run_end(row, 0, 512) == 512  # budget of one atom
    assert d._run_end(row, 0, 4096) == 512 * 6  # the tail atom stays apart
    assert d._run_end(row, 512 * 6, 4096) == len(ids)
    row.ckpts = [1024, len(ids) - 1]
    assert d._run_end(row, 0, 4096) == 1024  # stops at the checkpoint
    assert d._run_end(row, 1024, 4096) == 512 * 6
    assert d._run_end(row, 512 * 6, 4096) == len(ids) - 1
    row.ckpts = []
    assert d._run_end(row, 600, 4096) == 512 * 6  # off the grid: merges to the tail
    assert d._run_end(row, 1024, 1024) == 2048
    # a restored prompt's remainder off the grid is one forward up to its checkpoint
    long_ids = list(range(7248))
    row = SimpleNamespace(req=SimpleNamespace(ids=long_ids), ckpts=[7232])
    d.long_from = 1 << 60
    assert d._run_end(row, 6912, 2048) == 7232
    assert d._run_end(row, 7232, 2048) == 7248  # the tail stays apart


# ── keyed sampling: sampled rows draft and stay on their own stream ─────────


def _sampled_run(
    lm, drafter, prompts, seeds, *, stagger=False, oracle=None, wrong=False
):
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.vlm_batch_runner import RowParams

    d = RoundDriver(lm, drafter=drafter, stop_tokens=set())
    if oracle is not None:

        def draft(rows, heads, depths, first=None):
            out = []
            for row, depth in zip(rows, depths, strict=True):
                ref = oracle[row.req.handle]
                got = list(ref[row.generated : row.generated + depth])
                if wrong and got:
                    j = (row.generated + row.req.handle) % len(got)
                    got[j] = (got[j] + 1) % 500
                out.append(mx.array(got, dtype=mx.int32))
            return out

        d.head.draft = draft
    out = {i: [] for i in range(len(prompts))}
    todo = list(range(len(prompts)))

    def add(i):
        d.add(
            Request(
                prompts[i],
                N,
                sampling=RowParams(1.0, 0.95, 0, 0.0, seeds[i]),
                handle=i,
            )
        )

    if not stagger:
        for i in todo:
            add(i)
        todo = []
    steps = 0
    while d.busy() or todo:
        if todo and steps % 2 == 0:
            add(todo.pop(0))
        for e in d.step():
            out[e.handle].append(e.token)
        steps += 1
    return [out[i] for i in range(len(prompts))], d


def test_sampled_rows_keyed_positions_invariant(tiny):
    """A sampled row's stream depends on its seed and generation index only:
    the same alone, batched, staggered, drafting with every draft right, and
    drafting with partial acceptance (positions advance by the kept tokens)."""
    lm, drafter = tiny
    seeds = [11, 22, 33]
    ref = [
        _sampled_run(lm, None, [p], [s])[0][0]
        for p, s in zip(PROMPTS, seeds, strict=True)
    ]
    assert all(len(r) == N for r in ref)
    assert _sampled_run(lm, None, PROMPTS, seeds)[0] == ref
    assert _sampled_run(lm, drafter, PROMPTS, seeds, stagger=True)[0] == ref
    # another seed draws another stream
    assert _sampled_run(lm, None, [PROMPTS[0]], [99])[0][0] != ref[0]
    got, d = _sampled_run(lm, drafter, PROMPTS, seeds, oracle=dict(enumerate(ref)))
    assert got == ref
    assert d.accepted > N and d.steps < N + 6
    for stagger in (False, True):
        got, d = _sampled_run(
            lm,
            drafter,
            PROMPTS,
            seeds,
            stagger=stagger,
            oracle=dict(enumerate(ref)),
            wrong=True,
        )
        assert got == ref
        assert 0 < d.accepted < d.drafted


def test_head_readout_uses_the_reduced_vocabulary_for_one_row_only(tiny):
    """The lane's reduced-vocabulary readout serves a lone drafting row; with
    several rows the full head reads out (drafts only propose either way)."""
    from yunshu_engine.round_driver.driver import RoundDriver

    lm, drafter = tiny
    d = RoundDriver(lm, drafter=drafter, stop_tokens=set())
    calls = []
    d.head.vocab = object()
    drafter._greedy_token = lambda h: calls.append(1) or mx.zeros(h.shape[0], mx.int32)
    h = mx.random.normal((3, lm.args.hidden_size))
    d.head.readout(h, single=False)
    assert not calls
    d.head.readout(h, single=True)
    assert calls


def test_prefill_serves_the_oldest_long_prompt_first(tiny, monkeypatch):
    """Two long prompts are not interleaved: the first one's prefill finishes
    before the second one's starts (its first token is not delayed by the
    other's prompt), even when a checkpoint ends A's span with budget left."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    d = drv.RoundDriver(lm, stop_tokens=set(), chunk=64)
    d.idle_budget = 256
    long_a = [(5 * i + 1) % 500 for i in range(1000)]
    long_b = [(7 * i + 3) % 500 for i in range(1000)]
    short = [(11 * i + 2) % 500 for i in range(40)]
    d.add(drv.Request(long_a, 1, handle="a", use_apc=False))
    d.add(drv.Request(long_b, 1, handle="b", use_apc=False))
    d.add(drv.Request(short, 1, handle="c", use_apc=False))
    rows = {r.req.handle: r for r in d.rows}
    rows["a"].ckpts = [128, 512]  # a checkpoint cuts A's span with budget left
    d.step()
    assert rows["c"].done == 40 and rows["a"].done == 0  # the short prompt goes first
    d.step()
    assert rows["a"].done == 128
    assert rows["b"].done == 0  # long B waits behind long A
    while rows["a"].pending is None:
        assert rows["b"].done == 0
        d.step()


def test_single_segment_projection_is_not_copied():
    from yunshu_engine.round_driver import forward as fw

    mx = pytest.importorskip("mlx.core")
    out = mx.ones((1, 4, 3))

    def lin(x):
        return out

    got = fw._project(lin, mx.zeros((1, 4, 2)), [(0, 4)], [True])
    assert got is out
    assert fw._cat([out]) is out


def test_idle_prefill_step_is_short_enough_for_a_new_arrival_to_join(tiny):
    """A request arriving mid-step waits for the step to end: with 4096 tokens
    per step that was ~3.4 s on the 27B (c=2 mixed-length mean TTFT +21%). One
    idle step is IDLE_BUDGET tokens (2048, main's prefill step size)."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    d = drv.RoundDriver(lm, stop_tokens=set(), chunk=64)
    ids = [(5 * i + 1) % 500 for i in range(3000)]
    d.add(drv.Request(ids, 1, handle="a", use_apc=False))
    d.step()
    assert d.rows[0].done <= drv.IDLE_BUDGET + 64


def test_atoms_grow_with_the_absolute_position():
    """Beyond ``long_from`` atoms are ``long_chunk`` tokens (fewer attention
    passes over a long KV); the plan is a function of the position alone."""
    from types import SimpleNamespace

    from yunshu_engine.round_driver import driver as drv

    d = drv.RoundDriver.__new__(drv.RoundDriver)
    d.chunk = 512
    d.long_from, d.long_chunk = 2048, 1024
    ids = list(range(2048 + 1024 * 3 + 40))
    row = SimpleNamespace(req=SimpleNamespace(ids=ids), ckpts=[])
    assert [d.atom(p) for p in (0, 1536, 2048, 5000)] == [512, 512, 1024, 1024]
    assert d._span_end(row, 0) == 512
    assert d._span_end(row, 1536) == 2048
    assert d._span_end(row, 2048) == 3072
    assert d._span_end(row, 2500) == 3072  # restored off the grid: alone
    assert d._run_end(row, 0, 8192) == 2048  # 512-atoms merge up to the switch
    assert d._run_end(row, 2048, 8192) == 2048 + 1024 * 3
    assert d._run_end(row, 2048, 1024) == 3072


def test_sampled_bf16_rows_do_not_depend_on_drafting(tiny):
    """bf16 logits: the drafting row (keyed, float32 normalization) and the plain row draw the
    same stream."""
    from mlx.utils import tree_map

    lm, drafter = tiny
    original = lm.parameters()
    lm.update(
        tree_map(
            lambda a: (
                a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a
            ),
            original,
        )
    )
    try:
        seeds = [11, 22, 33]
        ref = [
            _sampled_run(lm, None, [p], [s])[0][0]
            for p, s in zip(PROMPTS, seeds, strict=True)
        ]
        got, _ = _sampled_run(lm, drafter, PROMPTS, seeds, oracle=dict(enumerate(ref)))
        assert got == ref
    finally:
        lm.update(original)


def test_short_remainder_prefills_before_a_long_prompt(tiny):
    """A prompt with few rows left (a restored one) is not queued behind a cold
    prompt's span: after one step it is done and the long one has not started."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    d = drv.RoundDriver(lm, stop_tokens=set(), chunk=64)
    long_ids = [(5 * i + 1) % 500 for i in range(1500)]
    short_ids = [(3 * i + 2) % 500 for i in range(40)]
    d.add(drv.Request(long_ids, 1, handle="long", use_apc=False))
    d.add(drv.Request(short_ids, 5, handle="short", use_apc=False))
    d.step()
    done = {r.req.handle: r.done for r in d.rows}
    assert done["short"] == 40 and done["long"] == 0


def test_finished_lone_slot_buffers_go_to_the_restore_pool(tiny):
    """When the last row leaves a one-slot batch its K/V buffers are offered to the restore
    pool (a later cache hit overwrites them) instead of just being dropped."""
    from types import SimpleNamespace

    from yunshu_engine.kernels import buffer_cache
    from yunshu_engine.round_driver import batch as rb

    buffer_cache.clear_restore_buffers()
    prev = dict(buffer_cache._STATE)
    buffer_cache._STATE["limit"] = 1 << 30
    try:
        b = rb.DecodeBatch.__new__(rb.DecodeBatch)
        b.linear, b.rows, b.slots = [False], [], rb.Slots(1)
        b.state, b.conv = [None], [None]
        k = mx.zeros((1, 2, 64, 4), dtype=mx.float32)
        row = SimpleNamespace(n=8, cache=[SimpleNamespace(keys=k, values=k)], slot=None)
        b.join([row])
        buffer_cache.clear_restore_buffers()
        b.leave([row])
        assert buffer_cache.restore_buffer_bytes() > 0 and not b.slots.arrays()
    finally:
        buffer_cache.clear_restore_buffers()
        buffer_cache._STATE.update(prev)


@pytest.mark.parametrize("length", [40, 200])
def test_cancelled_rehearsal_prefill_is_preempted_inside_the_step(tiny, length):
    """A lone preemptible (rehearsal) row whose cancel event is set stops inside its prefill
    step (layer-group check, kernels/preempt.py): no events, the row leaves the driver, and a
    normal row afterwards still prefills and decodes."""
    import threading
    from types import SimpleNamespace

    from yunshu_engine.kernels import preempt
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    assert preempt.install() and preempt.supported()
    ev = threading.Event()
    ev.preemptible = True
    handle = SimpleNamespace(cancel_event=ev)
    d = drv.RoundDriver(lm, stop_tokens=set(), chunk=64)
    # 200 rows: one step holds the row's span and its tail (two segments of ONE row)
    ids = [(5 * i + 1) % 500 for i in range(length)]
    d.add(drv.Request(ids, 4, handle=handle, use_apc=False))
    ev.set()
    assert d.step() == []
    assert not d.rows
    # a plain row (no preemptible event) is untouched by the scope
    d.add(drv.Request(ids, 4, handle=SimpleNamespace(cancel_event=None), use_apc=False))
    out = []
    while d.busy():
        out += d.step()
    assert len(out) == 4


def _row_tokens(lm, drafter, prompt, processors=None, others=()):
    from yunshu_engine.round_driver import driver as drv

    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    d.add(drv.Request(prompt, N, handle="row", processors=processors or []))
    for i, p in enumerate(others):
        d.add(drv.Request(p, N, handle=f"other{i}"))
    toks = {}
    while d.busy():
        for e in d.step():
            toks.setdefault(e.handle, []).append(e.token)
    return toks["row"]


@pytest.mark.parametrize("draft", [False, True])
def test_driver_rows_apply_each_processor_type(tiny, draft):
    """Logit bias, token suppression and repetition/frequency penalties act on a driver row
    (one token a round through its processors), the same alone and beside other rows."""
    from yunshu_engine.vlm_batch_runner import (
        LogitBiasProcessor,
        TokenMaskProcessor,
        build_penalty_processors,
    )

    lm, drafter = tiny
    dr = drafter if draft else None
    prompt = PROMPTS[0]
    ref = _row_tokens(lm, dr, prompt)
    top = max(set(ref), key=ref.count)

    biased = _row_tokens(lm, dr, prompt, [LogitBiasProcessor({7: 1e4})])
    assert biased == [7] * N

    masked = _row_tokens(lm, dr, prompt, [TokenMaskProcessor(suppress=[top])])
    assert top not in masked and masked != ref

    penalised = _row_tokens(
        lm, dr, prompt, build_penalty_processors(1.0, 2.0, 1.0, None)
    )
    assert penalised != ref or len(set(ref)) == len(ref)
    again = _row_tokens(
        lm,
        dr,
        prompt,
        build_penalty_processors(1.0, 2.0, 1.0, None),
        others=PROMPTS[1:3],
    )
    assert again == penalised  # row invariance with other rows beside it


def test_driver_rows_publish_spec_counters_and_phase_counts(tiny):
    """x_yunshu / console fields: a driver row fills its RunStats speculation counters, and
    the runner's phase counts include driver jobs."""
    from types import SimpleNamespace

    from yunshu_engine.round_driver import driver as drv
    from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner

    lm, drafter = tiny
    stats = RunStats()
    handle = SimpleNamespace(stats=stats)
    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    d.add(drv.Request(PROMPTS[1], N, handle=handle))
    while d.busy():
        d.step()
    assert stats.spec_rounds > 0 and stats.spec_drafted > 0
    assert 0 <= stats.spec_accepted <= stats.spec_drafted

    runner = VLMBatchRunner(object(), object(), apc_manager=None)
    job = SimpleNamespace(stats=RunStats(), terminal=False)
    runner._driver_jobs = {1: job}
    counts = runner.phase_counts()
    assert sum(counts.values()) >= 1


def test_cancel_mid_decode_leaves_the_other_row_unchanged(tiny):
    """A row removed while decoding (client disconnect) frees its slot; the row beside it keeps
    producing exactly the tokens it produces alone."""
    from yunshu_engine.round_driver import driver as drv

    lm, drafter = tiny
    solo = _row_tokens(lm, drafter, PROMPTS[1])
    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    d.add(drv.Request(PROMPTS[1], N, handle="keep"))
    d.add(drv.Request(PROMPTS[2], 4 * N, handle="drop"))
    got: list[int] = []
    steps = 0
    while d.busy():
        for e in d.step():
            if e.handle == "keep":
                got.append(e.token)
        steps += 1
        if steps == 6:
            d.remove("drop")
    assert got == solo
    assert not d.rows


def test_decode_batch_join_waits_until_after_the_first_token(tiny):
    """The prefill step returns the first token before the row joins the decode batch (the join,
    including the drafter's context ring, would otherwise sit on the first-token path); the next
    step joins it and decoding continues with the tokens a run alone produces."""
    from yunshu_engine.round_driver import driver as drv

    lm, drafter = tiny
    solo = _row_tokens(lm, drafter, PROMPTS[1])
    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    d.add(drv.Request(PROMPTS[1], N, handle="row"))
    first = []
    while not first:
        first = [e.token for e in d.step()]
    assert not d.batch.rows and len(d._pending_join) == 1
    got = list(first)
    while d.busy():
        got += [e.token for e in d.step()]
    assert d.batch.rows == [] and got == solo


def _embed_request(lm, ids, positions=None):
    ids_mx = mx.array(ids, dtype=mx.int32)
    embeds = lm.model.embed_tokens(ids_mx)
    if positions is None:
        positions = mx.tile(mx.arange(len(ids), dtype=mx.int32)[None, :], (3, 1))
    return embeds, positions


def test_media_prompt_embeds_and_positions_match_the_token_path(tiny):
    """A prompt carried as embeddings + mRoPE positions (what a media prompt is once its visual
    features are scattered in) prefills and generates exactly what the same prompt does as tokens
    when the embeddings are the token embeddings and the positions the plain sequence; positions
    with a gap (an image grid's jump) are consumed: the hidden states differ."""
    from yunshu_engine.round_driver import driver as drv
    from yunshu_engine.round_driver.forward import Segment, lane_forward

    lm, drafter = tiny
    prompt = PROMPTS[1]
    ref = _row_tokens(lm, drafter, prompt)
    embeds, positions = _embed_request(lm, prompt)
    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    d.add(drv.Request(prompt, N, handle="m", embeds=embeds, position_ids=positions))
    got = []
    while d.busy():
        got += [e.token for e in d.step() if e.handle == "m"]
    assert got == ref

    def hidden(pos):
        seg = Segment(
            lm.make_cache(),
            mx.array(prompt, dtype=mx.int32),
            embeds=embeds,
            position_ids=pos,
        )
        return lane_forward(lm, seg)

    gap = positions + (mx.arange(len(prompt)) >= 10).astype(mx.int32) * 7
    assert bool(mx.array_equal(hidden(positions), hidden(positions)))
    assert not bool(mx.array_equal(hidden(positions), hidden(gap)))


def _media_row(lm, drafter, prompt, jump, delta, others=()):
    """Tokens of a media-style row (embeddings + positions with a jump at index 10, decode offset
    ``delta``) run through the driver, alone or beside text rows."""
    from yunshu_engine.round_driver import driver as drv

    positions = mx.tile(mx.arange(len(prompt), dtype=mx.int32)[None, :], (3, 1))
    positions = positions + (mx.arange(len(prompt)) >= 10).astype(mx.int32) * jump
    embeds = lm.model.embed_tokens(mx.array(prompt, dtype=mx.int32))
    d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
    d.add(
        drv.Request(
            prompt,
            N,
            handle="m",
            embeds=embeds,
            position_ids=positions,
            rope_delta=delta,
        )
    )
    for i, p in enumerate(others):
        d.add(drv.Request(p, N, handle=f"o{i}"))
    toks: dict = {}
    while d.busy():
        for e in d.step():
            toks.setdefault(e.handle, []).append(e.token)
    return toks["m"]


def test_media_row_decode_positions_follow_the_row_delta(tiny):
    """Per-row mRoPE offset in decode: a media row with a position jump keeps producing the same
    tokens alone and beside text rows (which have offset 0), and the offset is consumed (a wrong
    delta changes the continuation)."""
    lm, drafter = tiny
    prompt = PROMPTS[1]
    alone = _media_row(lm, drafter, prompt, jump=7, delta=7)
    beside = _media_row(lm, drafter, prompt, jump=7, delta=7, others=PROMPTS[2:3])
    assert alone == beside
    wrong = _media_row(lm, drafter, prompt, jump=7, delta=900)
    assert wrong != alone


def test_media_row_drafts_with_the_mtp_head_and_matches_plain_decode(tiny):
    """The MTP head absorbs a media prompt at its real mRoPE positions and decodes at the row's
    offset: speculation on == off for a media row (positions with a jump, delta 7)."""
    lm, drafter = tiny
    prompt = PROMPTS[1]
    plain = _media_row(lm, None, prompt, jump=7, delta=7)
    drafted = _media_row(lm, drafter, prompt, jump=7, delta=7)
    assert drafted == plain
    beside = _media_row(lm, drafter, prompt, jump=7, delta=7, others=PROMPTS[2:3])
    assert beside == plain
