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
    """The 27B's kernel set on a tiny Qwen3.5 (128-dim GDN heads, batch-invariant verify kernels
    installed, lane projections) plus an MTP head: every decode forward runs the production arithmetic."""
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )
    from scripts.research.phaseb_verify import tiny as verify_tiny

    lm, _ = verify_tiny()
    # production serves with the dense-lane route off (a prompt tail turns it on for its own forward)
    from yunshu_engine.kernels import ragged_kv

    ragged_kv.set_dense_lane(False)
    cfg = dict(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=512,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=256,
        tie_word_embeddings=False,
    )
    mx.random.seed(3)
    drafter = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(text_config={**cfg, "mtp_num_hidden_layers": 1})
    )
    drafter.set_dtype(mx.bfloat16)
    mx.eval(drafter.parameters())
    return lm, drafter


@pytest.fixture(autouse=True)
def one_atom_prefill_segments(monkeypatch, request):
    """The tiny model's prefill is not split-invariant (the 27B's NAX / BQ32 / chunked-GDN prefill is):
    a segment that merges atoms only while no row decodes would make a row's solo and beside-others
    prefills differ. One atom per segment pins prefill, so these tests isolate the decode forwards."""
    from yunshu_engine.round_driver.layer import PackedLayer

    if any(k in request.node.name for k in ("atoms", "prompt_tail")):
        return  # these tests are about how segments merge
    monkeypatch.setattr(
        PackedLayer,
        "span_end",
        lambda self, row, start, budget, *, atom, scoring=False: self.cut(
            row, start, atom
        ),
    )


@pytest.fixture(scope="module", autouse=True)
def restore_probe_lane_mode():
    # the production tiny model enables a process-global dense-lane route
    from yunshu_engine.kernels import ragged_kv

    previous = ragged_kv._STATE["dense_lane"]
    yield
    ragged_kv.set_dense_lane(previous)


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
            # the head's prompt prefill stays queued (bounded by HEAD_LAZY_MAX) unless it is over the bound
            head = [] if d._lazy_ids.get(id(row)) else drv.cache_buffers(row.mtp_cache)
            for buf in drv.cache_buffers(row.cache) + head:
                assert id(buf) in seen[-1]
    assert prefill_steps >= 2


def test_head_prefill_stays_queued_past_the_first_token_and_changes_nothing(
    tiny, monkeypatch
):
    """The MTP head's prompt prefill is not evaluated before a packed row's first token (it ran on
    the cold-TTFT path); the head cache it builds equals the per-step one, and the tokens of solo,
    concurrent and spec-off runs are the same either way."""
    from yunshu_engine.round_driver import driver as drv

    lm, drafter = tiny
    real_eval = mx.eval

    def first_step(bound):
        monkeypatch.setattr(drv, "HEAD_LAZY_MAX", bound)
        seen: set = set()

        def spy(*arrays):
            for a in arrays:
                seen.update(id(x) for x in (a if isinstance(a, (list, tuple)) else [a]))
            return real_eval(*arrays)

        d = drv.RoundDriver(lm, drafter=drafter, stop_tokens=set())
        d.add(drv.Request(PROMPTS[1], N, handle=0))
        row = d.rows[0]
        monkeypatch.setattr(drv.mx, "eval", spy)
        while row.pending is None:
            seen.clear()
            d.step()
        monkeypatch.setattr(drv.mx, "eval", real_eval)
        bufs = drv.cache_buffers(row.mtp_cache)
        evaluated = any(id(b) in seen for b in bufs)
        mx.eval(*bufs)
        return d, evaluated, [mx.array(b) for b in bufs]

    d1, lazy_evaluated, lazy_cache = first_step(drv.HEAD_LAZY_MAX)
    assert all(d1._lazy_ids.values()) and not lazy_evaluated
    d2, eager_evaluated, eager_cache = first_step(0)
    assert not any(d2._lazy_ids.values()) and eager_evaluated
    for a, b in zip(lazy_cache, eager_cache, strict=True):
        assert mx.array_equal(a, b).item()
    monkeypatch.setattr(drv, "HEAD_LAZY_MAX", 512 << 20)
    lazy = _run(lm, drafter, PROMPTS)[0]
    monkeypatch.setattr(drv, "HEAD_LAZY_MAX", 0)
    assert _run(lm, drafter, PROMPTS)[0] == lazy
    assert _run(lm, None, PROMPTS)[0] == lazy
    assert [_run(lm, drafter, [p])[0][0] for p in PROMPTS] == lazy


def test_prefill_chunk_sets_the_span_a_decoding_row_waits_behind(tiny, monkeypatch):
    """While a row decodes, a prefill step is one span of the configured chunk; a
    smaller chunk means more, shorter prefill steps (decode steps in between)."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    steps: list[list[int]] = []
    from yunshu_engine.round_driver import forward as fwd

    real_forward = fwd.lane_forward

    def spy(model, seg, **kw):
        steps.append([seg.length])
        return real_forward(model, seg, **kw)

    monkeypatch.setattr(fwd, "lane_forward", spy)

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
    from yunshu_engine.round_driver import forward as fwd

    real = fwd.lane_forward

    def spy(model, seg, **kw):
        lengths.append(seg.length)
        return real(model, seg, **kw)

    monkeypatch.setattr(fwd, "lane_forward", spy)
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
    from yunshu_engine.round_driver.layer import PackedLayer

    layer = PackedLayer.__new__(PackedLayer)  # the packed prefill plan (layer.py)
    d.chunk = 512
    d.long_from, d.long_chunk = 1 << 60, 2048
    ids = list(range(512 * 6 + 40))
    row = SimpleNamespace(req=SimpleNamespace(ids=ids), ckpts=[])
    assert layer.span_end(row, 0, 512, atom=d.atom) == 512  # budget of one atom
    assert (
        layer.span_end(row, 0, 4096, atom=d.atom) == 512 * 6
    )  # the tail atom stays apart
    assert layer.span_end(row, 512 * 6, 4096, atom=d.atom) == len(ids)
    row.ckpts = [1024, len(ids) - 1]
    assert layer.span_end(row, 0, 4096, atom=d.atom) == 1024  # stops at the checkpoint
    assert layer.span_end(row, 1024, 4096, atom=d.atom) == 512 * 6
    assert layer.span_end(row, 512 * 6, 4096, atom=d.atom) == len(ids) - 1
    row.ckpts = []
    assert (
        layer.span_end(row, 600, 4096, atom=d.atom) == 512 * 6
    )  # off the grid: merges to the tail
    assert layer.span_end(row, 1024, 1024, atom=d.atom) == 2048
    # a restored prompt's remainder off the grid is one forward up to its checkpoint
    long_ids = list(range(7248))
    row = SimpleNamespace(req=SimpleNamespace(ids=long_ids), ckpts=[7232])
    d.long_from = 1 << 60
    assert layer.span_end(row, 6912, 2048, atom=d.atom) == 7232
    assert layer.span_end(row, 7232, 2048, atom=d.atom) == 7248  # the tail stays apart


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


def test_short_prompts_do_not_wait_out_a_long_prefill(tiny, monkeypatch):
    """2026-11-11 (real-1104 mixed cell: 1x128K + 7x1K, shorts waited 207-243 s): a prompt far
    shorter than the long one admitted before it prefills first, and the long one still advances."""
    from yunshu_engine.round_driver import driver as drv

    lm, _ = tiny
    monkeypatch.setattr(drv, "SHORT_FIRST", 16)
    monkeypatch.setattr(drv, "SHORT_STREAK_MAX", 3)
    d = drv.RoundDriver(lm, stop_tokens=set(), chunk=64)
    d.idle_budget = 256
    d.add(
        drv.Request(
            [(5 * i + 1) % 500 for i in range(2000)], 1, handle="long", use_apc=False
        )
    )
    for k in range(6):
        d.add(
            drv.Request(
                [(3 * i + k) % 500 for i in range(100)],
                1,
                handle=f"s{k}",
                use_apc=False,
            )
        )
    rows = {r.req.handle: r for r in d.rows}
    d.step()
    assert rows["long"].done == 0 and rows["s0"].done == 100 and rows["s1"].done == 100
    for _ in range(12):
        d.step()
    assert all(rows[f"s{k}"].pending is not None for k in range(6))
    assert rows["long"].done > 0  # the streak guard let the oldest prompt advance


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
    from yunshu_engine.round_driver.layer import PackedLayer

    layer = PackedLayer.__new__(PackedLayer)  # the packed prefill plan (layer.py)
    d.chunk = 512
    d.long_from, d.long_chunk = 2048, 1024
    ids = list(range(2048 + 1024 * 3 + 40))
    row = SimpleNamespace(req=SimpleNamespace(ids=ids), ckpts=[])
    assert [d.atom(p) for p in (0, 1536, 2048, 5000)] == [512, 512, 1024, 1024]
    assert layer.cut(row, 0, d.atom) == 512
    assert layer.cut(row, 1536, d.atom) == 2048
    assert layer.cut(row, 2048, d.atom) == 3072
    assert layer.cut(row, 2500, d.atom) == 3072  # restored off the grid: alone
    assert (
        layer.span_end(row, 0, 8192, atom=d.atom) == 2048
    )  # 512-atoms merge up to the switch
    assert layer.span_end(row, 2048, 8192, atom=d.atom) == 2048 + 1024 * 3
    assert layer.span_end(row, 2048, 1024, atom=d.atom) == 3072


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


def test_media_row_decode_positions_follow_the_row_delta(tiny, monkeypatch):
    """Rows keep their own decode offset; rotary positions and K expose its use
    even when a random model's greedy argmax does not change."""
    import numpy as np

    lm, drafter = tiny
    target = next(layer.self_attn for layer in lm.model.layers if not layer.is_linear)
    cls = type(target)
    original = cls._prepare_projected_qkv
    seen = []

    def prepare(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        cache = kwargs.get("cache", args[3] if len(args) > 3 else None)
        pos = kwargs.get("position_ids", args[4] if len(args) > 4 else None)
        if self is target and cache is None and pos is not None:
            seen.append((pos.tolist(), np.array(out[1].astype(mx.float32))))
        return out

    monkeypatch.setattr(cls, "_prepare_projected_qkv", prepare)
    prompt = PROMPTS[1]
    alone = _media_row(lm, drafter, prompt, jump=7, delta=7)
    good_pos, good_k = seen[0]
    assert [axis[0][0] for axis in good_pos] == [len(prompt) + 7] * 3
    beside = _media_row(lm, drafter, prompt, jump=7, delta=7, others=PROMPTS[2:3])
    assert alone == beside
    seen.clear()
    wrong = _media_row(lm, drafter, prompt, jump=7, delta=900)
    wrong_pos, wrong_k = seen[0]
    assert [axis[0][0] for axis in wrong_pos] == [len(prompt) + 900] * 3
    assert wrong[0] == alone[0]  # identical prefill, before the decode delta is used
    assert not np.array_equal(good_k[..., 0, :], wrong_k[..., 0, :])


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


def test_prompt_tail_runs_apart_even_without_a_checkpoint(tiny, monkeypatch):
    """A prompt whose grid cell contains its last 64-aligned boundary is cut there with no checkpoint
    planned (main's absolute prompt steps do the same): 99 rows run as 64 + a 35-row tail, not as one
    99-row span. The media-prompt probe (lp3) found the driver running it as one span: first-token
    logprobs differed from main in 43 of 46 prompts while text prompts below 64 rows were identical."""
    from types import SimpleNamespace

    from yunshu_engine.round_driver import driver as drv

    d = drv.RoundDriver.__new__(drv.RoundDriver)
    from yunshu_engine.round_driver.layer import PackedLayer

    layer = PackedLayer.__new__(PackedLayer)  # the packed prefill plan (layer.py)
    d.chunk = 512
    d.long_from, d.long_chunk = 1 << 60, 2048

    def plan(n, ckpts=()):
        row = SimpleNamespace(
            req=SimpleNamespace(ids=list(range(n))), ckpts=list(ckpts)
        )
        spans, at = [], 0
        while at < n:
            end = layer.span_end(row, at, 2048, atom=d.atom)
            spans.append((at, end))
            at = end
        return spans

    assert plan(99) == [(0, 64), (64, 99)]
    assert plan(64) == [(0, 64)]  # nothing after the boundary: no tail
    assert plan(65) == [(0, 64), (64, 65)]
    assert plan(30) == [(0, 30)]  # below one block: a single span, as before
    assert plan(1100) == [(0, 1088), (1088, 1100)]
    # a planned checkpoint on the boundary gives the same plan
    assert plan(99, [64]) == [(0, 64), (64, 99)]

    lm, _ = tiny
    lengths: list[int] = []
    from yunshu_engine.round_driver import forward as fwd

    real = fwd.lane_forward

    def spy(model, seg, **kw):
        lengths.append(seg.length)
        return real(model, seg, **kw)

    monkeypatch.setattr(fwd, "lane_forward", spy)
    drv2 = drv.RoundDriver(lm, stop_tokens=set(), chunk=512)
    drv2.add(
        drv.Request([(7 * i + 1) % 500 for i in range(99)], 2, handle=0, use_apc=False)
    )
    drv2.step()
    assert lengths == [64, 35]


def test_short_tail_attends_through_the_dense_lane_kernel(tiny, monkeypatch):
    """A 1..8-row tail forward turns main's dense lane on for its duration (single-row KVCache
    attention of <= 8 query rows on the ragged token-tile kernel) and restores the previous state;
    9 rows and more keep the prefill attention untouched."""
    from yunshu_engine.kernels import ragged_kv
    from yunshu_engine.round_driver.forward import Segment, lane_forward

    lm, _ = tiny
    cache = lm.make_cache()
    lane_forward(
        lm,
        Segment(
            cache, mx.array([(5 * i + 1) % 500 for i in range(40)], dtype=mx.int32)
        ),
    )
    calls = []
    real = ragged_kv.set_dense_lane

    def spy(active):
        calls.append(bool(active))
        real(active)

    monkeypatch.setattr(ragged_kv, "set_dense_lane", spy)
    for rows, expect in (
        (1, [True, False]),
        (7, [True, False]),
        (8, [True, False]),
        (9, []),
        (30, []),
    ):
        calls.clear()
        c = lm.make_cache()
        lane_forward(
            lm,
            Segment(
                c, mx.array([(3 * i + 2) % 500 for i in range(rows)], dtype=mx.int32)
            ),
        )
        assert calls == expect, (rows, calls)
        assert ragged_kv.dense_lane_active() is False


@pytest.mark.parametrize("sampled", [False, True])
def test_presence_penalty_keeps_the_mtp_row_drafting_and_equal_plain_decode(sampled):
    """presence / frequency penalty on an MTP row: it drafts, each window row is penalized over the tokens
    fed up to it (the serial ``window_row`` counts), and the tokens equal the row decoded without drafting."""
    from yunshu_engine.kernels.ragged_attention import tile_ready
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.text_logit_processors import GeneratedTokenPenalty
    from yunshu_engine.vlm_batch_runner import RowParams

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    lm, drafter = _tiny()
    params = RowParams(0.7, 0.95, 20, 0.0, seed=1234) if sampled else None

    def run(draft, penalty=True, head=True):
        d = RoundDriver(lm, drafter=drafter if head else None, stop_tokens=set())
        # this tiny decode never repeats a token: a negative presence penalty (generated tokens
        # favored) is what makes the penalty change it
        procs = [GeneratedTokenPenalty(0.0, -8.0)] if penalty else []
        d.add(
            Request(
                PROMPTS[1], 48, handle=0, sampling=params, processors=procs, draft=draft
            )
        )
        toks = []
        while d.busy():
            toks += [e.token for e in d.step()]
        return toks, d

    free, _ = run(False, penalty=False)
    plain, _ = run(False)
    assert plain != free
    drafted, d = run(True)
    assert drafted == plain
    assert d.drafted > 0
    # a spec-off server (no head) still keys a sampled row: the penalty must apply there too
    off, _ = run(True, head=False)
    assert off == plain


def test_packed_prompt_logprobs(tiny):
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.round_driver.forward import Segment, lane_forward, logits

    lm, _ = tiny
    ids = PROMPTS[0]
    language = lm.language_model if hasattr(lm, "language_model") else lm
    h = lane_forward(
        language,
        Segment(language.make_cache(), mx.array(ids, mx.int32), 512, 16384, 2048),
    )
    ref = logits(language, h).astype(mx.float32)
    ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    driver = RoundDriver(lm, plain_verify=True, stop_tokens=set())
    driver.add(Request(ids, 2, prompt_logprobs=3, draft=False))
    events = []
    while driver.busy():
        events.extend(driver.step())
    scores = events[0].prompt_logprobs
    assert scores[0] is None
    assert len(scores) == len(ids)
    for i, entry in enumerate(scores[1:]):
        assert entry["logprob"] == float(ref[i, ids[i + 1]])
    assert events[1].prompt_logprobs is None


@pytest.mark.parametrize("drafting", [False, True])
def test_forced_windows_on_packed_mtp_layer_equal_serial(tiny, monkeypatch, drafting):
    from tests.unit.test_round_driver_generic import _ForcedGuide
    from yunshu_engine import settings
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    lm, head = tiny
    original = settings.get
    steps = []
    for enabled in (False, True):
        monkeypatch.setattr(
            settings,
            "get",
            lambda key, *a, **kw: (
                enabled if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
            ),
        )
        d = RoundDriver(lm, drafter=head if drafting else None, stop_tokens=set())
        scripts = [list(range(7, 25)), list(range(11, 22))]
        out = {0: [], 1: []}
        for i, script in enumerate(scripts):
            d.add(
                Request(
                    PROMPTS[i],
                    len(script),
                    handle=i,
                    guide=_ForcedGuide(script, vocab=512),
                    draft=drafting,
                )
            )
        while d.busy():
            for event in d.step():
                out[event.handle].append(event.token)
        assert out == dict(enumerate(scripts))
        steps.append(d.steps)
    assert steps[1] < steps[0]


def test_a_narrow_window_never_costs_more_than_a_wider_one():
    """2026-11-12 costcurve-probe: the first one-row sample (107 ms, one-time work) priced a one-token
    window above a seven-token one (46 ms); the allocator drafted 6 deep on prose for the whole reply."""
    from yunshu_engine.round_driver.allocate import CostCurve, allocate

    curve = CostCurve()
    curve.observe(1, 107.0)
    curve.observe(7, 46.0)
    assert curve(1) <= curve(7)
    # prose-like acceptance: the 1st draft lands half the time, deeper ones rarely
    probs = [[0.5, 0.15, 0.05, 0.02, 0.01, 0.005]]
    curve.observe(1, 23.0)  # the honest one-row cost once measured
    assert allocate(1, probs, curve, 64, 1.6, padded=True)[0] <= 2
