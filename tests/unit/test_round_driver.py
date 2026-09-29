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
        def draft(rows, heads, depths):
            out = []
            for row, depth in zip(rows, depths, strict=True):
                ref = oracle[row.req.handle]
                got = list(ref[row.generated : row.generated + depth])
                if wrong and got:
                    # right up to a position that varies by row and step
                    j = (row.generated + row.req.handle) % len(got)
                    got[j] = (got[j] + 1) % 500
                out.append(got)
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
    real_forward = drv.forward

    def spy(model, segs):
        steps.append([s.length for s in segs])
        return real_forward(model, segs)

    monkeypatch.setattr(drv, "forward", spy)

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
