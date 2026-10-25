"""Real small-array DFlash transactions: per-row identity and fair progress."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


def _draft(lm):
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    from yunshu_engine.dflash_tree import quantize_drafter

    cfg = DFlash2Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=512,
        mask_token_id=7,
        max_position_embeddings=4096,
        block_size=8,
        num_target_layers=4,
        target_layer_ids=[0, 3],
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=8,
        selector_top_k=4,
        layer_types=["sliding_attention"] * 2,
        sliding_window=32,
    )
    d = DFlash2DraftModel(cfg)
    d.set_dtype(mx.bfloat16)
    quantize_drafter(d, group_size=64, bits=8)
    mx.eval(d.parameters())
    return d


@pytest.fixture(scope="module", autouse=True)
def restore_probe_lane_mode():
    # The probe enables a process-global kernel route for the tiny model.
    # A later module must see the mode it had before this module ran.
    from yunshu_engine.kernels import ragged_kv

    previous = ragged_kv._STATE["dense_lane"]
    yield
    ragged_kv.set_dense_lane(previous)


@pytest.fixture(scope="module")
def target():
    from scripts.research.phaseb_verify import tiny

    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs lane-capable Metal")
    return tiny()[0]


def _run(
    lm,
    prompts,
    *,
    draft,
    sampled=False,
    stagger=False,
    logprobs=True,
    backend=None,
    apc=None,
    tokens=12,
    guide_factory=None,
):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    driver = RoundDriver(
        lm,
        dflash_drafter=_draft(lm) if (draft if backend is None else backend) else None,
        chunk=16,
        apc=apc,
    )
    output = [[] for _ in prompts]
    todo = list(enumerate(prompts))
    steps = 0
    while driver.busy() or todo:
        if todo and (not stagger or steps % 3 == 0):
            i, prompt = todo.pop(0)
            params = (
                SimpleNamespace(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    xtc_probability=0.0,
                    seed=17 + prompt[0],
                )
                if sampled
                else None
            )
            driver.add(
                Request(
                    prompt,
                    tokens,
                    sampling=params,
                    handle=i,
                    logprobs=logprobs,
                    top_logprobs=2,
                    draft=draft,
                    guide=guide_factory() if guide_factory else None,
                )
            )
        for event in driver.step():
            output[event.handle].append((event.token, event.logprob))
        steps += 1
        assert steps < 100
    return output, driver


@pytest.mark.parametrize("logprobs", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("stagger", [False, True])
def test_dflash_rows_match_solo_target_tokens_and_logprobs(
    target, sampled, stagger, rows, logprobs
):
    prompts = [list(range(4 + b, 25 + 7 * b)) for b in range(rows)]
    ref = [
        _run(
            target,
            [p],
            draft=False,
            sampled=sampled,
            logprobs=logprobs,
            backend=not logprobs,
        )[0][0]
        for p in prompts
    ]
    got, driver = _run(
        target, prompts, draft=True, sampled=sampled, stagger=stagger, logprobs=logprobs
    )
    assert got == ref
    assert driver.drafted > 0
    assert (
        not driver.batch.rows and not driver.dflash.rings and not driver.dflash.hidden
    )


def test_cancel_drops_prefill_capture_and_live_ring(target):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
    a, b = object(), object()
    driver.add(Request(list(range(4, 60)), 20, handle=a))
    driver.add(Request(list(range(8, 70)), 20, handle=b))
    driver.idle_budget = 16
    driver.step()
    row = next(r for r in driver.rows if r.req.handle is a)
    assert id(row) in driver.dflash.hidden
    driver.remove(a)
    assert id(row) not in driver.dflash.hidden
    while driver.rows[0].pending is None:
        driver.step()
    live = driver.rows[0]
    assert id(live) in driver.dflash.rings
    driver.remove(b)
    assert not driver.busy()
    assert not driver.dflash.hidden and not driver.dflash.rings


def test_more_than_node_budget_still_advances_every_row(monkeypatch):
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.round_driver.dflash import DFlashBatch

    batch = object.__new__(DFlashBatch)
    batch.rows = list(range(35))
    packs = []
    monkeypatch.setattr(
        batch, "_step_rows", lambda driver, rows: packs.append(rows) or rows
    )
    before = lane_linear.sum_reuse_enabled()
    assert batch.step(None) == list(range(35))
    assert list(map(len, packs)) == [32, 3]
    assert lane_linear.sum_reuse_enabled() == before


def test_dflash_apc_hit_matches_cold_and_uses_target_layout_salt(target):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    from .test_round_driver_apc import _manager

    apc = _manager()
    prompt = [4 + (7 * i) % 500 for i in range(700)]
    draft = _draft(target)

    def generate():
        driver = RoundDriver(target, dflash_drafter=draft, chunk=128, apc=apc)
        hit = driver.add(Request(prompt, 12, logprobs=True, handle=0))
        assert driver.apc_salt(0, True) == driver.apc_salt(0, False)
        out = []
        while driver.busy():
            out.extend((e.token, e.logprob) for e in driver.step())
        return hit, out

    hit0, cold = generate()
    hit1, warm = generate()
    assert hit0 == 0 and hit1 > 0
    assert warm == cold


def test_oracle_copy_windows_commit_more_than_one_token(target, monkeypatch):
    from yunshu_engine.copy_drafter import CopyDrafter

    prompt = list(range(4, 25))
    ref, _ = _run(target, [prompt], draft=False, tokens=48)
    tokens = [t for t, _ in ref[0]]

    def oracle(copy, cap):
        generated = len(copy.ctx) - len(prompt)
        copy.last_match = 100
        return tokens[generated : generated + cap]

    monkeypatch.setattr(CopyDrafter, "draft", oracle)
    got, driver = _run(target, [prompt], draft=True, tokens=48)
    assert got == ref
    assert driver.accepted > 30
    assert driver.steps < 10


def test_guided_rows_stop_tree_at_entry_and_mask_each_body_token(target):
    import numpy as np

    class Guide:
        def __init__(self):
            self.position = 0

        @property
        def constrained(self):
            return 2 <= self.position < 5

        def arms(self, token):
            return self.position == 1

        def feed(self, token):
            self.position += 1

        def mask(self):
            if not self.constrained:
                return None
            words = np.zeros(16, dtype=np.uint32)
            token = 70 + self.position
            words[token // 32] = 1 << (token % 32)
            return words.view(np.int32)

    prompts = [list(range(4, 25)), list(range(10, 47))]
    ref, _ = _run(target, prompts, draft=False, logprobs=False, guide_factory=Guide)
    got, _ = _run(target, prompts, draft=True, logprobs=False, guide_factory=Guide)
    assert got == ref
    assert all([t for t, _ in row][2:5] == [72, 73, 74] for row in got)


def test_text_driver_clears_another_groups_mrope_before_prefill(target):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    target._rope_deltas = mx.array([[17]], dtype=mx.float32)
    driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
    driver.add(Request(list(range(4, 25)), 4))
    driver.step()
    assert target._rope_deltas is None
    driver.remove(None)
