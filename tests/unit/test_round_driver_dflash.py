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


@pytest.fixture(scope="module")
def target():
    from scripts.research.phaseb_verify import tiny

    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs lane-capable Metal")
    return tiny()[0]


def _run(lm, prompts, *, draft, sampled=False, stagger=False):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    driver = RoundDriver(lm, dflash_drafter=_draft(lm) if draft else None, chunk=16)
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
                    prompt, 12, sampling=params, handle=i, logprobs=True, top_logprobs=2
                )
            )
        for event in driver.step():
            output[event.handle].append((event.token, event.logprob))
        steps += 1
        assert steps < 100
    return output, driver


@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("stagger", [False, True])
def test_dflash_rows_match_solo_target_tokens_and_logprobs(
    target, sampled, stagger, rows
):
    prompts = [list(range(4 + b, 25 + 7 * b)) for b in range(rows)]
    ref = [_run(target, [p], draft=False, sampled=sampled)[0][0] for p in prompts]
    got, driver = _run(target, prompts, draft=True, sampled=sampled, stagger=stagger)
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
