"""Round driver + APC prefix cache on a tiny random 4-bit Qwen3.5: a row whose
prompt prefix hits a stored checkpoint (target KV + GDN state + MTP head KV)
generates the tokens it would without the cache."""

import pytest

from .test_round_driver import N, _tiny

mx = pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def tiny():
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    return _tiny()


def _manager():
    from mlx_vlm.apc import APCManager

    return APCManager(
        num_blocks=64,
        block_size=16,
        overrides={
            "memory_max_gb": 1,
            "checkpoint_entries": 4,
            "checkpoint_interval_tokens": 256,
        },
    )


def _run(lm, drafter, prompts, apc=None, chunk=128, sampled=False):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(lm, drafter=drafter, stop_tokens=set(), chunk=chunk, apc=apc)
    outs, hits = [], []
    for i, p in enumerate(prompts):
        sampling = None
        if sampled:
            from types import SimpleNamespace

            sampling = SimpleNamespace(
                temperature=0.0001,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
                xtc_special_tokens=None,
                seed=1,
            )
        hits.append(d.add(Request(p, N, handle=i, sampling=sampling)))
        out = []
        while d.busy():
            out += [e.token for e in d.step()]
        outs.append(out)
    return outs, hits, d


DOC = [(7 * i + 11) % 500 for i in range(700)]


@pytest.mark.parametrize("draft", [True, False])
def test_repeated_prompt_hits_and_matches(tiny, draft):
    lm, drafter = tiny
    dr = drafter if draft else None
    # a cold prefill with the cache on: the span plan (chunk grid plus the
    # prompt's checkpoints) is what a warm run continues
    ref = _run(lm, dr, [DOC], _manager())[0][0]
    apc = _manager()
    first, hits1, _ = _run(lm, dr, [DOC], apc)
    assert hits1 == [0] and first[0] == ref
    again, hits2, _ = _run(lm, dr, [DOC], apc)
    assert hits2[0] >= len(DOC) - 1 and again[0] == ref


def test_extension_of_a_stored_prefix_matches(tiny):
    """A longer prompt restores the checkpoint of a shorter one (on the chunk
    grid, so the spans after it are the cold spans) and continues exactly."""
    lm, drafter = tiny
    short = DOC[:513]  # checkpoints at 256, 512
    long = DOC + [(3 * i + 1) % 500 for i in range(200)]
    ref = _run(lm, drafter, [long], _manager())[0][0]
    apc = _manager()
    _run(lm, drafter, [short], apc)
    got, hits, _ = _run(lm, drafter, [long], apc)
    assert hits[0] in (256, 512)
    assert got[0] == ref


def test_partial_prefix_and_other_layouts_do_not_mix(tiny):
    lm, drafter = tiny
    apc = _manager()
    _run(lm, drafter, [DOC], apc)
    other = DOC[:300] + [(5 * i + 2) % 500 for i in range(300)]
    ref = _run(lm, drafter, [other], _manager())[0][0]
    got, hits, _ = _run(lm, drafter, [other], apc)
    assert got[0] == ref
    # a sampled row drafts like a greedy one: same layout, same entries
    _, hits, _ = _run(lm, drafter, [DOC], apc, sampled=True)
    assert hits[0] >= len(DOC) - 1
    # rows without a head keep their own entries
    apc = _manager()
    _run(lm, drafter, [DOC], apc)
    _, hits, _ = _run(lm, None, [DOC], apc)
    assert hits == [0]
    _, hits, _ = _run(lm, None, [DOC], apc)
    assert hits[0] >= len(DOC) - 1


def test_batch_of_hit_and_cold_rows_matches(tiny):
    """A restored row decodes in one batch with a cold row."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    lm, drafter = tiny
    cold = [(13 * i + 5) % 500 for i in range(90)]
    ref = [_run(lm, drafter, [p], _manager())[0][0] for p in (DOC, cold)]
    apc = _manager()
    _run(lm, drafter, [DOC], apc)
    d = RoundDriver(lm, drafter=drafter, stop_tokens=set(), chunk=128, apc=apc)
    hits = [d.add(Request(p, N, handle=i)) for i, p in enumerate((DOC, cold))]
    out = {0: [], 1: []}
    while d.busy():
        for e in d.step():
            out[e.handle].append(e.token)
    assert hits[0] > 0 and hits[1] == 0
    assert [out[0], out[1]] == ref


def test_driver_stores_only_planned_checkpoints_with_an_interval(tiny):
    """The checkpoint interval cuts spans at every multiple; only the last is a
    snapshot (with the final one). The driver used upstream's coordinator, which
    stored every one (22 vs 8.6 GiB at 2 x 32K)."""
    from yunshu_engine.apc_manager import YunshuAPCManager

    lm, drafter = tiny
    apc = YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 128},
    )
    _run(lm, drafter, [DOC], apc)
    stored = sorted(len(e.token_ids) for e in apc._exact_cache.values())
    assert 0 < len(stored) <= 2, stored


def test_row_near_its_end_finishes_prefill_in_one_step(tiny):
    """Checkpoint cuts end a row's span for the step; the atoms after them used
    to take one multi-row step each, delaying the first token."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    lm, drafter = tiny
    prompt = DOC[:530]
    ref = _run(lm, drafter, [prompt], _manager(), chunk=512)[0][0]
    d = RoundDriver(lm, drafter=drafter, stop_tokens=set(), chunk=512, apc=_manager())
    d.add(Request(prompt, N, handle=0))
    first = d.step()
    assert [e.token for e in first][:1] == ref[:1]
    out = [e.token for e in first]
    while d.busy():
        out += [e.token for e in d.step()]
    assert out == ref
