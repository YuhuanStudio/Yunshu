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


@pytest.mark.parametrize("draft", [True, False])
def test_long_atoms_keep_hit_equal_to_miss(tiny, draft):
    """With a coarser atom grid beyond ``long_from`` a restored session continues
    on the grid a cold prefill would have used."""
    from yunshu_engine.round_driver import driver as drv

    lm, drafter = tiny
    dr = drafter if draft else None
    short = DOC[:513]
    long = DOC + [(3 * i + 1) % 500 for i in range(200)]

    def run(prompts, apc):
        d = drv.RoundDriver(lm, drafter=dr, stop_tokens=set(), chunk=128, apc=apc)
        d.long_from, d.long_chunk = 256, 256
        outs, hits = [], []
        for i, p in enumerate(prompts):
            hits.append(d.add(drv.Request(p, N, handle=i)))
            out = []
            while d.busy():
                out += [e.token for e in d.step()]
            outs.append(out)
        return outs, hits

    ref = run([long], _manager())[0][0]
    apc = _manager()
    run([short], apc)
    got, hits = run([long], apc)
    assert hits[0] in (256, 512)
    assert got[0] == ref


@pytest.mark.parametrize("draft", [True, False])
def test_exact_repeat_replays_the_final_block(tiny, draft):
    """The second identical prompt writes the stored final block back (no target
    forward for it) and generates the same tokens as the first and as a cold run;
    an MTP-head row re-absorbs the tail into its head from the stored hidden rows."""
    from yunshu_engine.round_driver import replay

    lm, drafter = tiny
    dr = drafter if draft else None
    ref = _run(lm, dr, [DOC], _manager())[0][0]
    replay.clear()
    apc = _manager()
    first, _, _ = _run(lm, dr, [DOC], apc)
    assert replay.stats()["stored"] == 1 and replay.stats()["replayed"] == 0
    again, _, _ = _run(lm, dr, [DOC], apc)
    assert replay.stats()["replayed"] == 1
    assert first[0] == ref and again[0] == ref
    replay.clear()


def test_lone_row_checkpoints_are_deferred_then_flushed(tiny):
    """A lone row's checkpoint stores are captured (not copied) inside its prefill step and
    published by flush_checkpoints; a request added afterwards hits them and matches a cold run;
    a removed row's captures are discarded."""
    from yunshu_engine.apc_manager import YunshuAPCManager
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    def yunshu_manager():
        return YunshuAPCManager(
            num_blocks=64,
            block_size=16,
            overrides={
                "memory_max_gb": 1,
                "checkpoint_entries": 4,
                "checkpoint_interval_tokens": 256,
            },
        )

    lm, _ = tiny
    ref = _run(lm, None, [DOC], yunshu_manager())[0][0]
    apc = yunshu_manager()
    d = RoundDriver(lm, stop_tokens=set(), chunk=128, apc=apc)
    d.add(Request(DOC, N, handle=0))
    flushed = []
    real = d.flush_checkpoints
    d.flush_checkpoints = lambda: (flushed.append(len(d._deferred_apcs)), real())
    out = []
    while d.busy():
        out += [e.token for e in d.step()]
    assert out == ref
    assert any(n for n in flushed)  # something was captured and later flushed
    d.flush_checkpoints = real
    d.flush_checkpoints()
    again = d.add(Request(DOC, N, handle=1))
    assert again >= len(DOC) - 1  # the deferred checkpoint was published

    d2 = RoundDriver(lm, stop_tokens=set(), chunk=128, apc=yunshu_manager())
    d2.add(Request(DOC, N, handle="x"))
    while d2.busy() and not d2._deferred_apcs:
        d2.step()
    assert d2._deferred_apcs
    d2.remove("x")
    assert not d2._deferred_apcs


def test_row_reports_its_planned_checkpoints(tiny):
    """The runner's prefix single-flight asks the driver which checkpoints a row will publish."""
    from yunshu_engine.apc_manager import YunshuAPCManager
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    lm, _ = tiny
    apc = YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        overrides={
            "memory_max_gb": 1,
            "checkpoint_entries": 4,
            "checkpoint_interval_tokens": 256,
        },
    )
    d = RoundDriver(lm, stop_tokens=set(), chunk=128, apc=apc)
    handle = object()
    d.add(Request(DOC, N, handle=handle))
    salt, points = d.planned_checkpoints(handle)
    assert points and all(0 < p < len(DOC) for p in points) and salt == d.rows[0].salt
    assert d.planned_checkpoints(object()) is None


def test_media_prompt_checkpoints_avoid_media_spans_and_hits_match_cold(tiny):
    """With placeholder (media) tokens in the prompt, no planned checkpoint cuts the media span, a
    repeat restores a prefix that leaves a text-only suffix, and its tokens equal the cold run's
    (the prompt travels as embeddings + positions, as a media prompt does)."""
    from yunshu_engine.apc_manager import YunshuAPCManager
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    lm, _ = tiny
    media_id = 499
    prompt = list(DOC)
    prompt[100:120] = [media_id] * 20  # the placeholder span
    embeds = lm.model.embed_tokens(mx.array(prompt, dtype=mx.int32))
    pos = mx.tile(mx.arange(len(prompt), dtype=mx.int32)[None, :], (3, 1))

    def manager():
        return YunshuAPCManager(
            num_blocks=64,
            block_size=16,
            overrides={
                "memory_max_gb": 1,
                "checkpoint_entries": 4,
                "checkpoint_interval_tokens": 256,
            },
        )

    def run(apc, driver=None):
        d = driver or RoundDriver(lm, stop_tokens=set(), chunk=64, apc=apc)
        d._media_ids = {media_id}
        h = object()
        hit = d.add(Request(prompt, N, handle=h, embeds=embeds, position_ids=pos))
        planned = d.planned_checkpoints(h)
        out = []
        while d.busy():
            out += [e.token for e in d.step()]
        return out, hit, planned, d

    cold, hit0, planned, d = run(manager())
    assert hit0 == 0
    assert planned is not None
    assert all(not (100 < c < 120) for c in planned[1])  # never inside the media span
    again, hit1, _, _ = run(d._apc_manager, driver=None)
    assert (
        hit1 == 0 or hit1 >= 120
    )  # a restored prefix covers the whole placeholder span
    assert again == cold
