"""YunshuAPCManager: checkpoint policy (supersede, head boundary, entries) and lookup provenance."""

import time

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from yunshu_engine.apc_manager import (  # noqa: E402
    YunshuAPCManager,
    auto_memory_gb,
)

IM_START, USER = 900, 901


def _cache(n: int):
    """A hybrid-style prompt cache: recurrent state + dense KV of n tokens."""
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 4, 8)), mx.ones((1, 2, 3)) * n]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    mx.eval(rec.cache, kv.keys, kv.values)
    return [rec, kv]


def _mgr(**kw):
    return YunshuAPCManager(
        num_blocks=8,
        block_size=16,
        overrides={"memory_max_gb": 1},
        head_marker=(IM_START, USER),
        **kw,
    )


def _prompt(head, body, extra=()):
    """system turn (head tokens) + user turn (body tokens)."""
    return [7, 7, *head, IM_START, USER, *body, *extra]


def test_auto_memory_scales_with_machine_and_model():
    gib = 1 << 30
    assert auto_memory_gb(128 * gib, 16 * gib) == 32.0
    assert auto_memory_gb(64 * gib, 16 * gib) == 16.0
    assert auto_memory_gb(32 * gib, 16 * gib) == 4.0
    assert auto_memory_gb(16 * gib, 16 * gib) == 0.0
    assert auto_memory_gb(512 * gib, 16 * gib) == 32.0


def test_head_boundary_is_the_start_of_the_first_user_turn():
    m = _mgr()
    ids = _prompt(range(100, 150), range(200, 260))
    assert m.head_boundary(ids) == 2 + 50
    assert m.head_boundary([1, 2, 3]) == 0
    assert _mgr().head_boundary([IM_START, USER, 5]) == 0  # no system turn
    m.head_marker = None
    assert m.head_boundary(ids) == 0


def test_user_first_multiturn_chat_does_not_pin_a_fake_system_head():
    ids = [
        IM_START,
        USER,
        *range(100, 140),
        IM_START,
        999,
        *range(200, 240),
        IM_START,
        USER,
        5,
    ]
    assert _mgr().head_boundary(ids) == 0


def _coordinator(m):
    from types import SimpleNamespace

    from yunshu_engine.apc_manager import _Coordinator

    c = object.__new__(_Coordinator)
    c.manager = m
    c.model = None
    c.plan = SimpleNamespace(
        restorable=True, strategy="checkpoint", legacy_mode="exact"
    )
    return c


def test_checkpoint_lengths_are_final_one_interval_and_the_head():
    m = _mgr()
    c = _coordinator(m)
    head = list(range(100, 3000))  # a long system turn
    ids = _prompt(head, range(5000, 12000))
    lengths = c.checkpoint_lengths(ids, set())
    final = len(ids) - 1
    interval = ((final - 1) // 2048) * 2048
    assert lengths == sorted({final, interval, m.head_boundary(ids)})
    assert m._generation == 1
    # a short prompt: just the final one, no interval/head below the minimum
    short = _prompt(range(100, 110), range(200, 220))
    assert c.checkpoint_lengths(short, set()) == [len(short) - 1]
    m.keep_interval_checkpoint = False
    assert c.checkpoint_lengths(ids, set()) == sorted({final, m.head_boundary(ids)})


def test_head_checkpoint_never_cuts_media_tokens():
    m = _mgr()
    c = _coordinator(m)
    ids = _prompt(range(100, 400), range(500, 900))
    h = m.head_boundary(ids)
    assert h in c.checkpoint_lengths(ids, set())
    # an image after the head: restoring only the head would leave media in the suffix
    ids2 = list(ids)
    ids2[h + 5] = 4242
    assert h not in c.checkpoint_lengths(ids2, {4242})


def test_entries_are_not_capped_at_two():
    m = _mgr()
    assert m._exact_cache_max >= 8


def test_newer_checkpoint_supersedes_earlier_request_but_not_the_same_request():
    m = _mgr()
    a = list(range(1000, 1100))
    m.begin_request()
    m.store_exact_cache(a[:48], _cache(48))  # interval checkpoint of request 1
    m.store_exact_cache(a[:96], _cache(96))  # its final: both belong to request 1
    assert len(m._exact_cache) == 2
    m.begin_request()  # request 2 extends request 1
    m.store_exact_cache(a[:100] + [1, 2, 3], _cache(103))
    assert [len(e.token_ids) for e in m._exact_cache.values()] == [103]


def test_head_checkpoint_survives_supersede_and_other_sessions_are_untouched():
    m = _mgr()
    head = list(range(100, 150))
    s1 = _prompt(head, range(200, 232))
    s2 = _prompt(list(range(300, 350)), range(400, 432))
    for ids in (s1, s2):
        m.begin_request()
        h = m.head_boundary(ids)
        m.note_head(ids[:h])
        m.store_exact_cache(ids[:h], _cache(h))
        m.store_exact_cache(ids[:-1], _cache(len(ids) - 1))
    assert len(m._exact_cache) == 4
    # session 1 grows: its earlier final goes, both heads and session 2 stay
    m.begin_request()
    s1b = s1 + [1, 2, 3, 4]
    m.store_exact_cache(s1b[:-1], _cache(len(s1b) - 1))
    lengths = sorted(len(e.token_ids) for e in m._exact_cache.values())
    h1, h2 = m.head_boundary(s1), m.head_boundary(s2)
    assert lengths == sorted([h1, h2, len(s2) - 1, len(s1b) - 1])
    # a new session with the same head reuses the head checkpoint
    new = _prompt(head, range(500, 520))
    cache, n = m.lookup_exact_cache(new)
    assert n == h1 and cache is not None


def test_lookup_provenance_ram_and_none():
    m = _mgr()
    ids = list(range(1000, 1100))
    m.begin_request()
    m.store_exact_cache(ids[:96], _cache(96))
    _, n = m.lookup_exact_cache(ids + [5, 6])
    assert n == 96
    rec = m.take_lookup(len(ids) + 2, 96)
    assert rec.tier == "ram" and rec.cached == 96 and rec.ms >= 0
    _, n = m.lookup_exact_cache([9] * 50)
    assert n == 0
    assert m.take_lookup(50, 0).tier == "none"


def test_snapshot_reports_occupancy():
    m = _mgr()
    m.begin_request()
    m.store_exact_cache(list(range(1000, 1096)), _cache(96))
    snap = m.snapshot()
    assert snap["entries"] == 1 and snap["entry_tokens"] == [96]
    assert snap["memory_max_bytes"] == 1 << 30
    assert snap["resident_bytes"] > 0


def test_runner_records_tier_on_the_job_and_x_yunshu_reports_it():
    from types import SimpleNamespace

    from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner
    from yunshu_gateway import x_yunshu

    m = _mgr()
    ids = list(range(1000, 1100))
    m.begin_request()
    m.store_exact_cache(ids[:96], _cache(96))
    prompt = ids + [5, 6]
    _, n = m.lookup_exact_cache(prompt)
    runner = VLMBatchRunner(object(), object(), apc_manager=m)
    stats = RunStats(cached_tokens=n, used_apc=True)
    job = SimpleNamespace(ids=prompt, stats=stats)
    runner._note_cache(job)
    assert stats.cache_tier == "ram" and stats.cache_reload_ms is not None

    info = x_yunshu.RequestInfo("r", "POST", "/v1/chat/completions")
    info.gen = SimpleNamespace(stats=stats)
    out = x_yunshu.build_stats(
        info, {"prompt_tokens": len(prompt), "completion_tokens": 1}
    )
    assert out["cache"]["tier"] == "ram" and out["cache"]["cached_tokens"] == 96
    assert out["cache"]["reload_ms"] == stats.cache_reload_ms
    headers = dict(x_yunshu.stats_headers(out, info))
    assert headers[b"x-yunshu-cache-tier"] == b"ram"

    # a lookup that found nothing reports "none"; lookups from before admission are ignored
    m.lookup_exact_cache([9] * 40)
    cold = SimpleNamespace(ids=[9] * 40, stats=RunStats(used_apc=True))
    runner._note_cache(cold)
    assert cold.stats.cache_tier == "none"
    late = SimpleNamespace(
        ids=[9] * 40, stats=RunStats(used_apc=True, t_admit=time.perf_counter() + 1)
    )
    runner._note_cache(late)
    assert late.stats.cache_tier is None
    info2 = x_yunshu.RequestInfo("r2", "POST", "/v1/chat/completions")
    out2 = x_yunshu.build_stats(
        info2, {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}}
    )
    assert out2["cache"] == {"tier": "ram", "cached_tokens": 4, "reload_ms": None}


def _disk_mgr(tmp_path, gb=1, **kw):
    from yunshu_engine.apc_manager import SpillDiskStore

    disk = SpillDiskStore(tmp_path, namespace="t", num_workers=1, max_bytes=1 << 30)
    return YunshuAPCManager(
        num_blocks=8,
        block_size=16,
        disk=disk,
        overrides={"memory_max_gb": gb},
        head_marker=(IM_START, USER),
        **kw,
    )


def _files(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("*.safetensors"))


def test_ssd_receives_a_checkpoint_when_ram_evicts_it_not_when_stored(tmp_path):
    m = _disk_mgr(tmp_path, max_entries=2)
    a, b, c = (list(range(1000 * i, 1000 * i + 96)) for i in (1, 2, 3))
    for ids in (a, b):
        m.begin_request()
        m.store_exact_cache(ids, _cache(96))
    m.disk.flush()
    assert _files(tmp_path) == []  # both retained in RAM: nothing written
    m.begin_request()
    m.store_exact_cache(c, _cache(96))  # third entry: the LRU one (a) goes to the SSD
    m.disk.flush()
    assert len(_files(tmp_path)) == 1
    cache, n = m.lookup_exact_cache(a + [1, 2])
    assert n == 96 and cache is not None
    assert m.take_lookup(98, 96).tier == "ssd"


def test_superseded_checkpoints_are_never_written(tmp_path):
    m = _disk_mgr(tmp_path)
    ids = list(range(1000, 1200))
    for n in (96, 112, 128, 144):
        m.begin_request()
        m.store_exact_cache(ids[:n], _cache(n))
    assert len(m._exact_cache) == 1
    m.disk.flush()
    assert _files(tmp_path) == []


def test_close_persists_resident_checkpoints_for_a_restart(tmp_path):
    m = _disk_mgr(tmp_path)
    ids = list(range(2000, 2096))
    m.begin_request()
    m.store_exact_cache(ids, _cache(96))
    m.close()
    assert len(_files(tmp_path)) == 1
    fresh = _disk_mgr(tmp_path)  # a restarted server, empty RAM
    cache, n = fresh.lookup_exact_cache(ids + [9, 9])
    assert n == 96 and cache is not None
    assert fresh.take_lookup(98, 96).tier == "ssd"
    fresh.close()


def test_entry_too_large_for_ram_is_written_synchronously(tmp_path):
    m = _disk_mgr(tmp_path, gb=0.000001)  # ~1 KiB: nothing fits
    ids = list(range(3000, 3096))
    m.begin_request()
    m.store_exact_cache(ids, _cache(96))
    assert len(m._exact_cache) == 0
    assert len(_files(tmp_path)) == 1
    _, n = m.lookup_exact_cache(ids + [1])
    assert n == 96


def test_explicit_breakpoint_is_stored_and_survives_growing_turn():
    m = _mgr()
    c = _coordinator(m)
    ids = list(range(6000, 6300))
    plan = {"points": [(113, 300)], "written": 0}
    c.set_request(ids, plan)
    assert c.checkpoint_lengths(ids, set()) == [113]
    m.begin_request()
    m.protect_boundary(ids[:113], 0, 300)
    assert m.store_exact_cache(ids[:113], _cache(113))
    m.begin_request()
    assert m.store_exact_cache(ids[:280], _cache(280))
    assert 113 in [len(e.token_ids) for e in m._exact_cache.values()]


def test_explicit_ttl_refreshes_only_on_a_read(monkeypatch):
    m = _mgr()
    clock = [100.0]
    monkeypatch.setattr("yunshu_engine.apc_manager.time.monotonic", lambda: clock[0])
    ids = list(range(2000, 2200))
    m.protect_boundary(ids[:113], 0, 300)
    m.store_exact_cache(ids[:113], _cache(113))
    clock[0] = 350.0
    assert m.lookup_exact_cache(ids)[1] == 113
    clock[0] = 600.0
    assert m.lookup_exact_cache(ids)[1] == 113
    clock[0] = 901.0
    assert m.lookup_exact_cache(ids)[1] == 0


def test_failed_explicit_store_reports_no_creation_or_retention(monkeypatch):
    from mlx_vlm.apc_coordinator import APCCoordinator

    m = _mgr()
    c = _coordinator(m)
    ids = list(range(6000, 6300))
    plan = {"points": [(113, 300)], "written": 0}
    c.set_request(ids, plan)
    c.checkpoint_lengths(ids, set())
    monkeypatch.setattr(APCCoordinator, "store_checkpoint", lambda *a, **k: False)
    assert not c.store_checkpoint(ids[:113], [])
    assert plan["written"] == 0
    assert not m._retention


def test_successful_explicit_store_reports_full_rendered_prefix(monkeypatch):
    from mlx_vlm.apc_coordinator import APCCoordinator

    m = _mgr()
    c = _coordinator(m)
    ids = list(range(6000, 6300))
    plan = {"points": [(113, 300)], "written": 0}
    c.set_request(ids, plan)
    c.checkpoint_lengths(ids, set())
    monkeypatch.setattr(APCCoordinator, "store_checkpoint", lambda *a, **k: True)
    assert c.store_checkpoint(ids[:113], [])
    assert plan["written"] == 113
    assert m._retention


def test_expired_async_spill_cannot_resurrect_a_breakpoint(monkeypatch):
    from mlx_vlm.apc import APCManager

    m = _mgr()
    ids = list(range(2000, 2200))
    now = [100.0]
    monkeypatch.setattr("yunshu_engine.apc_manager.time.monotonic", lambda: now[0])
    m.protect_boundary(ids[:113], 0, 300)
    # A queued disk publication lands after expiry and returns its old state.
    monkeypatch.setattr(
        APCManager,
        "lookup_exact_cache",
        lambda self, tokens, *args, **kw: (
            (["old"], 113) if kw.get("max_prefix_tokens", 200) >= 113 else (None, 0)
        ),
    )
    now[0] = 401.0
    assert m.lookup_exact_cache(ids)[1] == 0


def test_rejected_span_plan_does_not_refresh_retention(monkeypatch):
    from mlx_vlm.apc import _sequence_hash

    m = _mgr()
    c = _coordinator(m)
    ids = list(range(2000, 2200))
    now = [100.0]
    monkeypatch.setattr("yunshu_engine.apc_manager.time.monotonic", lambda: now[0])
    c.set_request(ids, {"points": [(113, 300)], "written": 0})
    m.protect_boundary(ids[:113], 0, 300)
    m.store_exact_cache(ids[:113], _cache(113))
    key = _sequence_hash(tuple(ids[:113]), 0, m.block_size)
    m._span_plans[key] = (17,)  # the source used a different earlier split
    now[0] = 350.0
    assert (
        c.lookup(
            ids, extra_hash=0, safe_lookup_min=0, suffix_is_text_only=lambda n: True
        )
        is None
    )
    assert m._retention[key][0] == 400.0


def test_canonical_seams_without_a_write_do_not_clone(monkeypatch):
    from mlx_vlm.apc_coordinator import APCCoordinator

    m = _mgr()
    c = _coordinator(m)
    ids = list(range(6000, 6300))
    plan = {"points": [(80, 300), (113, 300)], "writes": [(113, 300)], "written": 0}
    c.set_request(ids, plan)
    assert c.checkpoint_lengths(ids, set()) == [80, 113]
    calls = []
    monkeypatch.setattr(
        APCCoordinator, "store_checkpoint", lambda *a, **k: calls.append(a) or True
    )
    assert not c.store_checkpoint(ids[:80], [])
    assert not calls
    assert c.store_checkpoint(ids[:113], [])
    assert len(calls) == 1
    assert plan["written"] == 113


def test_identical_prompt_policies_remain_fifo_until_prefill_finishes():
    c = _coordinator(_mgr())
    ids = list(range(2000, 2200))
    first = {"points": [(80, 300)], "written": 0}
    second = {"points": [(113, 3600)], "written": 0}
    c.set_request(ids, first)
    c.set_request(ids, second)
    assert c.request(ids) is first
    c.release_request(ids, first)
    assert c.request(ids) is second
    c.release_request(ids, second)
    assert c.request(ids) is None


def _turn(m, ids, stride=2048):
    """One request of a growing conversation: its interval checkpoint and its final."""
    m.begin_request()
    final = len(ids) - 1
    interval = (final - 1) // stride * stride
    m.store_exact_cache(ids[:interval], _cache(interval))
    m.store_exact_cache(ids[:final], _cache(final))


def _conversation(turns=12, step=3500, start=2000):
    ids = [3] * 40
    ids += [IM_START, USER]
    out = []
    n = start
    for t in range(turns):
        ids = ids + [10000 + t] * (n - len(ids))
        out.append(list(ids) + [5])
        n += step
    return out


def test_branch_inside_a_grown_conversation_hits_a_retained_turn_boundary():
    m = _mgr()
    prompts = _conversation()
    for p in prompts:
        _turn(m, p)
    full = prompts[-1]
    mid = len(full) // 2
    branch = full[:mid] + [777] * 500
    _, n = m.lookup_exact_cache(branch)
    # a radix cache would hit ~mid; superseding everything leaves only the head
    assert n >= mid - 12000, (n, mid)
    assert n % 16 == 0 or n


def test_retained_anchors_are_thinned_and_bounded():
    m = _mgr()
    prompts = _conversation(turns=40, step=1000)
    for p in prompts:
        _turn(m, p)
    lengths = sorted(len(e.token_ids) for e in m._exact_cache.values())
    # geometric spacing: a handful of anchors, not one per turn
    assert len(lengths) <= 14, lengths
    # linear follow-up still finds the newest checkpoint
    _, n = m.lookup_exact_cache(prompts[-1] + [9, 9])
    assert n == len(prompts[-1]) - 1


def test_anchor_rows_become_views_of_the_newest_checkpoint():
    m = _mgr()
    prompts = _conversation(turns=6, step=5000)
    rebound = 0
    for i, p in enumerate(prompts):
        if i:
            m.lookup_exact_cache(p)  # the request restores its predecessor's final
        _turn(m, p)
        rebound += m.share_anchor_rows(p[: len(p) - 1], 0)
    assert rebound > 0
    donor = max(m._exact_cache.values(), key=lambda e: len(e.token_ids))
    anchors = [
        e
        for e in m._exact_cache.values()
        if e is not donor and len(e.token_ids) in set(m._anchors.values())
    ]
    assert anchors
    shared = sum(
        1 for e in anchors if e.prompt_cache[1].keys.shape[-2] == len(e.token_ids)
    )
    assert shared == len(anchors)
    # same bits as before: every row of an anchor equals the donor's row
    for e in anchors:
        n = len(e.token_ids)
        assert bool(
            mx.all(e.prompt_cache[1].keys == donor.prompt_cache[1].keys[..., :n, :])
        )
    # a request that did not restore anything shares nothing
    m._last_hit = None
    assert m.share_anchor_rows(prompts[-1][:-1], 0) == 0


def _shared_conversation(m, turns=6, step=5000):
    prompts = _conversation(turns=turns, step=step)
    for i, p in enumerate(prompts):
        if i:
            m.lookup_exact_cache(p)
        _turn(m, p)
        m.share_anchor_rows(p[: len(p) - 1], 0)
    return prompts


def _logical(m):
    from mlx_vlm.apc import _cache_nbytes

    return sum(_cache_nbytes(e.prompt_cache) for e in m._exact_cache.values())


def test_resident_bytes_count_a_shared_kv_buffer_once():
    m = _mgr()
    _shared_conversation(m)
    assert m._kv_share
    saved = sum(v for _, v in m._kv_share.values())
    assert saved > 0
    assert m.resident_bytes() == _logical(m) - saved


def test_resident_bytes_count_a_buffer_pinned_by_views_after_its_owner_left():
    from mlx_vlm.apc import _cache_nbytes

    m = _mgr()
    _shared_conversation(m)
    root = next(iter({r for r, _ in m._kv_share.values()}))
    owner_kv = m._roots[root]
    owner_total = _cache_nbytes(m._exact_cache[root].prompt_cache)
    before = m.resident_bytes()
    m._exact_cache.pop(root)  # the owner is evicted; the anchors still pin its buffer
    assert m.resident_bytes() == before - owner_total + owner_kv


def test_anchor_bytes_are_capped_by_a_budget():
    m = _mgr()
    prompts = _conversation(turns=12, step=3500)
    for p in prompts:  # no restore, so nothing is shared: anchors own their rows
        _turn(m, p)
        m.share_anchor_rows(p[: len(p) - 1], 0)
    assert m._anchors
    m.memory_max_bytes = 20 << 20
    m.enforce_anchor_budget()
    assert m.anchor_bytes() <= m.anchor_budget_bytes() == int(0.15 * (20 << 20))
    newest = max(len(e.token_ids) for e in m._exact_cache.values())
    assert newest == len(prompts[-1]) - 1  # the newest checkpoint is never an anchor


def test_memory_pressure_evicts_anchors_before_a_big_allocation():
    m = _mgr()
    prompts = _conversation(turns=12, step=3500)
    for p in prompts:
        _turn(m, p)
    anchors = len(m._anchors)
    assert anchors
    m._memory_headroom = lambda: 0  # no free memory at all
    m._make_room(1 << 30)
    assert not m._anchors
    assert len(m._exact_cache) < anchors + 3


def test_request_path_never_clears_the_allocator_pool_but_pressure_does(monkeypatch):
    """Clearing the pool walks every pooled buffer (4-13 ms): it is the follow-up's TTFT."""
    import yunshu_engine.apc_manager as am

    calls = []
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append(1))
    monkeypatch.setattr(am, "RELEASE_FREED_BYTES", 1)
    m = _mgr()
    _shared_conversation(m, turns=4, step=5000)
    assert not calls
    m._memory_headroom = lambda: 0
    m._make_room(1 << 30)
    assert calls


def test_anchors_are_re_pointed_before_the_new_copy_is_evaluated():
    m = _mgr()
    prompts = _conversation(turns=5, step=5000)
    for i, p in enumerate(prompts[:-1]):
        if i:
            m.lookup_exact_cache(p)
        _turn(m, p)
        m.share_anchor_rows(p[: len(p) - 1], 0)
    last = prompts[-1]
    m.lookup_exact_cache(last)  # the request restores its predecessor
    final = last[: len(last) - 1]
    m.begin_request()
    m._supersede(final, 0, m._generation)  # what release_superseded does first
    snapshot = _cache(len(final))  # the copy, not stored yet
    held = {k: e.prompt_cache[1].keys for k, e in m._exact_cache.items()}
    views = m.share_anchor_rows_lazy(final, snapshot, 0)
    assert views, "anchors at or below the restore point must be re-pointed"
    changed = [
        k for k, e in m._exact_cache.items() if e.prompt_cache[1].keys is not held[k]
    ]
    assert changed
    assert all(k in m._kv_share for k in changed)
    mx.eval(views)
    m.finish_anchor_sharing()
    for k in changed:
        e = m._exact_cache[k]
        n = len(e.token_ids)
        assert bool(mx.all(e.prompt_cache[1].keys == snapshot[1].keys[..., :n, :]))


def test_make_room_reads_free_memory_once(monkeypatch):
    from mlx_vlm.apc import APCManager

    m = _mgr()
    for p in _conversation(turns=8, step=3500):
        _turn(m, p)
    assert m._anchors
    reads = []

    def headroom(self):
        reads.append(1)
        return 1 << 40

    monkeypatch.setattr(APCManager, "_memory_headroom", headroom)
    m._make_room(1 << 20)
    assert len(reads) == 1


def test_anchor_budget_on_a_big_machine_is_one_gib():
    m = _mgr()
    m.memory_max_bytes = 32 << 30  # the 128 GB machine's APC budget
    # each anchor owns a ~0.2 GiB recurrent state that cannot be shared: ~5 anchors at most
    assert m.anchor_budget_bytes() == 1 << 30


def test_immediate_checkpoint_store_re_points_anchors_at_once(monkeypatch):
    """A store that cannot be deferred (APC budget) has no flush after it: the anchors must
    not keep, and pin, their own K/V buffers until some later flush."""
    from mlx_vlm.apc_coordinator import APCCoordinator

    m = _mgr()
    prompts = _conversation(turns=6, step=5000)
    for i, p in enumerate(prompts[:-1]):
        if i:
            m.lookup_exact_cache(p)
        _turn(m, p)
        m.share_anchor_rows(p[: len(p) - 1], 0)
    last = prompts[-1]
    m.lookup_exact_cache(last)
    final = last[: len(last) - 1]

    def upstream_store(self, token_ids, prompt_cache, **kw):
        return self.manager.store_exact_cache(tuple(token_ids), prompt_cache)

    monkeypatch.setattr(APCCoordinator, "store_checkpoint", upstream_store)
    c = _coordinator(m)
    c.defer_checkpoint_stores = False
    m.begin_request()
    assert c.store_checkpoint(final, _cache(len(final)))
    anchors = [e for k, e in m._exact_cache.items() if k in m._anchors]
    assert anchors
    # every anchor at or below the restore point already views the new checkpoint's buffer
    assert all(k in m._kv_share for k in m._anchors if k in m._exact_cache)
    state_only = sum(sum(x.nbytes for x in e.prompt_cache[0].cache) for e in anchors)
    assert m.anchor_bytes() <= state_only + 4096


def test_store_re_points_anchors_before_it_copies_the_cache(monkeypatch):
    """The copy a store makes must not coexist with the anchors' own K/V rows (the peak)."""
    m = _mgr()
    prompts = _conversation(turns=6, step=5000)
    for i, p in enumerate(prompts[:-1]):
        if i:
            m.lookup_exact_cache(p)
        _turn(m, p)
        m.share_anchor_rows(p[: len(p) - 1], 0)
    last = prompts[-1]
    m.lookup_exact_cache(last)
    final = last[: len(last) - 1]
    from mlx_vlm.apc import APCManager

    seen = {}
    original = APCManager.store_exact_cache

    def spy(self, token_ids, prompt_cache, **kw):
        # the copy happens inside this call: by now no anchor may own K/V rows of its own
        seen["own_kv"] = [
            e.prompt_cache[1].keys is not None and k not in self._kv_share
            for k, e in self._exact_cache.items()
            if k in self._anchors
        ]
        return original(self, token_ids, prompt_cache, **kw)

    monkeypatch.setattr(APCManager, "store_exact_cache", spy)
    m.begin_request()
    assert m.store_exact_cache(final, _cache(len(final)))
    assert seen["own_kv"] and not any(seen["own_kv"])
    # and afterwards they view the stored copy, not the live cache that was passed in
    stored = m._exact_cache[
        next(k for k, e in m._exact_cache.items() if len(e.token_ids) == len(final))
    ]
    for k in m._anchors:
        if k in m._exact_cache:
            n = len(m._exact_cache[k].token_ids)
            assert bool(
                mx.all(
                    m._exact_cache[k].prompt_cache[1].keys
                    == stored.prompt_cache[1].keys[..., :n, :]
                )
            )


def test_enforcing_the_anchor_budget_does_not_re_walk_unchanged_anchors(monkeypatch):
    import mlx_vlm.apc as upstream

    m = _mgr()
    for p in _conversation(turns=8, step=3500):
        _turn(m, p)
    assert m._anchors
    m.anchor_bytes()  # warm
    calls = []
    real = upstream._cache_nbytes
    monkeypatch.setattr(
        upstream, "_cache_nbytes", lambda c, *a: calls.append(1) or real(c, *a)
    )
    for _ in range(5):
        m.enforce_anchor_budget()
    assert not calls


def test_small_releases_do_not_clear_the_allocator_pool(monkeypatch):
    import yunshu_engine.apc_manager as am

    calls = []
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append(1))
    am.release_freed_buffers(900 << 20)
    assert not calls
    am.release_freed_buffers(2 << 30)
    assert calls


def test_anchors_survive_the_pre_copy_re_point_of_a_store(monkeypatch):
    """The incoming cache is pinned by the anchors only until its copy is stored: charging
    those bytes to the anchors' budget at that moment dropped every anchor (>150K contexts)."""
    m = _mgr()
    prompts = _conversation(turns=6, step=5000)
    for i, p in enumerate(prompts[:-1]):
        if i:
            m.lookup_exact_cache(p)
        _turn(m, p)
        m.share_anchor_rows(p[: len(p) - 1], 0)
    last = prompts[-1]
    m.lookup_exact_cache(last)
    final = last[: len(last) - 1]
    before = len(m._anchors)
    assert before >= 3
    m.memory_max_bytes = (
        4 << 20
    )  # anchor budget ~600 KB, smaller than the incoming cache
    m.begin_request()
    assert m.store_exact_cache(final, _cache(len(final)))
    assert (
        len(m._anchors) >= before - 1
    )  # thinning may retire one, the budget may not wipe all


def test_budget_sheds_the_anchor_that_owns_its_rows_not_the_whole_chain():
    m = _mgr()
    _shared_conversation(m, turns=6, step=5000)
    anchors = [k for k in m._anchors if k in m._exact_cache and k in m._kv_share]
    assert len(anchors) >= 3
    owner = max(anchors, key=lambda k: len(m._exact_cache[k].token_ids))
    del m._kv_share[owner]  # this one owns its rows (as after an un-re-pointed store)
    m.memory_max_bytes = int(m.anchor_bytes() / 0.15 * 0.5)
    m.enforce_anchor_budget()
    assert owner not in m._exact_cache
    assert all(k in m._exact_cache for k in anchors if k != owner)


def test_dense_media_lane_uses_exact_checkpoint_and_rejects_foreign_pixels():
    from types import SimpleNamespace

    m = _mgr()
    c = m.coordinator(SimpleNamespace(make_cache=lambda: [KVCache()]))
    assert c.strategy == "block"  # text-only dense families keep their block pool
    c.media_checkpoint = True
    assert c.is_checkpoint and c.legacy_mode == "exact"
    ids = [1] * 20 + [4242] * 8 + [2] * 80
    points = c.checkpoint_lengths(ids, {4242})
    assert points == [107]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 107, 4)), mx.ones((1, 1, 107, 4)))
    assert c.store_checkpoint(ids[:107], [kv], extra_hash=123)
    args = dict(
        safe_lookup_min=28,
        suffix_is_text_only=lambda n: n >= 28,
        prefix_has_media=lambda n: n > 20,
    )
    hit = c.lookup(ids, extra_hash=123, **args)
    assert hit is not None and hit["prefix_len"] == 107
    assert c.lookup(ids, extra_hash=456, **args) is None
    # A multi-turn continuation reuses the same image-containing checkpoint.
    hit = c.lookup(ids + [3] * 12, extra_hash=123, **args)
    assert hit is not None and hit["prefix_len"] == 107
