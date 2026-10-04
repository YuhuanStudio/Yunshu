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
