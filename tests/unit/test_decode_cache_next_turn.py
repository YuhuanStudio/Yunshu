"""YUNSHU_NEXT_TURN_PREFILL=decode (default, user 2026-11-02): the cache a reply was decoded into is
stored under prompt + the tokens it holds, so the follow-up prefills only its new message (turn-2 TTFT
0.6 s -> TensorFold / mlx-serve's ~0.1 s, which reuse the reply KV)."""

from types import SimpleNamespace

from yunshu_engine import settings
from yunshu_engine.vlm_batch_runner import VLMBatchRunner, _Job


class _KV:
    def __init__(self, offset):
        self.keys, self.offset = object(), offset


class _Coord:
    def __init__(self):
        self.calls = []

    def store_decode_tail(self, tokens, caches, *, extra_hash=0):
        self.calls.append((list(tokens), extra_hash))
        return True


def _group(offset):
    coord = _Coord()
    gen = SimpleNamespace(
        apc=coord,
        _generation_batch=SimpleNamespace(prompt_cache=[_KV(offset), object()]),
    )
    return SimpleNamespace(gen=gen, last_caches=None, spec=True, jobs={}), coord


def _job(prompt, generated, salt=7):
    j = _Job.__new__(_Job)
    j.ids, j.generated, j.apc_salt = list(prompt), list(generated), salt
    return j


def test_the_reply_cache_is_stored_under_prompt_plus_held_tokens():
    group, coord = _group(
        offset=6
    )  # 4 prompt + 2 of 3 reply tokens fed (the last one is pending)
    VLMBatchRunner._store_decode_cache(group, _job([1, 2, 3, 4], [10, 11, 12]))
    assert coord.calls == [([1, 2, 3, 4, 10, 11], 7)]


def test_a_cache_that_disagrees_with_the_tokens_is_not_stored():
    for offset in (4, 9):  # no reply token held / more than were emitted
        group, coord = _group(offset)
        VLMBatchRunner._store_decode_cache(group, _job([1, 2, 3, 4], [10, 11, 12]))
        assert coord.calls == []


def test_decode_is_the_default_mode():
    assert settings.get("YUNSHU_NEXT_TURN_PREFILL") == "decode"


def test_emitted_tokens_are_recorded():
    import queue

    r = VLMBatchRunner.__new__(VLMBatchRunner)
    r._vocab_owner, r.gc_guard = None, None
    j = _job([1], [])
    j.terminal, j.out = False, queue.Queue()
    j.stats, j.first_delivery = SimpleNamespace(generated=0), None
    r._emit(j, (42, None))
    r._emit(j, (43, None))
    assert j.generated == [42, 43]


def test_the_lanes_batch_cache_counts_rows_past_its_left_padding():
    """The speculative lane's single-row BatchKVCache has an array offset; the rows it holds are
    _idx past the left padding (the first 27B check stored nothing: only int offsets were read)."""
    import numpy as np

    class _BatchKV:
        keys = object()
        offset = np.array([99])
        _idx = 7
        left_padding = np.array([1])

    coord = _Coord()
    gen = SimpleNamespace(
        apc=coord,
        _generation_batch=SimpleNamespace(prompt_cache=[_BatchKV(), object()]),
    )
    group = SimpleNamespace(gen=gen, last_caches=None, spec=True, jobs={})
    VLMBatchRunner._store_decode_cache(group, _job([1, 2, 3, 4], [10, 11, 12]))
    assert coord.calls == [([1, 2, 3, 4, 10, 11], 7)]


def test_a_reply_whose_consumer_closed_after_the_last_token_is_still_stored():
    """Turn 3 of the 27B probe (2026-11-02): the client counted max_tokens and closed its
    stream before the finishing slice ran, so ``abandoned`` was set and every follow-up of a
    max_tokens reply missed the reply cache."""
    group, _ = _group(6)
    job = _job([1, 2, 3, 4], [10, 11, 12])
    job.cache_store, job.abandoned, job.overflowed = True, True, False
    assert VLMBatchRunner._decode_store_skip(group, job, "length") == []
    assert VLMBatchRunner._decode_store_skip(group, job, "cancel") == ["finish cancel"]
    job.overflowed = True
    assert VLMBatchRunner._decode_store_skip(group, job, "stop") == ["overflowed"]
    group.jobs = {2: object()}
    job.overflowed = False
    assert VLMBatchRunner._decode_store_skip(group, job, "stop") == ["other rows"]


def _runner():
    import threading

    r = VLMBatchRunner.__new__(VLMBatchRunner)
    r._lock, r._pending, r._executor = threading.Lock(), [], None
    r._groups = lambda: []
    return r


def test_the_store_waits_for_the_follow_up_not_for_a_resend(monkeypatch):
    """The copy into the APC (~45 ms at 8K) ran right after the reply, so the same prompt
    sent again waited for it (8K warm TTFT 20 -> 58 ms). Held until the follow-up's
    admission or an idle runner; a resend of the same prompt does not trigger it."""
    from yunshu_engine import dflash_apc_window
    from yunshu_engine.kernels import buffer_cache

    offered = []
    monkeypatch.setattr(buffer_cache, "offer_restore_buffers", offered.append)
    monkeypatch.setattr(dflash_apc_window, "take_decode_tail", lambda: None)
    stored = []
    monkeypatch.setattr(
        VLMBatchRunner,
        "_store_decode_cache",
        staticmethod(
            lambda g, j, caches=None, window_tail=None, salt=None: stored.append(j)
        ),
    )
    import yunshu_engine.spec_release as sr

    monkeypatch.setattr(sr, "finished_caches", lambda gen, last: ["kv"])
    r = _runner()
    group, _ = _group(6)
    reply = _job([1, 2, 3, 4], [10, 11, 12])
    r._defer_decode_cache(group, reply)
    assert stored == [] and offered == []
    assert not r._decode_follows(_job([1, 2, 3, 4], []))  # the same prompt again
    assert r._decode_follows(_job([1, 2, 3, 4, 10, 11, 12, 99], []))
    r._flush_decode_cache_if_idle(r._decode_pending)
    assert stored == [reply] and offered == [["kv"]]
    # a newer reply drops an unflushed older one instead of copying it
    r._defer_decode_cache(group, _job([5], [6]))
    newer = _job([7], [8])
    r._defer_decode_cache(group, newer)
    r._flush_decode_cache()
    assert stored == [reply, newer]


def test_round_driver_last_finished_row_defers_its_reply_cache():
    """The round driver's rows never reach the runner's _finish: its hook holds the reply cache
    (the same deferred store), and the driver leaves the buffers to the store."""
    from yunshu_engine.round_driver.driver import RoundDriver

    coord = _Coord()
    row = SimpleNamespace(
        req=SimpleNamespace(
            ids=[1, 2, 3, 4],
            handle=_job([1, 2, 3, 4], [10, 11, 12]),
            cache_store=True,
            use_apc=True,
            max_tokens=9,
        ),
        apc=coord,
        cache=[_KV(6), object()],
        context=[1, 2, 3, 4, 10, 11, 12],
        generated=3,
        salt=5,
        finished=True,
        slot=None,
        hslot=None,
    )
    runner = _runner()
    drv = RoundDriver.__new__(RoundDriver)
    drv.rows, drv.head, drv.dflash = [], None, None
    drv.finish_hook = runner._driver_store_decode
    drv._release([row])
    assert runner._decode_pending_job() is row.req.handle  # held, not stored yet
    assert coord.calls == []
    runner._flush_decode_cache()
    assert coord.calls == [([1, 2, 3, 4, 10, 11], 5)]  # the driver's own APC salt

    # a cancelled row (not finished) and a cache_store-off row hold nothing
    for tweak in ("finished", "cache_store"):
        r2 = SimpleNamespace(**row.__dict__)
        if tweak == "finished":
            r2.finished = False
        else:
            r2.req = SimpleNamespace(**{**row.req.__dict__, "cache_store": False})
        drv._release([r2])
        assert runner._decode_pending_job() is None


def test_driver_standing_handoff_preserves_row_salt_and_cache_ownership(monkeypatch):
    from mlx_vlm import apc

    from yunshu_engine import dflash_apc_window
    from yunshu_engine.kernels import buffer_cache

    group, coord = _group(6)
    group.driver_salt = 5
    standing = []
    coord.offer_standing = lambda tokens, rows, **kw: standing.append(
        (list(tokens), rows, kw["extra_hash"])
    )
    rows = [object()]
    monkeypatch.setattr(apc, "snapshot_prompt_cache_row", lambda *a, **kw: rows)
    monkeypatch.setattr(dflash_apc_window, "store_decode_window", lambda *a, **kw: True)
    offered = []
    monkeypatch.setattr(buffer_cache, "offer_restore_buffers", offered.append)
    runner = _runner()
    job = _job([1, 2, 3, 4], [10, 11, 12], salt=7)
    caches = [_KV(6)]
    runner._decode_pending = (group, job, caches, None)
    assert runner._stand_decode_cache()
    assert standing == [([1, 2, 3, 4, 10, 11], rows, 5)]
    assert runner._decode_pending_job() is None
    assert coord.calls == [] and offered == []
