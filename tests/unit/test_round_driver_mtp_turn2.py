"""A packed MTP-batch row stores its finished reply (target cache + head cache) like a DFlash-batch row
does, and the follow-up turn restores it: turn 2 equals the drafting-off turn 2 (bf16 and int8 KV)."""

import pytest

from .test_round_driver import _tiny

mx = pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def tiny():
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    return _tiny()


def _manager():
    from yunshu_engine.apc_manager import YunshuAPCManager

    return YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        overrides={
            "memory_max_gb": 1,
            "checkpoint_entries": 8,
            "checkpoint_interval_tokens": 256,
        },
    )


DOC = [(7 * i + 11) % 500 for i in range(300)]
ASK = [(5 * i + 3) % 500 for i in range(20)]
N = 24


def _turns(lm, drafter, kv_format, stored):
    """Turn 1 then a follow-up that repeats prompt + reply; returns (reply, turn-2 tokens, turn-2 hit)."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(
        lm,
        drafter=drafter,
        stop_tokens=set(),
        chunk=128,
        apc=_manager(),
        kv_format=kv_format,
    )

    def hook(row):
        caches = list(row.cache) if row.cache is not None else None
        if caches is None:
            stored.append("no cache")
            return False
        if row.head_cache is not None:
            caches += row.head_cache
        offs = {int(c.offset) for c in caches if getattr(c, "keys", None) is not None}
        assert len(offs) == 1, offs
        held = offs.pop()
        ok = row.apc.store_decode_tail(row.context[:held], caches, extra_hash=row.salt)
        assert ok
        stored.append(held)
        return False

    d.finish_hook = hook
    outs, hits = [], []
    prompt = DOC
    for turn in range(2):
        hits.append(d.add(Request(prompt, N, handle=turn)))
        out = []
        while d.busy():
            out += [e.token for e in d.step()]
        outs.append(out)
        prompt = prompt + out + ASK
    return outs, hits


@pytest.mark.parametrize("kv_format", ["bf16", "int8"])
def test_mtp_row_reply_is_stored_and_turn2_equals_plain(tiny, kv_format):
    lm, drafter = tiny
    on_stored, off_stored = [], []
    on, on_hits = _turns(lm, drafter, kv_format, on_stored)
    off, off_hits = _turns(lm, None, kv_format, off_stored)
    assert on_stored and isinstance(on_stored[0], int), (
        on_stored
    )  # (a) the reply was stored
    assert on_stored[0] > len(DOC)
    assert on_hits[1] > len(DOC), (
        on_hits
    )  # (b) turn 2 restores prompt + reply, not just the prompt
    assert on_hits[1] == off_hits[1], (on_hits, off_hits)
    assert on[0] == off[0]
    assert on[1] == off[1]  # (c) spec on == spec off
