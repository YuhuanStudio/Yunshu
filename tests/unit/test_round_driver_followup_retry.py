"""A follow-up resumes its previous reply's cache three ways, and each generates the same tokens:

- ``copy``: the reply was stored and the follow-up restores a copy of it (the APC path);
- ``stage``: stored and also held standing (the idle store): the follow-up appends to the reply's own
  buffers, which keep spare capacity, so nothing is copied;
- ``stand``: held standing only (a follow-up that came before the idle store).

A retry of the follow-up (the same prompt again) restores the very state the follow-up resumed from and
generates the same tokens: in ``stage`` the stored copy, kept by the supersede of the follow-up's own
reply; in ``stand`` the restore point republished from the follow-up's prefill buffers once the decode
batch copied them (no copy: the checkpoint is their only holder, not also the restore pool). Before
2026-11-12 both retries fell back to an older checkpoint (2.95 s at 33K)."""

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

    # the served 27B's layout: split-invariant prefill, checkpoints only on the 64 grid
    return YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        overrides={
            "memory_max_gb": 1,
            "checkpoint_entries": 8,
            "checkpoint_interval_tokens": 256,
        },
        prefill_stride=2048,
        prefill_alignment=64,
    )


DOC = [(7 * i + 11) % 500 for i in range(300)]
ASK = [(5 * i + 3) % 500 for i in range(20)]
N = 24


def _session(lm, drafter, mode):
    """Turn 1, its follow-up, and a retry of the follow-up; returns (outputs, hits, manager, follow-up prompt)."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    mgr = _manager()
    d = RoundDriver(lm, drafter=drafter, stop_tokens=set(), chunk=128, apc=mgr)

    def hook(row):
        caches = list(row.cache)
        if row.head_cache is not None:
            caches += row.head_cache
        offs = {int(c.offset) for c in caches if getattr(c, "keys", None) is not None}
        assert len(offs) == 1, offs
        n = offs.pop()
        tokens = row.context[:n]
        row.apc.take_standing()
        if mode == "copy":
            assert row.apc.store_decode_tail(tokens, caches, extra_hash=row.salt)
            return False
        if mode == "stage":
            # the runner's idle store (_flush_decode_cache): the APC keeps the exact cut-out, and a
            # copy with spare keys stands for the follow-up
            from yunshu_engine.vlm_batch_runner import _standing_copy

            assert row.apc.store_decode_tail(
                tokens, caches, extra_hash=row.salt, owned=True
            )
            caches = _standing_copy(caches, n)
        # the runner's hand-off: the buffers stay held for the follow-up
        row.apc.offer_standing(tokens, caches, extra_hash=row.salt)
        return True

    d.finish_hook = hook
    outs: list = []
    hits: list = []
    turn2: list = []
    for turn in range(3):
        if turn == 1:
            turn2 = DOC + outs[0] + ASK
        hits.append(d.add(Request(turn2 or DOC, N, handle=turn)))
        out = []
        while d.busy():
            out += [e.token for e in d.step()]
        outs.append(out)
    return outs, hits, mgr, turn2


@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("mode", ["copy", "stage", "stand"])
def test_follow_up_resumes_and_its_retry_is_bit_identical(tiny, mode, draft):
    lm, drafter = tiny
    outs, hits, _, _ = _session(lm, drafter if draft else None, mode)
    assert hits[1] == len(DOC) + N - 1  # prompt + the reply tokens the cache holds
    # the retry restores the state the follow-up resumed from and replays it exactly
    assert hits[2] == hits[1], hits
    assert outs[2] == outs[1]
    # the reply's own buffers (spare capacity) and a copy of them give the same bits, and
    # spec on == spec off whichever way the follow-up resumed
    assert outs == _session(lm, None, "copy")[0]


def test_republished_restore_point_owns_the_row_buffers(tiny):
    """``stand``: the restore point is the follow-up's own prefill buffers (their capacity, rows past the
    restore point zeroed), stored once, held by no restore pool too."""
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.apc_manager import _sequence_hash
    from yunshu_engine.kernels import buffer_cache

    lm, drafter = tiny
    outs, hits, mgr, turn2 = _session(lm, drafter, "stand")
    prefix = hits[1]
    salts = {e.extra_hash for e in mgr._exact_cache.values()}
    assert len(salts) == 1
    salt = salts.pop()
    assert mgr.holds_exact(turn2[:prefix], salt)
    entry = mgr._exact_cache[
        _sequence_hash(tuple(turn2[:prefix]), salt, mgr.block_size)
    ]
    kv = [c for c in entry.prompt_cache if type(c) is KVCache]
    assert kv
    pooled = {id(a) for a in buffer_cache._RESTORE_BUFFERS}
    for c in kv:
        assert c.offset == prefix
        # a copy would hold just the rows; this is the row's buffer, reserved for the whole prompt
        assert c.keys.shape[-2] >= len(turn2) > prefix
        assert not mx.any(c.keys[..., prefix:, :]).item()
        assert not mx.any(c.values[..., prefix:, :]).item()
        assert id(c.keys) not in pooled and id(c.values) not in pooled
