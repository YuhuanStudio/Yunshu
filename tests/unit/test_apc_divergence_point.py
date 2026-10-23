"""A prompt that diverges from a recent one checkpoints the shared head for later sessions."""

import mlx.core as mx
import pytest

from tests.unit.test_apc_manager import _coordinator, _mgr


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def _ids(session, n=400, shared=200):
    return list(range(shared)) + [10_000 + session] * (n - shared)


def test_second_session_checkpoints_the_shared_head():
    manager = _mgr(prefill_alignment=64)
    c = _coordinator(manager)
    first = c.checkpoint_lengths(_ids(1), set())
    assert 192 not in first  # nothing to diverge from yet
    second = c.checkpoint_lengths(_ids(2), set())
    assert 192 in second  # lcp 200 aligned down to 192
    # every call of one request agrees (the runner asks more than once)
    assert c.checkpoint_lengths(_ids(2), set(), begin=False) == second


def test_next_turn_of_the_same_conversation_is_not_a_divergence():
    manager = _mgr(prefill_alignment=64)
    c = _coordinator(manager)
    base = _ids(1)
    c.checkpoint_lengths(base, set())
    follow = base + list(range(500, 700))
    assert manager.divergence_point(follow) == 0


def test_next_turn_resumes_from_the_servers_own_prompt_end_checkpoint():
    # The tail of turn k is not one of turn k+1's declared breakpoints; the explicit-plan lookup used to
    # ignore it, so a conversation whose last block carried no marker stayed at its first checkpoint.
    from mlx_vlm.models.cache import ArraysCache

    manager = _mgr(prefill_alignment=64)
    c = _coordinator(manager)
    first = list(range(300))
    policy = {
        "points": [(128, 300), (256, 300)],
        "writes": [(128, 300), (256, 300)],
        "lookup_points": [128, 256],
        "tail": 256,
        "written": 0,
    }
    c.set_request(first, policy)
    c.checkpoint_lengths(first, set())
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    for n in (128, 256):
        assert c.store_checkpoint(first[:n], [rec])
    second = first + list(range(1000, 1200))
    policy2 = {
        "points": [(128, 300), (448, 300)],
        "writes": [(128, 300), (448, 300)],
        "lookup_points": [128, 448],
        "tail": 448,
        "written": 0,
    }
    c.set_request(second, policy2)
    got = c.lookup(
        second,
        extra_hash=0,
        safe_lookup_min=0,
        suffix_is_text_only=lambda _: True,
        prefix_has_media=lambda _: False,
    )
    assert got is not None and got["prefix_len"] == 256
