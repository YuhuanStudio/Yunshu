"""Cache lifecycle events name their request: the id queue the APC pops at plan time.

Pure Python (cache_observation has no mlx import): the round driver's admission registers the
request id under the prompt ids, the APC's checkpoint plan pops it into the observation, and the
events written while the row prefills carry it. Regression: the 2026-11-08 round-driver merge
dropped the registration, so every checkpoint event had request_id None (console_backend_gaps).
"""

import collections

from yunshu_engine.cache_observation import (
    CONSOLE_REQUEST_CAP,
    Observation,
    register_console_request,
    release_console_request,
)


def _plan_pop(pending, ids):
    """What APCManager's checkpoint_lengths(begin=True) does with the queue."""
    queue = pending.get(tuple(ids))
    request_id = queue.popleft() if queue else None
    if queue is not None and not queue:
        pending.pop(tuple(ids), None)
    return request_id


def test_registered_id_links_the_plan_and_the_events_it_writes():
    pending = collections.OrderedDict()
    obs = Observation()
    ids = [5, 6, 7, 8]
    register_console_request(pending, ids, "consolegaps-schema")
    # the plan at begin: set the link from the queue, as checkpoint_lengths does
    obs.request_id = _plan_pop(pending, ids)
    assert obs.request_id == "consolegaps-schema"
    obs.event("admission", "k1", 4, "ram", "store")
    assert obs.events[-1]["request_id"] == "consolegaps-schema"
    # the queue entry is consumed: the next plan for the same ids is unlinked, not stale
    assert tuple(ids) not in pending
    assert _plan_pop(pending, ids) is None


def test_queue_is_fifo_per_prompt_and_release_drops_only_the_leftover():
    pending = collections.OrderedDict()
    ids = [1, 2, 3]
    register_console_request(pending, ids, "first")
    register_console_request(pending, ids, "second")
    assert _plan_pop(pending, ids) == "first"
    # the second row failed before its plan: its registration is released, not left behind
    release_console_request(pending, ids, "second")
    assert tuple(ids) not in pending


def test_release_after_consumption_is_a_no_op_and_keeps_other_rows():
    pending = collections.OrderedDict()
    register_console_request(pending, [9], "a")
    register_console_request(pending, [9], "b")
    assert _plan_pop(pending, [9]) == "a"  # consumed by the planned row
    release_console_request(pending, [9], "a")  # already gone: nothing removed
    assert list(pending[(9,)]) == ["b"]
    release_console_request(pending, [7], "x")  # unknown ids: no-op


def test_queue_is_bounded():
    pending = collections.OrderedDict()
    for i in range(CONSOLE_REQUEST_CAP + 10):
        register_console_request(pending, [i], f"r{i}")
    assert len(pending) == CONSOLE_REQUEST_CAP
    assert (0,) not in pending


def test_unlinked_plan_yields_none_not_a_previous_request():
    """Without a registration the event is unlinked (never the previous request's id)."""
    obs = Observation()
    obs.request_id = "previous"
    pending = collections.OrderedDict()
    obs.request_id = _plan_pop(pending, [1, 2])
    obs.event("admission", "k", 2, "ram", "store")
    assert obs.events[-1]["request_id"] is None
