"""the VLM NON-streaming chat path (_handle_vlm_chat, n=1 and n>1) had NO
cancellation wiring — no RequestTracker.register, no cancel_event in gen_kwargs, and no
run_with_disconnect_guard. So POST /v1/cancel 404'd for an in-flight image/video
generation, and a client HTTP disconnect did NOT stop the GPU (it ran to max_tokens). A
fresh, un-propagated sibling of the W798/W806 cancel keystone — and the most expensive
path to leave uncancellable. Now it registers under completion_id, threads cancel_event
into generate, wraps the work in run_with_disconnect_guard, and unregisters in finally."""
from __future__ import annotations

import inspect

from yunshu_gateway.routers.chat import _handle_vlm_chat


def test_vlm_nonstream_has_cancellation_wiring():
    src = inspect.getsource(_handle_vlm_chat)
    # the non-streaming branch (after the `if req.stream:` streaming return) must:
    # 1) register a tracker entry under completion_id
    assert "get_request_tracker()" in src
    assert ".register(completion_id, req.model)" in src
    # 2) thread the cancel_event into the engine call
    assert 'gen_kwargs["cancel_event"] = _vlm_cancel' in src
    # 3) wrap the generation in the disconnect guard with that cancel_event
    assert "run_with_disconnect_guard(" in src
    assert "cancel_event=_vlm_cancel" in src
    # 4) unregister in finally (no leaked tracker entry → no false "busy")
    assert "_vlm_tracker.unregister(completion_id)" in src


def test_cancel_event_propagates_to_all_choices():
    # cancel_event is set on gen_kwargs BEFORE the choice loop, so the n>1 _vlm_gen_one
    # closure (which spreads {**gen_kwargs, ...}) passes it to every choice.
    src = inspect.getsource(_handle_vlm_chat)
    i_set = src.index('gen_kwargs["cancel_event"] = _vlm_cancel')
    i_run = src.index("_run_all_choices()")
    assert i_set < i_run, "cancel_event must be set on gen_kwargs before choices run"
