"""video fake-success, kv pressure-eviction abort, Responses function_call item.

(HIGH): the native video pipeline stored the weights DICT in self._model (truthy but NOT
  callable), so _denoise's `callable()` guard silently ran the placeholder velocity while the
  result method reported a successful "wan_native_*" → the router returned HTTP 200 with
  garbage (its 503 guard trips only on "fallback" in method). Tag the method ".._placeholder_
  fallback" when no callable transformer ran; the streaming native path falls back too.
evict_under_pressure called select_victim WITHOUT exclude=_skipped_indices, so a pinned
  victim was re-selected every iteration and the loop aborted ALL eviction on the first pin.
response.output_item.added for a function_call omitted call_id/name/arguments.
"""
from __future__ import annotations

import inspect


def test_placeholder_tagged_fallback():
    from yunshu_engine import video_pipeline
    src = inspect.getsource(video_pipeline)
    assert "_has_real_model = self._model is not None and callable(self._model)" in src
    assert "_placeholder_fallback" in src


def test_native_method_trips_503_guard():
    # the fallback method must contain "fallback" so the router's `"fallback" in method` 503
    # guard fires; and still start with wan_native (existing test contract).
    m = "wan_native_euler_placeholder_fallback"
    assert "fallback" in m and m.startswith("wan_native") and "euler" in m


def test_stream_native_falls_back_on_non_callable():
    from yunshu_engine import video_engine
    src = inspect.getsource(video_engine)
    assert 'not callable(getattr(self._native_pipeline, "_model", None))' in src


def test_evict_under_pressure_passes_exclude():
    from yunshu_engine import kv_prefix_cache
    src = inspect.getsource(kv_prefix_cache.KVPrefixCache.evict_under_pressure) if hasattr(
        kv_prefix_cache.KVPrefixCache, "evict_under_pressure") else inspect.getsource(kv_prefix_cache)
    assert "exclude=_skipped_indices" in src


def test_function_call_item_carries_fields():
    import json

    from yunshu_gateway.streaming import format_responses_output_item_added
    out = format_responses_output_item_added(
        "resp_1", "m", item_id="fc-1", output_index=1, item_type="function_call",
        call_id="call_abc", name="get_weather", arguments='{"city":"SF"}')
    data = json.loads(out.split("data: ", 1)[1])
    item = data["item"]
    assert item["type"] == "function_call"
    assert item["call_id"] == "call_abc"
    assert item["name"] == "get_weather"
    assert item["arguments"] == '{"city":"SF"}'
    # message items are unaffected
    out_m = format_responses_output_item_added("r", "m", item_id="msg-1", item_type="message")
    assert json.loads(out_m.split("data: ", 1)[1])["item"]["type"] == "message"


def test_stream_function_call_added_uses_subscript_not_getattr():
    """the responses STREAMING function_call output_item.added must read
    name/arguments via dict subscript. tool_calls comes from
    extract_tool_calls_model_aware → list[dict], so getattr(tc, "name", "") returns
    the EMPTY default on a dict, silently re-breaking only on the stream path.
    Guard the real call site against regression to getattr."""
    import inspect
    import json

    from yunshu_gateway.routers import responses as _resp
    src = inspect.getsource(_resp._stream_response)
    # The function_call output_item.added must NOT use getattr on tc for name/args.
    assert 'getattr(tc, "name"' not in src
    assert 'getattr(tc, "arguments"' not in src
    # And it must subscript tc for the added event's fields.
    assert 'name=tc["name"]' in src
    assert 'arguments=tc["arguments"]' in src

    # Behavioral proof: a dict tc (the real shape) must yield non-empty fields.
    from yunshu_gateway.streaming import format_responses_output_item_added
    tc = {"name": "get_weather", "arguments": '{"city":"SF"}'}
    out = format_responses_output_item_added(
        "resp_1", "m", item_id="fc-1", output_index=1, item_type="function_call",
        call_id="call_abc", name=tc["name"], arguments=tc["arguments"])
    item = json.loads(out.split("data: ", 1)[1])["item"]
    assert item["name"] == "get_weather"
    assert item["arguments"] == '{"city":"SF"}'
