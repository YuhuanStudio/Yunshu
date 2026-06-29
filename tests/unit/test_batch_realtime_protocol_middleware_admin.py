"""realtime protocol-completeness + middleware/admin hardening.

realtime function_call output items are now announced via response.output_item.added
  at distinct output_index 1+i with their OWN item id (deltas/done reference that id, not the
  assistant message item at index 0), and appear in response.done's output array.
prometheus_exporter._format_labels now escapes \\r (keystone propagation).
admin unload_model honors the bool return ("already_unloaded" when refused/no-op).
realtime response.cancel with no active response emits an error (was a silent no-op).
session.update validates modalities (list drawn from {text,audio}).
"""

from __future__ import annotations

import inspect


def test_function_call_emits_output_item_added_with_own_id():
    from yunshu_gateway.routers import realtime

    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    # each function call is its own output item at output_index 1+i
    assert "_fc_oidx = 1 + _fc_i" in src
    assert '"response.output_item.added"' in src
    # deltas/done reference the function-call item's OWN id, not item_id (the message)
    assert "item_id=_fc_item.item_id" in src
    # function_call items are included in response.done output
    assert "[fi.to_dict() for fi in _fc_items]" in src
    # the old leak — streaming under the message item_id at output_index 0 — is gone
    assert (
        "item_id=item_id,\n                        output_index=0,\n                        call_id"
        not in src
    )


def test_metrics_middleware_is_outermost():
    """MetricsMiddleware must wrap auth/rate-limit so it records their rejections
    (401/429/413/503) — it was innermost and saw none of them."""
    from yunshu_gateway.main import create_app

    app = create_app()
    names = [m.cls.__name__ for m in app.user_middleware]
    # outermost is index 0 (Starlette wraps in order); Metrics must precede TenantAuth
    assert names[0] == "MetricsMiddleware"
    assert names.index("MetricsMiddleware") < names.index("TenantAuthMiddleware")
    assert names.index("MetricsMiddleware") < names.index("RateLimitMiddleware")


def test_prometheus_exporter_escapes_carriage_return():
    from yunshu_gateway.middleware import prometheus_exporter as pe

    # build a label set with an embedded CR and confirm it's escaped
    out = pe._format_labels(frozenset({("endpoint", "a\rb")}))
    assert "\\r" in out
    assert "\r" not in out


def test_realtime_cancel_no_active_emits_error():
    from yunshu_gateway.routers import realtime

    src = inspect.getsource(realtime.RealtimeSession._handle_response_cancel)
    assert "response_cancel_not_active" in src
    # the guard fires when there is no live task
    assert "if not (task and not task.done()):" in src


def test_session_update_validates_modalities():
    # find the SessionConfig class (has update() + SUPPORTED_AUDIO_FORMATS) and drive it
    from yunshu_gateway.routers import realtime as _rt

    cfg_cls = None
    for _name, obj in vars(_rt).items():
        if (
            isinstance(obj, type)
            and hasattr(obj, "update")
            and hasattr(obj, "SUPPORTED_AUDIO_FORMATS")
        ):
            cfg_cls = obj
            break
    assert cfg_cls is not None
    assert 'if key == "modalities":' in inspect.getsource(cfg_cls.update)
    cfg = cfg_cls()
    before = list(cfg.modalities)
    # garbage rejected
    cfg.update({"modalities": "text"})  # not a list
    assert cfg.modalities == before
    cfg.update({"modalities": ["video"]})  # invalid member
    assert cfg.modalities == before
    cfg.update({"modalities": []})  # empty
    assert cfg.modalities == before
    # valid accepted
    cfg.update({"modalities": ["text"]})
    assert cfg.modalities == ["text"]
