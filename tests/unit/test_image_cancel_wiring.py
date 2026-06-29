"""the NON-streaming image routes (edit/variation/inpaint/...) never received the
client-disconnect cancel keystone — only the streaming generate path did,
despite commit note claiming images were covered. An abandoned n>1 request ran all n
diffusions to completion on the serial GPU executor, head-of-line-blocking other work.

variation + edit (which call the engine's generate(), the method that honors cancel_event
mid-diffusion) now register with the request tracker, wrap each generate call in
run_with_disconnect_guard, pass cancel_event, break the n-loop on disconnect, and unregister
in finally. inpaint/controlnet/depth use engine methods without a cancel_event param
(documented follow-up — needs an engine-side change).
"""
from __future__ import annotations

import inspect
import threading

from yunshu_gateway.routers import (
    images as I,  # noqa: N812  # intentional short module alias
)


def test_register_helper_returns_cancel_event(monkeypatch):
    class _Gen:
        def __init__(self):
            self.cancel_event = threading.Event()

    class _Tracker:
        def __init__(self):
            self.reg, self.unreg = [], []

        def register(self, rid, model):
            self.reg.append((rid, model))
            return _Gen()

        def unregister(self, rid):
            self.unreg.append(rid)

    tracker = _Tracker()
    import yunshu_engine.request_tracker as rt
    monkeypatch.setattr(rt, "get_request_tracker", lambda: tracker)

    ev, tr, rid = I._register_image_cancel("z-image")
    assert isinstance(ev, threading.Event)
    assert tr is tracker and tracker.reg == [(rid, "z-image")]
    I._unregister_image_cancel(tr, rid)
    assert tracker.unreg == [rid]


def test_register_helper_tracker_unavailable_is_safe(monkeypatch):
    import yunshu_engine.request_tracker as rt
    monkeypatch.setattr(rt, "get_request_tracker", lambda: (_ for _ in ()).throw(RuntimeError()))
    ev, tr, rid = I._register_image_cancel("m")
    assert ev is None and tr is None and rid.startswith("img-")
    I._unregister_image_cancel(tr, rid)  # must not raise


def test_variation_and_edit_routes_wire_cancel():
    for fn in (I.create_image_variation, I.create_image_edit):
        src = inspect.getsource(fn)
        assert "run_with_disconnect_guard(" in src, f"{fn.__name__} missing disconnect guard"
        assert "cancel_event=_img_cancel" in src, f"{fn.__name__} missing engine cancel_event"
        assert "_register_image_cancel(" in src
        assert "_unregister_image_cancel(" in src
        assert "if result is None:" in src  # n-loop breaks on disconnect
