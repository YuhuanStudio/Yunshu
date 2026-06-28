"""the W902 image cancel fix (img2img/variations) was never swept to the
controlnet / inpaint / depth-guided pipelines or the /generations control-image branch — a
client disconnect there ran all n diffusions to completion, head-of-line-blocking the serial
MLX executor. Thread cancel_event through each async wrapper → _run_cancellable (thread-safe
flag + late-disconnect watcher) → the sync pipeline's per-step cancel_flag check, and
register + run_with_disconnect_guard in the 4 router handlers.
"""
from __future__ import annotations

import inspect

from yunshu_engine.image_engine import ImageGenEngine
from yunshu_gateway.routers import images


def test_engine_pipelines_check_cancel_flag_per_step():
    for name in ("_run_pipeline", "_run_inpaint_pipeline",
                 "_run_controlled_pipeline", "_run_depth_guided_pipeline"):
        src = inspect.getsource(getattr(ImageGenEngine, name))
        assert "cancel_flag=None" in src, f"{name} missing cancel_flag param"
        assert "cancel_flag is not None and cancel_flag.is_set()" in src, f"{name} no per-step check"


def test_engine_wrappers_thread_cancel_event():
    for name in ("generate_controlled_image", "generate_controlled",
                 "inpaint", "generate_depth_guided"):
        src = inspect.getsource(getattr(ImageGenEngine, name))
        assert "cancel_event=None" in src, f"{name} missing cancel_event param"
        assert "self._run_cancellable(_make, cancel_event)" in src, f"{name} not routed via _run_cancellable"


def test_run_cancellable_helper_present():
    src = inspect.getsource(ImageGenEngine._run_cancellable)
    assert "threading.Event()" in src
    assert "_watch" in src
    assert "make_sync(_cancel)" in src


def test_router_handlers_register_and_guard():
    for name in ("create_image_inpaint", "create_image_controlnet",
                 "create_image_depth_guided", "create_image"):
        src = inspect.getsource(getattr(images, name))
        assert "_register_image_cancel(req.model)" in src, f"{name} no cancel registration"
        assert "run_with_disconnect_guard(" in src, f"{name} no disconnect guard"
        assert "_unregister_image_cancel(_img_tracker, _img_id)" in src, f"{name} no cleanup"
