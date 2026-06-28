"""image generate() swallowed cancel_event, so /images/variations and
/images/edits ran the full img2img diffusion to completion after a client disconnect —
the router's "honors cancel_event mid-diffusion" comment was false. Thread cancel_event
through generate → _generate_variation → _run_img2img_pipeline and check a thread-safe
flag per denoise step.
"""
from __future__ import annotations

import inspect

from yunshu_engine.image_engine import ImageGenEngine


def test_generate_forwards_cancel_event_to_variation():
    src = inspect.getsource(ImageGenEngine.generate)
    assert 'cancel_event=kwargs.get("cancel_event")' in src


def test_variation_accepts_and_mirrors_cancel_event():
    src = inspect.getsource(ImageGenEngine._generate_variation)
    assert "cancel_event=None" in src
    # mirrored into a thread-safe flag and threaded into the pipeline
    assert "threading.Event()" in src
    assert "cancel_flag=_cancel" in src
    # a watcher task propagates a late disconnect into the flag
    assert "_watch_cancel" in src


def test_img2img_pipeline_checks_cancel_flag_per_step():
    src = inspect.getsource(ImageGenEngine._run_img2img_pipeline)
    assert "cancel_flag=None" in src
    assert "cancel_flag is not None and cancel_flag.is_set()" in src
    # the check sits inside the denoise loop (before the transformer forward)
    i = src.index("for t in range(start_step, num_steps):")
    j = src.index("self._transformer(", i)
    assert "cancel_flag.is_set()" in src[i:j]
