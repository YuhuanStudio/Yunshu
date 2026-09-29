"""A normally finished VLM stream must not set the request's cancel event.

The cancel event is shared with the router, which reads it once the stream ends to
decide "cancelled" vs "completed". The engine used to set it on exit whenever the
runner job had not yet returned (it winds down just after the last token), so a
healthy request was sometimes reported as ``response.incomplete: cancelled``.
"""

import asyncio
import concurrent.futures
import threading
import time

from yunshu_engine.request_tracker import RequestTracker
from yunshu_engine.vlm_engine import RequestOutput, VLMEngine


def _engine(finish_delay: float):
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = object()
    eng._model_path = "org/m"
    eng._executor = concurrent.futures.ThreadPoolExecutor(1)
    eng._active_count = 0
    eng._active_count_lock = threading.Lock()
    eng._temp_files = None
    eng._temp_files_lock = threading.Lock()
    eng._batch_runner = object()

    async def _none(_messages):
        return []

    eng._extract_images = _none
    eng._extract_audio = _none
    eng._extract_video_frames = _none
    eng._request_template_extra = lambda kw: None
    eng._default_enable_thinking = lambda et, constrained=False: et
    eng._track_pipeline = lambda *a, **k: None
    eng._check_request_supported = lambda *a, **k: None
    eng._runner_kwargs = lambda **p: {}
    eng._runner_input = lambda *a, **k: ([1], {}, None)
    eng._cleanup_temp_files = lambda files: None

    def fake_runner(ids, req_id, q, **kw):
        q.put_nowait(RequestOutput(request_id=req_id, new_text="hi"))
        q.put_nowait(
            RequestOutput(
                request_id=req_id, new_text="", finished=True, finish_reason="stop"
            )
        )
        time.sleep(finish_delay)  # runner still winding down after the last token
        q.put_nowait(None)

    eng._stream_vlm_runner_text = fake_runner
    return eng


async def _drive(eng, cancel):
    outs = []
    async for o in eng.generate_stream(prompt="x", cancel_event=cancel):
        outs.append(o)
    return outs


def test_normal_finish_does_not_set_cancel_event():
    eng = _engine(finish_delay=0.3)
    gen = RequestTracker().register("r1", "m")
    outs = asyncio.run(_drive(eng, gen.cancel_event))
    assert outs and outs[-1].finished
    assert not gen.cancel_event.is_set()


def test_client_break_still_cancels():
    eng = _engine(finish_delay=0.3)
    cancel = threading.Event()

    async def go():
        agen = eng.generate_stream(prompt="x", cancel_event=cancel)
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(go())
    assert cancel.is_set()
