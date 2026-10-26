"""A timed-out legacy non-stream generate() keeps its lease (``_active_count``) until the
executor thread actually exits, like the stream path; ``runner.inflight`` reads this count."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from yunshu_engine.vlm_engine import VLMEngine


def _engine(release: threading.Event, executor):
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = object()
    eng._model_path = "local-model"
    eng._model_name = "m"
    eng._num_requests_processed = 0
    eng._total_reasoning_tokens = 0
    eng._active_count = 0
    eng._active_count_lock = threading.Lock()
    eng._temp_files = None
    eng._temp_files_lock = threading.Lock()
    eng._executor = executor

    async def _none(*a, **k):
        return []

    eng._orphan_key = lambda messages, params: None
    eng._request_template_extra = lambda kw: None
    eng._extract_images = _none
    eng._extract_audio = _none
    eng._extract_video_frames = _none
    eng._default_enable_thinking = lambda *a, **k: False
    eng._track_pipeline = lambda *a, **k: None
    eng._check_request_supported = lambda *a, **k: None
    eng._settle_rehearsal_early = _none
    eng._runner_kwargs = lambda **k: {}
    eng._runner_input = lambda *a, **k: ([1, 2], {}, None)
    eng._settle_rehearsal = lambda ids: None
    eng._resolve_prompt_cache_plan = lambda *a, **k: None
    eng._runner_consumers = lambda: executor

    def _worker(*a, **k):
        # stands in for the runner consumer: ignores the timeout, exits on `release`
        release.wait(10)
        return ("x", 0, 1, True, False, 0)

    eng._generate_vlm_runner_text = _worker
    return eng


def test_timeout_keeps_lease_until_worker_exits():
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=2)
    eng = _engine(release, executor)

    async def run():
        out = await eng.generate(
            messages=[{"role": "user", "content": "hi"}], timeout_seconds=0.2
        )
        during = eng._active_count
        release.set()
        for _ in range(100):
            if eng._active_count == 0:
                break
            await asyncio.sleep(0.02)
        return out, during, eng._active_count

    try:
        out, during, after = asyncio.run(run())
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert out["finish_reason"] == "timeout"
    assert during == 1, "lease released while the executor thread was still running"
    assert after == 0


def test_normal_completion_releases_immediately():
    release = threading.Event()
    release.set()
    executor = ThreadPoolExecutor(max_workers=2)
    eng = _engine(release, executor)
    try:
        out = asyncio.run(eng.generate(messages=[{"role": "user", "content": "hi"}]))
    finally:
        executor.shutdown(wait=True)
    assert out["finish_reason"] == "stop"
    assert eng._active_count == 0
