"""R11: the VLM runner decodes in single-step slices on the shared MLX executor, so another
modality's work (ASR, TTS, OCR) queued during a long generation runs after at most one
slice, not after the whole generation."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tests.unit.test_vlm_runner_batching import FakeGen, runner  # noqa: F401,F811

STEP_S = 0.01


class _SlowGen(FakeGen):
    def next(self):
        time.sleep(STEP_S)
        return super().next()


def test_other_modality_waits_one_slice_not_the_whole_generation(runner, monkeypatch):  # noqa: F811
    import importlib

    ar = importlib.import_module("mlx_vlm.generate.ar")
    monkeypatch.setattr(ar, "BatchGenerator", _SlowGen)
    ex = ThreadPoolExecutor(max_workers=1)
    runner._executor = ex
    tokens: list[int] = []

    def consume():
        tokens.extend(runner.iter_tokens([1, 2, 3], max_tokens=200, prompt_kwargs={}))

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.15)  # well inside the 200-step (about 2 s) generation
    queued = time.perf_counter()
    other = ex.submit(time.perf_counter)  # an ASR / TTS call reaching the executor
    waited = other.result(timeout=5) - queued
    assert t.is_alive(), (
        "the generation must still be running for this to mean anything"
    )
    assert waited < 20 * STEP_S, f"waited {waited:.3f}s behind the generation"
    t.join(120)  # generous: slow when tests run on efficiency cores
    assert len(tokens) == 200
    ex.shutdown(wait=True)
