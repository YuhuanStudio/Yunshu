"""OCREngine generates through VLMEngine.generate (the round driver), not mlx-vlm's stock
generate. Fake VLM engine, CPU only: asserts the call shape (messages, task prompt, language
hint, greedy, 4096-token cap, cancel_event forwarded), the start/stop lifecycle, and the
OCR-only image-token estimate that now reads the VLM's config."""

from __future__ import annotations

import asyncio
import threading
import types

import pytest
from PIL import Image


def _png(tmp_path, size=(64, 48), name="x.png"):
    path = tmp_path / name
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


class _FakeVLM:
    """Stands in for VLMEngine: records generate() kwargs, returns a fixed reply."""

    def __init__(self, text="INVOICE 2026", config=None):
        self.calls: list[dict] = []
        self.is_loaded = True
        self.stopped = False
        self._text = text
        self._config = config or {}
        self._model = None

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": self._text,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "finish_reason": "stop",
        }

    async def stop(self):
        self.stopped = True
        self.is_loaded = False


def _engine(vlm):
    from yunshu_engine.ocr_engine import OCREngine

    engine = OCREngine("/models/GLM-OCR-bf16")
    engine._vlm = vlm
    engine._owns_vlm = True  # built by start(), so stop() releases it
    return engine


def test_extract_text_generates_through_vlm_engine(tmp_path):
    png = _png(tmp_path)
    vlm = _FakeVLM()
    result = asyncio.run(_engine(vlm).extract_text(png, task="text"))

    assert len(vlm.calls) == 1
    kw = vlm.calls[0]
    assert kw["max_tokens"] == 4096
    assert kw["temperature"] == 0.0
    assert kw["enable_thinking"] is False
    (message,) = kw["messages"]
    assert message["role"] == "user"
    image_part, text_part = message["content"]
    assert image_part == {"type": "image_url", "image_url": {"url": f"file://{png}"}}
    assert text_part == {"type": "text", "text": "Text Recognition:"}

    assert result["text"] == "INVOICE 2026"
    assert result["confidence"] is None
    assert result["language"] is None
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 3
    assert result["total_tokens"] == 13


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("text", "Text Recognition:"),
        ("formula", "Formula Recognition:"),
        ("table", "Table Recognition:"),
        ("Table Recognition", "Table Recognition:"),
        ("  FORMULA recognition ", "Formula Recognition:"),
        ("ocr", "Text Recognition:"),
        ("plain text", "Text Recognition:"),
        ("unknown-task", "Text Recognition:"),
        ("", "Text Recognition:"),
    ],
)
def test_task_prompts_are_preserved(tmp_path, task, expected):
    vlm = _FakeVLM()
    asyncio.run(_engine(vlm).extract_text(_png(tmp_path), task=task))
    text = vlm.calls[0]["messages"][0]["content"][1]["text"]
    assert text == expected


def test_language_hint_is_appended_to_task_prompt(tmp_path):
    vlm = _FakeVLM()
    result = asyncio.run(
        _engine(vlm).extract_text(_png(tmp_path), language="Japanese", task="formula")
    )
    text = vlm.calls[0]["messages"][0]["content"][1]["text"]
    assert text == "Formula Recognition:\n\nRespond in Japanese."
    assert result["language"] == "Japanese"


def test_cancel_event_propagates_to_generate(tmp_path):
    vlm = _FakeVLM()
    ev = threading.Event()
    asyncio.run(_engine(vlm).extract_text(_png(tmp_path), cancel_event=ev))
    assert vlm.calls[0]["cancel_event"] is ev


def test_extract_text_requires_started_engine(tmp_path):
    from yunshu_engine.ocr_engine import OCREngine

    with pytest.raises(RuntimeError, match="not started"):
        asyncio.run(OCREngine("/models/x").extract_text(_png(tmp_path)))


def test_start_loads_through_vlm_engine_not_mlx_vlm_load(monkeypatch):
    from yunshu_engine import ocr_engine as ocr_mod
    from yunshu_engine import vlm_engine as vlm_mod
    from yunshu_engine.ocr_engine import OCREngine

    built: list[object] = []

    class _RecordingVLM(_FakeVLM):
        def __init__(self, model_path, config=None):
            super().__init__()
            self.model_path = model_path
            self.started = False
            built.append(self)

        async def start(self):
            self.started = True

    def _no_stock_load(self):
        raise AssertionError("OCREngine must not call mlx_vlm.load itself")

    monkeypatch.setattr(vlm_mod, "VLMEngine", _RecordingVLM)
    monkeypatch.setattr(OCREngine, "load", _no_stock_load)
    monkeypatch.setattr(
        ocr_mod.OCREngine, "_import_glm_ocr_processing", staticmethod(lambda: None)
    )

    engine = OCREngine("/models/GLM-OCR-bf16")

    async def _go():
        await engine.start()
        try:
            return engine.is_loaded, engine.get_stats()
        finally:
            await engine.stop()

    loaded, stats = asyncio.run(_go())
    assert loaded is True
    assert stats["loaded"] is True and stats["running"] is True
    assert len(built) == 1 and built[0].started
    assert built[0].model_path == "/models/GLM-OCR-bf16"
    assert built[0].stopped is True  # the finally block stopped the VLMEngine
    assert engine.is_loaded is False


def test_start_failure_leaves_engine_unloaded(monkeypatch):
    from yunshu_engine import vlm_engine as vlm_mod
    from yunshu_engine.ocr_engine import OCREngine

    class _BrokenVLM(_FakeVLM):
        def __init__(self, model_path, config=None):
            super().__init__()

        async def start(self):
            raise RuntimeError("weights missing")

    monkeypatch.setattr(vlm_mod, "VLMEngine", _BrokenVLM)
    monkeypatch.setattr(
        OCREngine, "_import_glm_ocr_processing", staticmethod(lambda: None)
    )
    engine = OCREngine("/models/GLM-OCR-bf16")
    asyncio.run(engine.start())
    assert engine.is_loaded is False
    assert engine.get_stats()["running"] is False


def test_stop_stops_the_vlm_engine(tmp_path):
    vlm = _FakeVLM()
    engine = _engine(vlm)
    asyncio.run(engine.stop())
    assert vlm.stopped is True
    assert engine.is_loaded is False


def test_image_token_estimate_reads_vlm_vision_config(tmp_path):
    # 280x280 with patch 14 and spatial merge 2: (20*20) // (2*2) = 100
    png = _png(tmp_path, size=(280, 280))
    vlm = _FakeVLM(
        config={"vision_config": {"patch_size": 14, "spatial_merge_size": 2}}
    )
    assert _engine(vlm)._estimate_image_tokens(png) == 100


def test_extract_text_counts_as_an_active_request(tmp_path):
    """@tracks_active is kept: the model manager sees the OCR call as in flight."""
    vlm = _FakeVLM()
    engine = _engine(vlm)
    seen: list[bool] = []
    original = vlm.generate

    async def _probe(**kwargs):
        seen.append(engine.has_active_requests())
        return await original(**kwargs)

    vlm.generate = _probe
    asyncio.run(engine.extract_text(_png(tmp_path)))
    assert seen == [True]
    assert engine.has_active_requests() is False


def test_stop_clears_the_mlx_buffer_pool_on_the_executor(monkeypatch):
    """stop() still releases the Metal buffer pool (the old standalone path's OOM fix)."""
    from yunshu_engine import mlx_executor as ex_mod

    cleared: list[str] = []
    monkeypatch.setattr(ex_mod, "sync_and_clear_cache", lambda: cleared.append("clear"))
    engine = _engine(_FakeVLM())

    engine._executor = ex_mod.get_mlx_executor()
    asyncio.run(engine.stop())
    assert cleared == ["clear"]


def test_glm_ocr_processing_is_imported_before_the_processor_loads():
    """The GLM-OCR processing module installs the AutoProcessor patch; start() imports it."""
    import sys

    pytest.importorskip("mlx_vlm.models.glm_ocr.processing")
    from yunshu_engine.ocr_engine import OCREngine

    sys.modules.pop("mlx_vlm.models.glm_ocr.processing", None)
    OCREngine._import_glm_ocr_processing()
    assert "mlx_vlm.models.glm_ocr.processing" in sys.modules


def _count_vlm_engines(monkeypatch, fake_cls):
    from yunshu_engine import vlm_engine as vlm_mod

    monkeypatch.setattr(vlm_mod, "VLMEngine", fake_cls)
    return fake_cls


def test_start_builds_and_starts_exactly_one_vlm_engine(monkeypatch):
    from yunshu_engine.ocr_engine import OCREngine

    counts = {"init": 0, "start": 0}

    class _Counting(_FakeVLM):
        def __init__(self, model_path, config=None):
            super().__init__()
            counts["init"] += 1

        async def start(self):
            counts["start"] += 1

    _count_vlm_engines(monkeypatch, _Counting)
    monkeypatch.setattr(
        OCREngine, "_import_glm_ocr_processing", staticmethod(lambda: None)
    )
    engine = OCREngine("/models/GLM-OCR-bf16")
    asyncio.run(engine.start())
    assert counts == {"init": 1, "start": 1}
    assert engine._owns_vlm is True


def test_start_reuses_a_loaded_vlm_without_a_second_load(monkeypatch, tmp_path):
    from yunshu_engine.ocr_engine import OCREngine

    class _MustNotBuild:
        def __init__(self, *a, **k):
            raise AssertionError(
                "a second VLMEngine must not be built for a shared checkpoint"
            )

    _count_vlm_engines(monkeypatch, _MustNotBuild)
    monkeypatch.setattr(
        OCREngine, "_import_glm_ocr_processing", staticmethod(lambda: None)
    )
    shared = _FakeVLM()
    engine = OCREngine("/models/GLM-OCR-bf16", shared_vlm=shared)

    async def _go():
        await engine.start()
        out = await engine.extract_text(_png(tmp_path))
        await engine.stop()
        return out

    out = asyncio.run(_go())
    assert out["text"] == "INVOICE 2026"
    assert len(shared.calls) == 1  # OCR went through the shared engine
    assert shared.stopped is False  # stop() leaves the owner's engine running
    assert engine._owns_vlm is False


def test_find_loaded_vlm_matches_only_a_loaded_engine_for_the_same_checkpoint():
    from yunshu_engine.model_manager import find_loaded_vlm
    from yunshu_engine.vlm_engine import VLMEngine

    loaded = VLMEngine("/models/Qwen3.5-9B")
    loaded._model = object()
    unloaded = VLMEngine("/models/GLM-OCR-bf16")
    entries = [
        types.SimpleNamespace(model_path="/models/Qwen3.5-9B", engine=loaded),
        types.SimpleNamespace(model_path="/models/GLM-OCR-bf16", engine=unloaded),
    ]
    assert find_loaded_vlm(entries, "/models/Qwen3.5-9B") is loaded
    assert find_loaded_vlm(entries, "/models/GLM-OCR-bf16") is None  # not loaded
    assert find_loaded_vlm(entries, "/models/other") is None


def test_instantiate_ocr_engine_reuses_the_shared_vlm(monkeypatch):
    from yunshu_engine import vlm_engine as vlm_mod
    from yunshu_engine.model_manager import ModelType, instantiate_engine

    class _MustNotBuild:
        def __init__(self, *a, **k):
            raise AssertionError("no second VLMEngine")

    monkeypatch.setattr(vlm_mod, "VLMEngine", _MustNotBuild)
    from yunshu_engine.ocr_engine import OCREngine

    monkeypatch.setattr(
        OCREngine, "_import_glm_ocr_processing", staticmethod(lambda: None)
    )
    shared = _FakeVLM()
    engine = asyncio.run(
        instantiate_engine(ModelType.OCR, "/models/GLM-OCR-bf16", shared_vlm=shared)
    )
    assert engine._vlm is shared and engine.is_loaded
