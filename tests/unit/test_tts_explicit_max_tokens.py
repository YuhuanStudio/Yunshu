"""TTS passes max_tokens explicitly when the model leaves it open.

mlx-audio <= 0.5 gives Qwen3-TTS generate() max_tokens=4096; 0.6+ defaults it to None ->
generation_config.json, else 2048, which truncates long speech. The engine passes the
effective value (4096) itself unless the checkpoint sets one, and never overrides a model
whose own default is an int.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from yunshu_engine.audio_engine import TTSEngine


def _engine(tmp_path, generate, cfg=None):
    path = tmp_path / "tts"
    path.mkdir(exist_ok=True)
    if cfg is not None:
        (path / "generation_config.json").write_text(json.dumps(cfg))
    engine = TTSEngine(str(path))
    model = SimpleNamespace(generate=generate, sample_rate=16000)
    engine._model = model
    return engine


def _gen(received, default):
    def generate(text, max_tokens=default, verbose=False):
        received["max_tokens"] = max_tokens
        r = MagicMock()
        r.audio = np.zeros(1600)
        return [r]

    return generate


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_open_max_tokens_gets_4096(tmp_path):
    got: dict = {}
    _run(_engine(tmp_path, _gen(got, None)).synthesize("hi"))
    assert got["max_tokens"] == 4096


def test_stream_open_max_tokens_gets_4096(tmp_path):
    got: dict = {}
    engine = _engine(tmp_path, _gen(got, None))

    async def go():
        async for _ in engine.synthesize_stream("hi"):
            pass

    _run(go())
    assert got["max_tokens"] == 4096


def test_checkpoint_setting_is_left_to_the_model(tmp_path):
    got: dict = {}
    _run(_engine(tmp_path, _gen(got, None), {"max_new_tokens": 1500}).synthesize("hi"))
    assert got["max_tokens"] is None


@pytest.mark.parametrize("default", [1200, 4096])
def test_models_with_their_own_int_default_are_untouched(tmp_path, default):
    got: dict = {}
    _run(_engine(tmp_path, _gen(got, default)).synthesize("hi"))
    assert got["max_tokens"] == default


def test_a_caller_max_tokens_wins(tmp_path):
    got: dict = {}
    _run(_engine(tmp_path, _gen(got, None)).synthesize("hi", max_tokens=300))
    assert got["max_tokens"] == 300
