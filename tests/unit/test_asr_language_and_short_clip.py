"""ASR: detected Japanese/Korean is not relabeled Chinese, and speech confined to a
sub-frame or final partial VAD frame still reaches the model."""

from __future__ import annotations

import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from yunshu_engine import audio_engine
from yunshu_engine.vad import EnergyVAD


class _Model:
    def __init__(self, text, lang):
        self.text, self.lang, self.calls = text, lang, 0

    def generate(self, path, **kw):
        self.calls += 1
        return SimpleNamespace(
            text=self.text,
            language=self.lang,
            segments=[{"text": self.text, "language": self.lang}],
            total_time=0.1,
        )


def _engine(model):
    eng = audio_engine.ASREngine.__new__(audio_engine.ASREngine)
    eng._model = model
    eng._vad = EnergyVAD()
    eng._executor = ThreadPoolExecutor(max_workers=1)
    eng._stats_lock = threading.Lock()
    eng._transcribe_count = 0
    eng._total_transcribe_ms = 0.0
    return eng


def _wav(path, silent, loud):
    pcm = np.concatenate([np.zeros(silent), np.full(loud, 0.5 * 32767)]).astype(
        np.int16
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


@pytest.mark.asyncio
async def test_japanese_not_relabeled_zh(tmp_path):
    p = tmp_path / "a.wav"
    _wav(p, 0, 16000)
    eng = _engine(_Model("東京都の天気予報です", "ja"))
    out = await eng.transcribe(str(p))
    eng._executor.shutdown(wait=True)
    assert out["language"] == "ja"
    assert out["segments"][0]["language"] == "ja"


@pytest.mark.asyncio
@pytest.mark.parametrize("silent,loud", [(0, 320), (480, 320)])
async def test_partial_frame_speech_reaches_asr(tmp_path, silent, loud):
    p = tmp_path / "a.wav"
    _wav(p, silent, loud)
    model = _Model("hi", "en")
    eng = _engine(model)
    out = await eng.transcribe(str(p))
    eng._executor.shutdown(wait=True)
    assert model.calls == 1
    assert out["text"] == "hi"
