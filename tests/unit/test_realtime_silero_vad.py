"""Optional Silero neural VAD in the Realtime router — gating + decision path.

Uses a FAKE Silero model (monkeypatched) so CI never downloads/loads weights.
The real model is validated separately; here we cover the integration contract:
the gate, the streaming-window scoring, and that _run_vad uses the probability.
"""
from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import yunshu_gateway.routers.realtime as rt


@pytest.fixture(autouse=True)
def _reset_silero_singleton():
    rt._silero_vad = None
    yield
    rt._silero_vad = None


class _FakeBranchCfg:
    chunk_size = 512


class _FakeBranch:
    config = _FakeBranchCfg()


class _FakeSilero:
    """Returns a fixed speech probability per fed window; counts feeds."""

    def __init__(self, prob: float):
        self.prob = prob
        self.feeds = 0

    def _branch(self, rate):
        return _FakeBranch()

    def feed(self, samples, state, rate):
        self.feeds += 1
        return np.array([self.prob]), (state or 0) + 1


def _session():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    s = rt.RealtimeSession(ws)
    s.session.turn_detection = {"type": "server_vad", "threshold": 0.5,
                                "silence_duration_ms": 500}
    return s, ws


def _pcm24k(n, amp=0.3):
    return struct.pack(f"<{n}h", *([int(amp * 32767)] * n))


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("YUNSHU_REALTIME_VAD", raising=False)
    assert rt._silero_vad_enabled() is False
    assert rt._get_silero_vad() is None
    monkeypatch.setenv("YUNSHU_REALTIME_VAD", "silero")
    assert rt._silero_vad_enabled() is True


def test_silero_speech_prob_windows_and_carries_leftover(monkeypatch):
    fake = _FakeSilero(prob=0.9)
    monkeypatch.setattr(rt, "_get_silero_vad", lambda: fake)
    s, _ = _session()
    # 1s @24k → ~16k samples @16k → ~31 windows of 512
    prob = s._silero_speech_prob(_pcm24k(24000))
    assert prob == 0.9
    assert fake.feeds >= 30          # scored many 32ms windows
    assert s._silero_leftover is not None  # remainder carried for continuity


@pytest.mark.asyncio
async def test_run_vad_uses_silero_probability(monkeypatch):
    monkeypatch.setattr(rt, "_get_silero_vad", lambda: _FakeSilero(prob=0.95))
    s, ws = _session()
    await s._run_vad(_pcm24k(24000))  # high prob → speech
    assert s._vad_speaking is True
    types = [c[0][0]["type"] for c in ws.send_json.call_args_list]
    assert "input_audio_buffer.speech_started" in types


@pytest.mark.asyncio
async def test_run_vad_silence_via_silero(monkeypatch):
    monkeypatch.setattr(rt, "_get_silero_vad", lambda: _FakeSilero(prob=0.02))
    s, ws = _session()
    await s._run_vad(_pcm24k(24000))  # low prob → silence, never speaks
    assert s._vad_speaking is False
    types = [c[0][0]["type"] for c in ws.send_json.call_args_list]
    assert "input_audio_buffer.speech_started" not in types


@pytest.mark.asyncio
async def test_energy_fallback_when_silero_disabled(monkeypatch):
    # No Silero → energy heuristic still drives the decision (loud chunk = speech).
    monkeypatch.setattr(rt, "_get_silero_vad", lambda: None)
    s, ws = _session()
    await s._run_vad(_pcm24k(24000, amp=0.5))  # RMS well above threshold*0.05
    assert s._vad_speaking is True


def test_reset_silero_clears_state(monkeypatch):
    monkeypatch.setattr(rt, "_get_silero_vad", lambda: _FakeSilero(prob=0.9))
    s, _ = _session()
    s._silero_speech_prob(_pcm24k(24000))
    assert s._silero_state is not None
    s._reset_silero()
    assert s._silero_state is None and s._silero_leftover is None
