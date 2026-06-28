"""Waves 950 + 952: gateway shutdown-drain reliability + ASR VAD short-clip gate.

W950 (HIGH): the shutdown drain awaited a one-shot asyncio.Event that, once .set() (the
  active-request count touched 0 at ANY point in the server's life), stayed set forever — so
  _drain_event.wait() returned IMMEDIATELY even with a request in-flight at shutdown, letting
  engine.stop() tear the model out from under a running generation. Poll the LIVE counter.
W952: the ASR VAD pre-gate required ~3 consecutive loud frames (streaming turn-detection
  latch), so a short utterance (<~90ms) never latched → the file returned an empty transcript
  on real speech. For a one-shot file gate, any single clearly-loud frame proceeds.
"""
from __future__ import annotations

import inspect


def test_w950_shutdown_polls_live_counter():
    from yunshu_gateway import main
    src = inspect.getsource(main)
    # the one-shot event wait (the actual await call) is gone; a live-counter poll loop is used
    assert "await asyncio.wait_for(_drain_event.wait()" not in src
    assert "while _active_requests > 0 and time.monotonic() < _drain_deadline:" in src


def test_w952_vad_energy_fallback_in_source():
    from yunshu_engine import audio_engine
    src = inspect.getsource(audio_engine.ASREngine.transcribe)
    assert "vad_result.energy > self._vad.threshold" in src


def test_w952_short_loud_frame_has_energy_above_threshold():
    # a single loud frame must register energy above the base threshold (so the W952 gate
    # proceeds even though the 3-frame latch hasn't engaged).
    import numpy as np

    from yunshu_engine.vad import EnergyVAD
    vad = EnergyVAD()
    sr = vad.sample_rate
    n = int(sr * vad.frame_duration_ms / 1000)
    loud = (np.ones(n, dtype=np.float32) * 0.5 * 32767).astype(np.int16).tobytes()
    r = vad.process_frame(loud, sample_rate=sr)
    # one frame → not yet latched as is_speech, but energy is well above the base threshold
    assert r.energy > vad.threshold
    # and a silent frame stays below
    quiet = (np.zeros(n, dtype=np.float32)).astype(np.int16).tobytes()
    assert vad.process_frame(quiet, sample_rate=sr).energy <= vad.threshold
