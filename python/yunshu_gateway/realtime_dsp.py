"""Streaming input-audio noise reduction for the Realtime ``noise_reduction`` setting.

OpenAI applies noise reduction to the input audio before VAD and the model
(``audio.input.noise_reduction: {"type": "near_field" | "far_field"}``). This is a small
CPU implementation with the same contract: it runs on every appended PCM16 chunk, keeps
per-session state, and never changes the length of the audio.

Two stages, both cheap and causal:

* a one-pole high-pass filter (removes rumble / DC / mains hum below the speech band);
* a downward expander driven by a tracked noise floor: 10 ms frames whose level is close to
  the estimated floor are attenuated (smoothly, so speech onsets are not clipped), frames
  well above it pass unchanged.

``near_field`` (headset / close mic) keeps the expander mild; ``far_field`` (laptop / room
mic) uses a higher cut-off and a deeper expander.
"""

from __future__ import annotations

import numpy as np

# type -> (high-pass cut-off Hz, expander depth (max attenuation, linear), margin over floor)
_PROFILES: dict[str, tuple[float, float, float]] = {
    "near_field": (80.0, 0.35, 1.8),
    "far_field": (120.0, 0.12, 2.5),
}
_FRAME_MS = 10
_FLOOR_RISE = 1.002  # per frame: the floor estimate creeps up slowly ...
_FLOOR_MIN = 4.0  # ... and never below a few LSBs of PCM16


def supported(kind: object) -> bool:
    return kind in _PROFILES


class NoiseReducer:
    """Causal, stateful denoiser for mono PCM16 at a fixed sample rate."""

    def __init__(self, kind: str, sample_rate: int = 24000) -> None:
        if kind not in _PROFILES:
            raise ValueError(f"unknown noise_reduction type {kind!r}")
        cutoff, depth, margin = _PROFILES[kind]
        self.kind = kind
        self.rate = int(sample_rate)
        self._depth = depth
        self._margin = margin
        # one-pole high-pass y[n] = a * (y[n-1] + x[n] - x[n-1])
        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / self.rate
        self._a = rc / (rc + dt)
        self._x1 = 0.0
        self._y1 = 0.0
        self._floor = 0.0
        self._gain = 1.0
        self._frame = max(1, self.rate * _FRAME_MS // 1000)

    def _highpass(self, x: np.ndarray) -> np.ndarray:
        # y[n] - a*y[n-1] = a*(x[n] - x[n-1]): a first-order IIR, run per chunk
        d = np.empty_like(x)
        d[0] = x[0] - self._x1
        d[1:] = x[1:] - x[:-1]
        d *= self._a
        y = np.empty_like(x)
        prev = self._y1
        a = self._a
        for i in range(len(x)):  # tiny loop body; chunks are ~20-100 ms
            prev = a * prev + d[i]
            y[i] = prev
        self._x1 = float(x[-1])
        self._y1 = float(prev)
        return y

    def process(self, pcm16: bytes) -> bytes:
        """Denoise one chunk of little-endian PCM16; same length out as in."""
        n = len(pcm16) // 2
        if n == 0:
            return pcm16
        x = np.frombuffer(pcm16, dtype="<i2", count=n).astype(np.float32)
        y = self._highpass(x)
        frame = self._frame
        out = np.empty_like(y)
        for pos in range(0, n, frame):
            seg = y[pos : pos + frame]
            level = float(np.sqrt(np.mean(seg * seg)))
            if len(seg) == frame:  # the floor only learns from whole frames
                if self._floor == 0.0:
                    self._floor = max(level, _FLOOR_MIN)
                elif level < self._floor:
                    self._floor = max(0.9 * self._floor + 0.1 * level, _FLOOR_MIN)
                else:
                    self._floor = max(self._floor * _FLOOR_RISE, _FLOOR_MIN)
            ratio = level / (max(self._floor, _FLOOR_MIN) * self._margin)
            target = 1.0 if ratio >= 1.0 else max(self._depth, ratio**2)
            # fast attack (do not clip speech starts), slower release (no pumping)
            coef = 0.6 if target > self._gain else 0.15
            g0 = self._gain
            g1 = g0 + coef * (target - g0)
            out[pos : pos + len(seg)] = seg * np.linspace(
                g0, g1, len(seg), dtype=np.float32
            )
            self._gain = g1
        return np.clip(out, -32768, 32767).astype("<i2").tobytes()
