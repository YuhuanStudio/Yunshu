"""Realtime voice-agent client — speak in, hear the model speak back.

Connects to Yunshu's OpenAI-Realtime WebSocket and runs ONE native
speech-to-speech turn: it streams a WAV of your spoken question to the
server, the unified Qwen3-Omni model (Thinker+Talker) answers, and this
client assembles the streamed audio into `reply.wav`.

This uses manual turn control (turn_detection disabled) so it's fully
deterministic — no server-side VAD timing. For a live mic + barge-in, keep
turn_detection on "server_vad" and stream input_audio_buffer.append frames.

Prereq — a Yunshu server with the native-omni realtime path enabled:
    YUNSHU_OMNI_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct-4bit \
    YUNSHU_REALTIME_OMNI=1 \
    yunshu serve -m /path/to/any-small-text-model --port 8000

Run:
    pip install websockets
    python examples/realtime_voice.py question.wav            # ws://localhost:8000
    python examples/realtime_voice.py question.wav ws://host:8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import wave

import websockets

OMNI_RATE = 24000  # Qwen3-Omni Talker output; pcm16 input is also treated as 24 kHz


def _read_wav_as_pcm16_24k(path: str) -> bytes:
    """Load a mono WAV and return raw PCM16 @ 24 kHz (nearest-sample resample)."""
    with wave.open(path, "rb") as wf:
        n, sw, ch, rate = (
            wf.getnframes(),
            wf.getsampwidth(),
            wf.getnchannels(),
            wf.getframerate(),
        )
        raw = wf.readframes(n)
    if sw != 2:
        raise SystemExit(f"{path}: need 16-bit PCM WAV (got sampwidth={sw})")
    import array

    samples = array.array("h", raw)
    if ch == 2:  # downmix stereo → mono
        samples = array.array(
            "h", [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
        )
    if rate != OMNI_RATE:  # crude nearest-sample resample (examples don't ship scipy)
        ratio = OMNI_RATE / rate
        out = array.array(
            "h",
            [
                samples[min(int(i / ratio), len(samples) - 1)]
                for i in range(int(len(samples) * ratio))
            ],
        )
        samples = out
    return samples.tobytes()


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python examples/realtime_voice.py question.wav [ws://host:port]"
        )
    wav_path = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000"
    url = base.rstrip("/") + "/v1/realtime"

    pcm_in = _read_wav_as_pcm16_24k(wav_path)
    print(f"→ {url}  ({len(pcm_in) // 2 / OMNI_RATE:.2f}s spoken input)")

    async with websockets.connect(url, max_size=None) as ws:

        async def send(ev: dict) -> None:
            await ws.send(json.dumps(ev))

        # Configure: text + audio out, manual turns (no VAD), pcm16 both ways.
        await send(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": None,
                },
            }
        )
        # Stream the spoken turn, commit it, ask for a reply.
        await send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_in).decode(),
            }
        )
        await send({"type": "input_audio_buffer.commit"})
        await send({"type": "response.create"})

        out = bytearray()
        transcript = ""
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            # With audio requested, the model emits its spoken text on BOTH
            # response.text.delta AND response.audio_transcript.delta — read only
            # the transcript channel so we don't double-count the words.
            if t == "response.audio_transcript.delta":
                transcript += ev.get("delta", "")
            elif t == "response.audio.delta":
                out += base64.b64decode(ev["delta"])
            elif t == "error":
                print("server error:", ev.get("error"))
            elif t == "response.done":
                break

    with wave.open("reply.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OMNI_RATE)
        wf.writeframes(bytes(out))
    print(f"← transcript: {transcript!r}")
    print(f"← wrote reply.wav ({len(out) // 2 / OMNI_RATE:.2f}s of audio)")


if __name__ == "__main__":
    asyncio.run(main())
