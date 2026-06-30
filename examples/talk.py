"""talk.py — have a real spoken conversation with Yunshu, live.

Speak into your microphone, hear the model speak back through your speakers,
in a back-and-forth loop. This is the flagship: native **speech-to-speech**
(Qwen3-Omni Thinker+Talker) over Yunshu's OpenAI-Realtime WebSocket. Your voice
goes in as raw audio (no speech-to-text step), the model thinks and speaks its
reply directly (no text-to-speech step) — first audio comes back in ~1.4 s.

The conversation keeps its history, so you can refer back to earlier turns
("what did I just say?").

─────────────────────────────────────────────────────────────────────────────
1. Start a server with a Qwen3-Omni model — native voice is on automatically
   (the same loaded model serves both text and speech, no extra memory):

       uv run yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000

2. Install this client's deps and run it (sounddevice bundles PortAudio on macOS):

       pip install sounddevice numpy websockets
       python examples/talk.py                      # connects to ws://localhost:8000
       python examples/talk.py ws://other-host:8000

3. Each turn: press Enter, speak, press Enter again to send. The model answers
   out loud. Press Ctrl-C to quit.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import sys
import threading

SR = 24000  # Qwen3-Omni speaks (and we record) at 24 kHz mono PCM16

try:
    import numpy as np
    import sounddevice as sd
    import websockets
except ImportError as e:  # pragma: no cover - friendly first-run message
    raise SystemExit(
        f"missing dependency ({e.name}). Run:  pip install sounddevice numpy websockets"
    ) from e


def record_utterance() -> bytes:
    """Record from the mic until the user presses Enter. Returns PCM16 bytes."""
    input("\n🎤  Press Enter, speak, then press Enter again to send…")
    frames: list[np.ndarray] = []
    stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="int16",
        callback=lambda indata, *_: frames.append(indata.copy()),
    )
    with stream:
        input("    ● recording — press Enter to send  ")
    if not frames:
        return b""
    return np.concatenate(frames, axis=0).tobytes()


def start_player() -> tuple[queue.Queue, threading.Thread]:
    """Background player: streams PCM chunks to the speakers as they arrive."""
    q: queue.Queue = queue.Queue()

    def run() -> None:
        with sd.RawOutputStream(samplerate=SR, channels=1, dtype="int16") as out:
            while True:
                chunk = q.get()
                if chunk is None:
                    return
                out.write(chunk)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return q, t


async def take_turn(ws, pcm: bytes) -> None:
    """Send one spoken turn and play the model's spoken reply as it streams."""
    await ws.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )
    )
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    await ws.send(json.dumps({"type": "response.create"}))

    audio_q, player = start_player()
    transcript = ""
    print("🤖  ", end="", flush=True)
    async for raw in ws:
        ev = json.loads(raw)
        kind = ev.get("type", "")
        if kind == "response.audio.delta":
            audio_q.put(base64.b64decode(ev["delta"]))
        elif kind == "response.audio_transcript.delta":
            # The model's spoken words, as text — print them so you can read along.
            delta = ev.get("delta", "")
            transcript += delta
            print(delta, end="", flush=True)
        elif kind == "error":
            err = ev.get("error") or {}
            if err.get("code") == "no_asr_engine":
                raise SystemExit(
                    "\n✗ The server isn't running the native voice path — it tried to "
                    "transcribe your audio and has no ASR model.\n\n"
                    "  Serve a Qwen3-Omni model (native voice is automatic):\n\n"
                    "      uv run yunshu serve -m <your-omni-model> --port 8000\n\n"
                    "  (if you set YUNSHU_REALTIME_OMNI=0, drop it.)"
                )
            print(f"\n    server error: {err}")
            break
        elif kind == "response.done":
            break
    print()
    audio_q.put(None)  # signal end-of-turn
    await asyncio.to_thread(player.join)  # wait until playback finishes


async def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000"
    url = base.rstrip("/") + "/v1/realtime"

    try:
        ws = await websockets.connect(url, max_size=None)
    except OSError as e:
        raise SystemExit(
            f"could not connect to {url} — is the server running? ({e})"
        ) from None

    async with ws:
        # Manual turns (no server-side VAD) keep this simple and deterministic;
        # pcm16 both ways matches the Talker's native rate.
        await ws.send(
            json.dumps(
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
        )
        print(f"Connected to {url}. Talk to it — Ctrl-C to quit.")
        while True:
            pcm = await asyncio.to_thread(record_utterance)
            if not pcm:
                print("    (nothing recorded — try again)")
                continue
            secs = len(pcm) // 2 / SR
            print(f"    → sent {secs:.1f}s of audio")
            await take_turn(ws, pcm)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye 👋")
