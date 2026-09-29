"""Multi-turn check of Yunshu's /v1/omni/speech/stream (text/speech in, speech out).

Sends alternating text and spoken turns to one running server, records first text
delta, first audio chunk, total time, decoded audio seconds and the text, and
flags turn-2+ corruption ("!!!!") or empty audio. Spoken turns use local WAV
files (e.g. generated with macOS ``say``).

    python scripts/research/probe_omni_http.py --url http://127.0.0.1:18766 \
        --audio q1.wav q2.wav --output runs/omni-http.jsonl
"""

import argparse
import base64
import http.client
import json
import time
import urllib.parse
from pathlib import Path

TEXT_TURNS = [
    ("Reply with exactly one word: ALPHA", "alpha"),
    ("What is 2 + 3? Answer with just the number.", "5"),
    ("Name the capital of France in one word.", "paris"),
]


def turn(url, payload, timeout=600):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    t0 = time.perf_counter()
    out = {
        "text": "",
        "first_text_s": None,
        "first_audio_s": None,
        "audio_samples": 0,
        "done": None,
    }
    try:
        conn.request(
            "POST",
            "/v1/omni/speech/stream",
            json.dumps(payload),
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        out["status"] = resp.status
        if resp.status != 200:
            out["error"] = resp.read().decode(errors="replace")[:300]
            return out
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            now = time.perf_counter() - t0
            if ev.get("type") == "text":
                out["first_text_s"] = out["first_text_s"] or round(now, 3)
                out["text"] += ev.get("delta", "")
            elif ev.get("type") == "audio":
                out["first_audio_s"] = out["first_audio_s"] or round(now, 3)
                out["audio_samples"] += len(base64.b64decode(ev["delta"])) // 2
                out["sr"] = ev.get("sr", 24000)
            elif ev.get("type") == "done":
                out["done"] = {k: v for k, v in ev.items() if k != "type"}
            elif ev.get("type") == "error":
                out["error"] = ev.get("message")
    finally:
        conn.close()
    out["total_s"] = round(time.perf_counter() - t0, 3)
    out["audio_s"] = round(out["audio_samples"] / out.get("sr", 24000), 2)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.output.open("a") as f:
        for _ in range(a.rounds):
            for prompt, expect in TEXT_TURNS:
                r = turn(a.url, {"text": prompt})
                n += 1
                row = {
                    "turn": n,
                    "input": "text",
                    "prompt": prompt,
                    "expect": expect,
                    "ok": expect in r["text"].lower()
                    and r["audio_samples"] > 0
                    and not r["text"].strip().startswith("!!"),
                    **r,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "turn",
                                "input",
                                "ok",
                                "text",
                                "first_text_s",
                                "first_audio_s",
                                "total_s",
                                "audio_s",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            for path in a.audio:
                r = turn(
                    a.url,
                    {
                        "text": "Answer the spoken question in a few words.",
                        "audio_path": str(Path(path).resolve()),
                    },
                )
                n += 1
                row = {
                    "turn": n,
                    "input": f"audio:{Path(path).name}",
                    "ok": r.get("status") == 200
                    and r["audio_samples"] > 0
                    and bool(r["text"].strip())
                    and not r["text"].strip().startswith("!!"),
                    **r,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "turn",
                                "input",
                                "ok",
                                "text",
                                "first_text_s",
                                "first_audio_s",
                                "total_s",
                                "audio_s",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
