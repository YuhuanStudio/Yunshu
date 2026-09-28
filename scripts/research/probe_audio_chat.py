"""Spoken questions through /v1/chat/completions (input_audio), stream and not.

    python scripts/research/probe_audio_chat.py --url http://127.0.0.1:18764 \
        --audio q1.wav q2.wav --output runs/audio-chat.jsonl
"""

import argparse
import base64
import http.client
import json
import time
import urllib.parse
from pathlib import Path


def ask(url, wav, stream):
    u = urllib.parse.urlparse(url)
    data = base64.b64encode(Path(wav).read_bytes()).decode()
    body = {
        "model": "x",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data, "format": "wav"},
                    },
                    {
                        "type": "text",
                        "text": "Answer the spoken question in a few words.",
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": stream,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if not stream:
        raw = resp.read().decode(errors="replace")
        conn.close()
        try:
            text = json.loads(raw)["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            text = raw[:300]
        return {
            "status": resp.status,
            "text": text,
            "wall_s": round(time.perf_counter() - t0, 2),
        }
    text = ""
    for line in resp:
        if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
            for ch in json.loads(line[6:]).get("choices") or []:
                text += (ch.get("delta") or {}).get("content") or ""
    conn.close()
    return {
        "status": resp.status,
        "text": text,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        for wav in a.audio:
            for stream in (False, True):
                row = {
                    "label": a.label,
                    "audio": Path(wav).name,
                    "stream": stream,
                    **ask(a.url, wav, stream),
                }
                row["ok"] = row["status"] == 200 and bool(row["text"].strip())
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
