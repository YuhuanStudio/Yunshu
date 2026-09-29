"""Revisit a long document after RAM eviction: APC disk tier vs cold re-prefill.

Sends documents A, B, C (each ~--lines long, distinct content) and then A again
to a running Yunshu server. With a RAM APC budget smaller than three documents,
A is evicted before the revisit; with ``YUNSHU_VLM_APC_DISK_DIR`` set it should
come back from disk instead of a full prefill. Records first-token time,
cached tokens and answer correctness for every request.

    python scripts/research/probe_apc_disk_tier.py --url http://127.0.0.1:18764 \
        --output runs/apc-disk.jsonl --label disk-6g
"""

import argparse
import http.client
import json
import time
import urllib.parse
from pathlib import Path


def doc(tag, lines):
    return "".join(
        f"Ledger {tag}-{i}: shipment {i * 7 % 997} cleared customs at dock {i % 23}.\n"
        for i in range(lines)
    )


def ask(url, content, max_tokens=12):
    u = urllib.parse.urlparse(url)
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
    t0 = time.perf_counter()
    first, text, usage = None, "", None
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    for line in conn.getresponse():
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                first = first or time.perf_counter() - t0
                text += piece
    conn.close()
    return {
        "first_s": round(first or 0, 3),
        "wall_s": round(time.perf_counter() - t0, 3),
        "text": text.strip(),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--lines", type=int, default=1400)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    codes = {"A": "ORCHID", "B": "COBALT", "C": "LILAC"}
    plan = ["A", "B", "C", "A"]
    with a.output.open("a") as f:
        for step, tag in enumerate(plan, 1):
            r = ask(
                a.url,
                doc(tag, a.lines)
                + f"\nThe vault code for ledger {tag} is {codes[tag]}. "
                "What is the vault code? Reply with the code only.",
            )
            row = {
                "label": a.label,
                "step": step,
                "doc": tag,
                "ok": codes[tag] in r["text"].upper(),
                **r,
            }
            f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
