"""Repeated-document TTFT over HTTP: the same long prompt three times (cold,
then cached), then the document with a different question (prefix reuse).
Temperature 0, so the text of the repeats must equal the cold run's.

    python scripts/research/probe_repeat_doc.py --url http://127.0.0.1:18990 \
        --model Qwen3.8-27B --tokenizer <ckpt dir> --tokens 8192
"""

import argparse
import http.client
import json
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_context_batch import CORPUS, INSTRUCTION, make_prompt  # noqa: E402


def ask(url, model, prompt, max_tokens=48):
    u = urllib.parse.urlparse(url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    first, text, usage = None, "", {}
    for line in resp:
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or ""
            if piece and first is None:
                first = time.perf_counter() - t0
            text += piece
    conn.close()
    return {
        "ttft_s": round(first, 3) if first else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    corpus = CORPUS.read_text()
    doc = make_prompt(tok, corpus, a.tokens, INSTRUCTION)
    body = doc[: -len(INSTRUCTION)]
    runs = [
        ("cold", doc),
        ("repeat-1", doc),
        ("repeat-2", doc),
        ("same-doc-other-question", body + "\n\nList the function names above."),
    ]
    rows, base = [], None
    for name, prompt in runs:
        r = ask(a.url, a.model, prompt)
        if name == "cold":
            base = r["text"]
        r["run"] = name
        r["same_text_as_cold"] = (
            (r["text"] == base) if name.startswith("repeat") else None
        )
        r["text"] = r["text"][:60]
        rows.append(r)
        print(json.dumps(r), flush=True)
    if a.output:
        with a.output.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


main()
