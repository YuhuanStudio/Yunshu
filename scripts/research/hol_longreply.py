"""Long-reply decode vs cold prefill: does a generous decode share slow a long cold prefill?

One server per arm (`--arm name=WORKTREE/python`), prefix cache cold for every B document.
Scenarios (each timed per rep; A = ~2000-token reply on a warm 8K context, B = cold long prompt):
  s1  A decoding, B (--b-tokens[0]) arrives `--delay` s later
  s2  same with the second B length
  s3  reverse: B prefilling, A (warm cache) arrives `--delay` s later
Reports B TTFT/total, A decode tok/s and total. Fails closed on a wrong needle or a short reply.

    python hol_longreply.py --model M --arm main=/p/python --out r.json --b-tokens 19700 59500
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from covaudit_session import Srv, file_text, needle  # noqa: E402

MIN_REPLY = 1200


def a_body(model: str, doc: int, tokens: int, max_tokens: int) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": f"<file>\n{file_text(doc, tokens)}\n</file>\nWrite a very long, detailed essay (at least 1500 words) about the history of computing. Do not stop early.",
            }
        ],
    }


def b_body(model: str, doc: int, tokens: int) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 40,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": f"<file>\n{file_text(doc, tokens)}\n</file>\nReply with exactly the SECRET_CODE of the file above, nothing else.",
            }
        ],
    }


def stream(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.monotonic()
    stamps: list[float] = []
    text, done = "", False
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                continue
            for ch in json.loads(payload).get("choices") or []:
                c = (ch.get("delta") or {}).get("content")
                if c:
                    stamps.append(time.monotonic() - t0)
                    text += c
    return summarize(stamps, text, done, time.monotonic() - t0)


def summarize(stamps: list[float], text: str, done: bool, total: float) -> dict:
    n = len(stamps)
    rate = (
        (n - 1) / (stamps[-1] - stamps[0]) if n > 1 and stamps[-1] > stamps[0] else 0.0
    )
    return {
        "ttft_s": round(stamps[0], 2) if stamps else None,
        "total_s": round(total, 2),
        "chunks": n,
        "tok_s": round(rate, 1),
        "text": text,
        "done": done,
    }


def judge(res: dict) -> list:
    bad = []
    for arm, reps in res.items():
        if not reps:
            bad.append(f"{arm}: no reps")
        for k, r in enumerate(reps):
            for sc, d in r["scenarios"].items():
                if not (d["a"]["done"] and d["b"]["done"]):
                    bad.append(f"{arm} rep{k} {sc}: incomplete stream")
                if d["a"]["chunks"] < MIN_REPLY:
                    bad.append(
                        f"{arm} rep{k} {sc}: A reply too short ({d['a']['chunks']})"
                    )
                if needle(d["doc"]) not in d["b"]["text"]:
                    bad.append(f"{arm} rep{k} {sc}: B needle missing")
    return bad


def run_pair(url, model, first, second, delay):
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(first=stream(url, first)))
    t.start()
    time.sleep(delay)
    box["second"] = stream(url, second)
    t.join()
    return box["first"], box["second"]


def run_arm(name, src, model, a):
    out = Path(a.out)
    srv = Srv(
        model,
        src,
        Path(f"/Volumes/P5Plus/yunshu-build/covaudit/home-hol-{name}"),
        out.with_name(f"{out.stem}-{name}.server.log"),
        [],
    )
    reps = []
    try:
        srv.wait_ready()
        for k in range(a.reps):
            rep = {"scenarios": {}}
            warm_doc = a.doc_base + k * 10
            # Warm the 8K context A reuses (and make A's own prefill a cache hit).
            stream(srv.url, a_body(srv.model_id, warm_doc, a.a_tokens, 8))
            for i, bt in enumerate(a.b_tokens):
                doc = a.doc_base + k * 10 + 1 + i
                af, bs = run_pair(
                    srv.url,
                    srv.model_id,
                    a_body(srv.model_id, warm_doc, a.a_tokens, a.reply),
                    b_body(srv.model_id, doc, bt),
                    a.delay,
                )
                rep["scenarios"][f"s{i + 1}_b{bt}"] = {"doc": doc, "a": af, "b": bs}
            doc = a.doc_base + k * 10 + 9
            bf, as_ = run_pair(
                srv.url,
                srv.model_id,
                b_body(srv.model_id, doc, a.b_tokens[0]),
                a_body(srv.model_id, warm_doc, a.a_tokens, a.reply),
                a.delay,
            )
            rep["scenarios"]["s3_reverse"] = {"doc": doc, "a": as_, "b": bf}
            reps.append(rep)
            for sc, d in rep["scenarios"].items():
                print(
                    f"{name} rep{k} {sc}: B ttft {d['b']['ttft_s']} total {d['b']['total_s']} | "
                    f"A {d['a']['tok_s']} tok/s total {d['a']['total_s']} chunks {d['a']['chunks']}",
                    flush=True,
                )
    finally:
        srv.kill()
    return reps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b-tokens", type=int, nargs="+", default=[19700, 59500])
    ap.add_argument("--a-tokens", type=int, default=4900)
    ap.add_argument("--reply", type=int, default=2000)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--doc-base", type=int, default=700)
    a = ap.parse_args(argv)
    res = {}
    try:
        for arm in a.arm:
            name, src = arm.split("=", 1)
            res[name] = run_arm(name, src, a.model, a)
    except BaseException as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    Path(a.out).write_text(json.dumps(res))
    bad = judge(res)
    for b in bad:
        print("JUDGE FAIL:", b)
    print("RESULT", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
