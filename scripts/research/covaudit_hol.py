"""Head-of-line blocking: a short tool-call request arriving 1 s after a long cold prefill started.

Per arm (`--arm name=WORKTREE/python`, one server each, default config, prefix cache cold because
every rep uses a different document) and rep: the short request alone (baseline TTFT), then a
`--long-tokens` cold prompt with the same short request sent `--delay` s later. Records the short
request's time to first streamed token and to completion, and the long request's wall time.
Fails closed: the long reply must contain its needle and the short one a valid tool call.

    python covaudit_hol.py --model M --arm main=/p/python --arm v013=/p2/python --out hol.json
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

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather forecast for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"],
        },
    },
}


def short_body(model: str) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 100,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [TOOL],
        "tool_choice": "required",
        "messages": [
            {"role": "user", "content": "Get the weather in Taipei for 3 days."}
        ],
    }


def long_body(model: str, tokens: int, i: int) -> dict:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 40,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": f"<file>\n{file_text(i, tokens)}\n</file>\nReply with exactly the SECRET_CODE of the file above, nothing else.",
            }
        ],
    }


def stream(url: str, body: dict) -> dict:
    """POST a chat stream; time to first content/tool delta, total time, folded text and tool calls."""
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.monotonic()
    first, text, calls, done = None, "", {}, False
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                continue
            d = json.loads(payload)
            for ch in d.get("choices") or []:
                x = ch.get("delta") or {}
                if x.get("content") or x.get("tool_calls"):
                    if first is None:
                        first = time.monotonic() - t0
                    text += x.get("content") or ""
                    for tc in x.get("tool_calls") or []:
                        c = calls.setdefault(
                            tc.get("index", 0), {"name": "", "args": ""}
                        )
                        f = tc.get("function") or {}
                        c["name"] = f.get("name") or c["name"]
                        c["args"] += f.get("arguments") or ""
    return {
        "ttft_s": None if first is None else round(first, 2),
        "total_s": round(time.monotonic() - t0, 2),
        "text": text,
        "calls": list(calls.values()),
        "done": done,
    }


def tool_ok(r: dict) -> bool:
    for c in r["calls"]:
        try:
            a = json.loads(c["args"])
        except ValueError:
            continue
        if c["name"] == "get_weather" and "taipei" in str(a.get("city", "")).lower():
            return True
    return False


def judge(res: dict) -> list:
    """Failure reasons over {arm: [rep, ...]} (pure; unit-tested)."""
    bad = []
    for arm, reps in res.items():
        if not reps:
            bad.append(f"{arm}: no reps")
        for k, r in enumerate(reps):
            if not (r["solo"]["done"] and tool_ok(r["solo"])):
                bad.append(f"{arm} rep{k}: solo short request not a valid tool call")
            if not (r["short"]["done"] and tool_ok(r["short"])):
                bad.append(
                    f"{arm} rep{k}: short request during prefill not a valid tool call"
                )
            if not (r["long"]["done"] and needle(r["doc"]) in r["long"]["text"]):
                bad.append(
                    f"{arm} rep{k}: long reply wrong: {r['long']['text'][:50]!r}"
                )
            if r["short"]["ttft_s"] is None:
                bad.append(f"{arm} rep{k}: short request produced no token")
    return bad


def run_arm(name: str, src: str, model: str, a) -> list:
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
            doc = 40 + k
            solo = stream(srv.url, short_body(srv.model_id))
            box: dict = {}
            t = threading.Thread(
                target=lambda: box.update(
                    long=stream(srv.url, long_body(srv.model_id, a.long_tokens, doc))
                )
            )
            t.start()
            time.sleep(a.delay)
            short = stream(srv.url, short_body(srv.model_id))
            t.join()
            reps.append({"doc": doc, "solo": solo, "short": short, "long": box["long"]})
            print(
                f"{name} rep{k}: solo ttft {solo['ttft_s']}s | short ttft {short['ttft_s']}s total {short['total_s']}s "
                f"| long total {box['long']['total_s']}s",
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
    ap.add_argument("--long-tokens", type=int, default=32000)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3)
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
