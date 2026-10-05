"""Concurrent agent + sub-agent requests against one server arm (coverage audit).

Four distinct chat requests (long document recall, medium document recall, a short question, a
tool call, a JSON-schema answer), greedy, prefix cache OFF (every request is cold, so a phase's
output cannot depend on what an earlier phase cached). Phases: solo (one at a time), c2 (long +
short together), c4 (all four, staggered 0.3 s). Fails when a request errors, an answer is wrong,
a tool call / JSON does not parse, or (identity) a concurrent output differs from its solo output.

    python covaudit_conc.py run --model M --src WORKTREE/python --out arm.jsonl [--long-tokens 16000]
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
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def make_requests(long_tokens: int, mid_tokens: int, model: str) -> dict:
    base = {
        "model": model,
        "temperature": 0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    doc = lambda i, n: f"<file>\n{file_text(i, n)}\n</file>\n"  # noqa: E731
    return {
        "long": (
            dict(
                base,
                messages=[
                    {
                        "role": "user",
                        "content": doc(11, long_tokens)
                        + "Reply with exactly the SECRET_CODE value of the file above, nothing else.",
                    }
                ],
            ),
            lambda r: needle(11) in (r["text"] or ""),
        ),
        "mid": (
            dict(
                base,
                messages=[
                    {
                        "role": "user",
                        "content": doc(12, mid_tokens)
                        + "Reply with exactly the SECRET_CODE value of the file above, nothing else.",
                    }
                ],
            ),
            lambda r: needle(12) in (r["text"] or ""),
        ),
        "short": (
            dict(
                base,
                messages=[
                    {
                        "role": "user",
                        "content": "What is 17 multiplied by 23? Reply with just the number.",
                    }
                ],
            ),
            lambda r: "391" in (r["text"] or ""),
        ),
        "tool": (
            dict(
                base,
                tools=[TOOL],
                tool_choice="required",
                messages=[
                    {"role": "user", "content": "Get the weather in Taipei for 3 days."}
                ],
            ),
            lambda r: any(
                c.get("name") == "get_weather"
                and isinstance(c.get("args"), dict)
                and "taipei" in str(c["args"].get("city", "")).lower()
                for c in r["tool_calls"]
            ),
        ),
        "json": (
            dict(
                base,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": SCHEMA, "strict": True},
                },
                messages=[
                    {"role": "user", "content": "What is 9 plus 8? Answer as JSON."}
                ],
            ),
            lambda r: _json_ok(r["text"]),
        ),
    }


def _json_ok(t) -> bool:
    try:
        d = json.loads(t)
        return isinstance(d, dict) and d.get("answer") == 17
    except Exception:
        return False


def post(url: str, body: dict, timeout=1800) -> dict:
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    calls = []
    for c in m.get("tool_calls") or []:
        try:
            args = json.loads(c["function"]["arguments"])
        except Exception:
            args = None
        calls.append({"name": c["function"]["name"], "args": args})
    return {
        "text": m.get("content") or "",
        "tool_calls": calls,
        "finish": d["choices"][0]["finish_reason"],
        "usage": d.get("usage") or {},
        "wall_s": round(time.monotonic() - t0, 2),
    }


def run_phase(url: str, reqs: dict, names: list, stagger: float) -> dict:
    out: dict = {}

    def one(n, delay):
        time.sleep(delay)
        try:
            out[n] = post(url, reqs[n][0])
        except Exception as e:  # recorded; the judge fails closed on it
            out[n] = {"error": f"{type(e).__name__}: {e}"}

    ts = [
        threading.Thread(target=one, args=(n, i * stagger)) for i, n in enumerate(names)
    ]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def judge(phases: dict, checks: dict) -> list:
    """Reasons for failure: errors, wrong answers, concurrent != solo (pure; unit-tested)."""
    bad = []
    solo = phases.get("solo", {})
    for ph, res in phases.items():
        for n, r in res.items():
            if "error" in r:
                bad.append(f"{ph}/{n}: {r['error']}")
                continue
            if r["finish"] not in ("stop", "tool_calls", "length"):
                bad.append(f"{ph}/{n}: finish {r['finish']}")
            if not checks[n](r):
                bad.append(
                    f"{ph}/{n}: wrong answer {r['text'][:60]!r} {r['tool_calls'][:1]}"
                )
            if ph != "solo" and n in solo and "error" not in solo[n]:
                s = solo[n]
                if (s["text"], s["tool_calls"]) != (r["text"], r["tool_calls"]):
                    bad.append(
                        f"{ph}/{n}: differs from solo: {r['text'][:50]!r} vs {s['text'][:50]!r}"
                    )
    if "solo" not in phases or not phases["solo"]:
        bad.append("no solo phase")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", required=True)
    ap.add_argument("--src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-tokens", type=int, default=16000)
    ap.add_argument("--mid-tokens", type=int, default=4000)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = Path(f"/Volumes/P5Plus/yunshu-build/covaudit/home-{out.stem}")
    sets = ["YUNSHU_VLM_APC_MEMORY_GB=0", "YUNSHU_VLM_APC_DISK=0", *a.set]
    srv = Srv(a.model, a.src, home, out.with_suffix(".server.log"), sets)
    rc = 2
    try:
        srv.wait_ready()
        reqs = make_requests(a.long_tokens, a.mid_tokens, srv.model_id)
        phases = {
            "solo": {},
        }
        for n in reqs:
            phases["solo"].update(run_phase(srv.url, reqs, [n], 0))
            print("solo", n, phases["solo"][n].get("wall_s"), flush=True)
        phases["c2"] = run_phase(srv.url, reqs, ["long", "short"], 0.3)
        phases["c4"] = run_phase(srv.url, reqs, ["long", "mid", "tool", "json"], 0.3)
        for ph in ("c2", "c4"):
            print(
                ph,
                {n: r.get("wall_s", r.get("error")) for n, r in phases[ph].items()},
                flush=True,
            )
        out.write_text(json.dumps(phases))
        bad = judge(phases, {n: v[1] for n, v in reqs.items()})
        for b in bad:
            print("JUDGE FAIL:", b)
        print("RESULT", "FAIL" if bad else "PASS")
        rc = 1 if bad else 0
    except BaseException as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        srv.kill()
    return rc


if __name__ == "__main__":
    sys.exit(main())
