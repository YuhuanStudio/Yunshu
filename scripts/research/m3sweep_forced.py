"""Repeat forced tool_choice requests on one server and count the ones without a tool call.

Correctness probe (M3 lane): `--model M --out F`. Each variant runs N times against one server;
a request is BAD when the reply carries no parsed tool call. Writes counts and the first bad
bodies; passes only when no variant has a BAD reply.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3sweep_jobs as mj  # noqa: E402
import m3sweep_probe as mp  # noqa: E402

PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
ATOOL = {"name": "get_weather", "description": "weather", "input_schema": PARAMS}
CTOOL = {
    "type": "function",
    "function": {"name": "get_weather", "description": "weather", "parameters": PARAMS},
}


def has_call(kind, stream, raw):
    """Does the reply carry a tool call? (pure; sniffs the wire text)"""
    if kind == "messages":
        return '"type": "tool_use"' in raw or '"type":"tool_use"' in raw
    if kind == "chat":
        return '"tool_calls"' in raw and '"arguments"' in raw
    return '"function_call"' in raw


def variants(m):
    msgs = [{"role": "user", "content": "hi"}]
    out = {}
    for stream in (False, True):
        for temp in (None, 0):
            t = {} if temp is None else {"temperature": temp}
            tag = f"{'s' if stream else 'j'}{'' if temp is None else '-t0'}"
            out[f"messages-named-{tag}"] = (
                "messages",
                stream,
                "/v1/messages",
                {
                    "model": m,
                    "max_tokens": 1500,
                    "stream": stream,
                    "messages": msgs,
                    "tools": [ATOOL],
                    "tool_choice": {"type": "tool", "name": "get_weather"},
                    **t,
                },
            )
        out[f"messages-any-{'s' if stream else 'j'}"] = (
            "messages",
            stream,
            "/v1/messages",
            {
                "model": m,
                "max_tokens": 1500,
                "stream": stream,
                "messages": msgs,
                "tools": [ATOOL],
                "tool_choice": {"type": "any"},
            },
        )
        out[f"responses-required-{'s' if stream else 'j'}"] = (
            "responses",
            stream,
            "/v1/responses",
            {
                "model": m,
                "max_output_tokens": 1500,
                "stream": stream,
                "input": "hi",
                "tools": [{"type": "function", **CTOOL["function"]}],
                "tool_choice": "required",
            },
        )
        out[f"chat-named-{'s' if stream else 'j'}"] = (
            "chat",
            stream,
            "/v1/chat/completions",
            {
                "model": m,
                "max_tokens": 1500,
                "stream": stream,
                "messages": msgs,
                "tools": [CTOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            },
        )
        out[f"chat-required-{'s' if stream else 'j'}"] = (
            "chat",
            stream,
            "/v1/chat/completions",
            {
                "model": m,
                "max_tokens": 1500,
                "stream": stream,
                "messages": msgs,
                "tools": [CTOOL],
                "tool_choice": "required",
            },
        )
        out[f"chat-required-serial-{'s' if stream else 'j'}"] = (
            "chat",
            stream,
            "/v1/chat/completions",
            {
                "model": m,
                "max_tokens": 1500,
                "stream": stream,
                "messages": msgs,
                "tools": [CTOOL],
                "tool_choice": "required",
                "parallel_tool_calls": False,
            },
        )
        out[f"messages-any-serial-{'s' if stream else 'j'}"] = (
            "messages",
            stream,
            "/v1/messages",
            {
                "model": m,
                "max_tokens": 1500,
                "stream": stream,
                "messages": msgs,
                "tools": [ATOOL],
                "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
            },
        )
        out[f"responses-named-{'s' if stream else 'j'}"] = (
            "responses",
            stream,
            "/v1/responses",
            {
                "model": m,
                "max_output_tokens": 1500,
                "stream": stream,
                "input": "hi",
                "tools": [{"type": "function", **CTOOL["function"]}],
                "tool_choice": {"type": "function", "name": "get_weather"},
            },
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--only", default="", help="comma list of name substrings")
    a = ap.parse_args()
    srv = mj.start_server(a.model, "forced", ["YUNSHU_VLM_APC_DISK=0", *a.set])
    res = {"complete": False, "pass": False, "variants": {}, "bad": {}}
    try:
        want = [w for w in a.only.split(",") if w]
        for name, (kind, stream, path, body) in variants(srv.model_id).items():
            if want and not any(w in name for w in want):
                continue
            bad = []
            for _i in range(a.reps):
                st, raw = mp.post(srv.url, path, body)
                if st != 200 or not has_call(kind, stream, raw):
                    bad.append(raw[:4000])
            res["variants"][name] = {"bad": len(bad), "of": a.reps}
            res["bad"][name] = bad[:2]
            print(name, f"bad {len(bad)}/{a.reps}", flush=True)
        res["pass"] = all(v["bad"] == 0 for v in res["variants"].values())
    finally:
        res["server_log_tail"] = srv.log_tail(40)
        srv.kill()
    res["complete"] = True
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", "PASS" if res["pass"] else "FAIL", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
