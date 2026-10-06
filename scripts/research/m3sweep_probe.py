"""Raw request/response dump for wire findings (run through the M3 lane; correctness only)."""

import argparse
import contextlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3sweep_jobs as mj  # noqa: E402

W = {
    "type": "function",
    "name": "get_weather",
    "description": "weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
CW = {
    "type": "function",
    "function": {k: W[k] for k in ("name", "description", "parameters")},
}


def post(url, path, body, headers=None):
    req = urllib.request.Request(
        url + path,
        json.dumps(body).encode(),
        {
            "content-type": "application/json",
            "x-api-key": "x",
            "anthropic-version": "2023-06-01",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def summarize_sse(txt):
    """Event types in order (run-length) plus the concatenated text / thinking / tool json deltas."""
    ev, text = [], {"text": "", "think": "", "args": ""}
    for blk in txt.split("\n\n"):
        lines = {
            x.split(": ", 1)[0]: x.split(": ", 1)[1]
            for x in blk.splitlines()
            if ": " in x
        }
        t = lines.get("event")
        data = {}
        with contextlib.suppress(ValueError):
            data = json.loads(lines.get("data", "{}"))
        t = t or ("chunk" if data else "")
        if not ev or ev[-1][0] != t:
            ev.append([t, 0])
        ev[-1][1] += 1
        d = data.get("delta")
        if isinstance(d, dict):
            text["text"] += d.get("text") or ""
            text["think"] += d.get("thinking") or ""
            text["args"] += d.get("partial_json") or ""
        elif isinstance(d, str):
            text["text"] += d
        for c in data.get("choices") or []:
            dd = c.get("delta") or {}
            text["text"] += dd.get("content") or ""
            text["think"] += dd.get("reasoning_content") or ""
            for tc in dd.get("tool_calls") or []:
                text["args"] += (tc.get("function") or {}).get("arguments") or ""
    return json.dumps({"events": ev, **text, "tail": txt[-1200:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grammar", action="store_true")
    a = ap.parse_args()
    srv = mj.start_server(
        a.model,
        "probe",
        ["YUNSHU_VLM_APC_DISK=0"] + (["YUNSHU_TOOL_GRAMMAR=1"] if a.grammar else []),
    )
    res = {"complete": False, "pass": True, "rows": {}}
    try:
        m = srv.model_id
        msgs = [{"role": "user", "content": "hi"}]
        cases = {
            "chat_named": (
                "/v1/chat/completions",
                {
                    "model": m,
                    "messages": msgs,
                    "tools": [CW],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "get_weather"},
                    },
                    "max_tokens": 1500,
                },
            ),
            "chat_named_s": (
                "/v1/chat/completions",
                {
                    "model": m,
                    "messages": msgs,
                    "tools": [CW],
                    "tool_choice": "required",
                    "max_tokens": 1500,
                    "stream": True,
                },
            ),
            "resp_req": (
                "/v1/responses",
                {
                    "model": m,
                    "input": "hi",
                    "tools": [W],
                    "tool_choice": "required",
                    "max_output_tokens": 1500,
                },
            ),
            "resp_req_s": (
                "/v1/responses",
                {
                    "model": m,
                    "input": "hi",
                    "tools": [W],
                    "tool_choice": "required",
                    "max_output_tokens": 1500,
                    "stream": True,
                },
            ),
            "msg_req": (
                "/v1/messages",
                {
                    "model": m,
                    "max_tokens": 1500,
                    "messages": msgs,
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "weather",
                            "input_schema": W["parameters"],
                        }
                    ],
                    "tool_choice": {"type": "any"},
                },
            ),
            "msg_req_s": (
                "/v1/messages",
                {
                    "model": m,
                    "max_tokens": 1500,
                    "stream": True,
                    "messages": msgs,
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "weather",
                            "input_schema": W["parameters"],
                        }
                    ],
                    "tool_choice": {"type": "any"},
                },
            ),
            "count": (
                "/v1/messages/count_tokens",
                {
                    "model": m,
                    "messages": msgs,
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "weather",
                            "input_schema": W["parameters"],
                        }
                    ],
                    "tool_choice": {"type": "any"},
                },
            ),
        }
        for k, (path, body) in cases.items():
            st, txt = post(srv.url, path, body)
            if body.get("stream"):
                txt = summarize_sse(txt)
            res["rows"][k] = {"status": st, "body": txt[:6000]}
            print(k, st, txt[:300].replace("\n", " "), flush=True)
    finally:
        res["server_log_tail"] = srv.log_tail(60)
        srv.kill()
    res["complete"] = True
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
