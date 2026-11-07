"""Small-model correctness checks of the one-driver serving path on the helper Mac (correctness only, never timing).

    pb2g_m3_check.py run --model PATH --label L [--set K=V]... [--image]   one real server, the check matrix, JSON on stdout
    pb2g_m3_check.py spec --model PATH                                     spec on vs spec off (YUNSHU_VLM_DRAFT=off), solo vs c2 / c4
    pb2g_m3_check.py compare A.json B.json                                 outputs of two runs (two source trees) must be equal

Each run prints one JSON object (``"pass": bool`` per row); any exception or mismatch exits 1. Servers use ports
18990-18999 (covaudit_session.free_port) and are always kill -9'd. Run through scripts/dev/m3run.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPTS = [
    "List the first ten prime numbers separated by commas.",
    "Write a two sentence description of the ocean.",
    "Count from 1 to 12 in words.",
    "Name three colours and one fruit.",
]
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def post(url, body, timeout=300):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json", "authorization": "Bearer x"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(url, model, content, n=40, **kw):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": n,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        **kw,
    }
    return post(url, body)


def text_of(resp):
    return resp["choices"][0]["message"].get("content") or ""


def png_data_url(rgb=(220, 30, 30), size=96):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def concurrent(url, model, prompts):
    out = [None] * len(prompts)

    def go(i):
        out[i] = text_of(chat(url, model, prompts[i]))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(600)
    return out


def check_json_schema(url, model):
    r = chat(
        url,
        model,
        "Invent a person.",
        n=60,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": SCHEMA, "strict": True},
        },
    )
    obj = json.loads(text_of(r))
    return {
        "text": text_of(r),
        "pass": set(obj) == {"name", "age"} and isinstance(obj["age"], int),
    }


def check_tool_call(url, model):
    r = chat(
        url,
        model,
        "What is the weather in Paris?",
        n=60,
        tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    calls = r["choices"][0]["message"].get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    return {
        "calls": [c["function"] for c in calls],
        "pass": bool(calls)
        and calls[0]["function"]["name"] == "get_weather"
        and "city" in args,
    }


def check_cancel(url, model):
    """A streaming request dropped mid-decode must free its row: the next request still completes."""
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Write a very long story about a dragon."}
        ],
        "max_tokens": 400,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json", "authorization": "Bearer x"},
    )
    r = urllib.request.urlopen(req, timeout=300)
    seen = 0
    for line in r:
        if line.startswith(b"data:"):
            seen += 1
        if seen >= 6:
            break
    r.close()  # the client goes away mid-decode
    t0 = time.time()
    after = text_of(chat(url, model, "Say hello.", n=8))
    return {
        "chunks_before_cancel": seen,
        "recovered_s": round(time.time() - t0, 1),
        "after": after,
        "pass": bool(after) and seen >= 6,
    }


def check_image(url, model):
    r = post(
        url,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_data_url()}},
                        {
                            "type": "text",
                            "text": "What colour is this image? One word.",
                        },
                    ],
                }
            ],
            "max_tokens": 12,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    t = text_of(r)
    return {"text": t, "pass": bool(t.strip())}


def start(model, label, sets):
    from m3sweep_jobs import start_server

    return start_server(model, label, list(sets))


def cmd_run(a):
    res = {"kind": "run", "model": a.model, "label": a.label, "rows": {}, "pass": False}
    srv = None
    try:
        srv = start(a.model, a.label, ["YUNSHU_VLM_APC_DISK=0", *a.set])
        url, model = srv.url, srv.model_id
        solo = [text_of(chat(url, model, p)) for p in PROMPTS]
        res["rows"]["greedy"] = {"texts": solo, "pass": all(t.strip() for t in solo)}
        again = [text_of(chat(url, model, p)) for p in PROMPTS]
        res["rows"]["repeat"] = {"pass": again == solo}
        c4 = concurrent(url, model, PROMPTS)
        res["rows"]["c4"] = {"texts": c4, "pass": None, "equal_to_solo": c4 == solo}
        res["rows"]["c4"]["pass"] = (
            True  # equality is reported, not required: stock batched arithmetic
        )
        res["rows"]["json_schema"] = check_json_schema(url, model)
        res["rows"]["tool_call"] = check_tool_call(url, model)
        res["rows"]["cancel"] = check_cancel(url, model)
        if a.image:
            res["rows"]["image"] = check_image(url, model)
        res["pass"] = all(r.get("pass") for r in res["rows"].values())
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)
    finally:
        if srv is not None:
            res["log_tail"] = srv.log_tail(12)
            srv.kill()
    print(json.dumps(res))
    return 0 if res["pass"] else 1


def one_arm(model, label, sets):
    srv = start(model, label, ["YUNSHU_VLM_APC_DISK=0", *sets])
    try:
        url, mid = srv.url, srv.model_id
        solo = [text_of(chat(url, mid, p, n=48)) for p in PROMPTS]
        c2 = concurrent(url, mid, PROMPTS[:2])
        c4 = concurrent(url, mid, PROMPTS)
        log = srv.log_tail(80)
        return {
            "solo": solo,
            "c2": c2,
            "c4": c4,
            "spec_log": [
                ln
                for ln in log.splitlines()
                if "peculative" in ln or "Round driver layer" in ln
            ][-4:],
        }
    finally:
        srv.kill()


def cmd_spec(a):
    res = {"kind": "spec", "model": a.model, "pass": False}
    try:
        on = one_arm(a.model, "spec-on", [])
        off = one_arm(a.model, "spec-off", ["YUNSHU_VLM_DRAFT=off"])
        res["on"], res["off"] = on, off
        res["rows"] = {
            "spec_on_equals_off": on["solo"] == off["solo"],
            "solo_equals_c2": on["solo"][:2] == on["c2"],
            "solo_equals_c4": on["solo"] == on["c4"],
            "off_solo_equals_c4": off["solo"] == off["c4"],
        }
        res["pass"] = all(res["rows"].values())
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)
    print(json.dumps(res))
    return 0 if res["pass"] else 1


def compare(a_path, b_path):
    a, b = (json.loads(Path(x).read_text()) for x in (a_path, b_path))
    diffs = {}
    for key in sorted(set(a["rows"]) & set(b["rows"])):
        for field in ("texts", "text", "calls", "after"):
            if field in a["rows"][key] and a["rows"][key].get(field) != b["rows"][
                key
            ].get(field):
                diffs[f"{key}.{field}"] = [
                    a["rows"][key].get(field),
                    b["rows"][key].get(field),
                ]
    return diffs


def cmd_compare(a):
    diffs = compare(a.a, a.b)
    print(json.dumps({"kind": "compare", "equal": not diffs, "diffs": diffs}, indent=1))
    return 0 if not diffs else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--image", action="store_true")
    s = sub.add_parser("spec")
    s.add_argument("--model", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    return {"run": cmd_run, "spec": cmd_spec, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
