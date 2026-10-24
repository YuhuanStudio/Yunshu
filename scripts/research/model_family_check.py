"""Per-model pass/fail receipt for docs/guides/MODEL_SUPPORT.md: load + chat + stream + tools + vision.

    model_family_check.py --model DIR --out F [--no-vision] [--timeout S]

One real server on one checkpoint (default settings, APC disk off), then served checks through the
official `openai` SDK: the model card (`/v1/models/{id}`), a chat completion, a streamed chat
completion with usage, a forced function call, and (when the card advertises image input) a
solid-red image the model must name. Each cell is `pass`, `fail` (with the reason) or `untested`
(with the reason); nothing is inferred. Exit 1 on any fail. Run only through gpuq.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

CELLS = ("load", "chat", "stream", "tools", "vision")
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def judge(cells: dict) -> list[str]:
    """Problems with one model's receipt (pure; unit-tested): every cell present, none failed."""
    bad = []
    for name in CELLS:
        row = cells.get(name)
        if row is None:
            bad.append(f"{name}: missing")
        elif row.get("status") not in ("pass", "untested"):
            bad.append(f"{name}: {row.get('detail', row.get('status'))}")
        elif row["status"] == "untested" and not row.get("detail"):
            bad.append(f"{name}: untested without a reason")
    if cells.get("load", {}).get("status") != "pass":
        bad.append("load: did not pass")
    return bad


def _text(msg) -> str:
    return (msg.content or "") + (getattr(msg, "reasoning_content", None) or "")


def check_chat(c):
    r = c.oa.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=300,
    )
    ch = r.choices[0]
    if not _text(ch.message).strip():
        raise AssertionError(f"empty answer, finish={ch.finish_reason}")
    if not r.usage or r.usage.completion_tokens < 1:
        raise AssertionError(f"usage {r.usage}")
    return {
        "finish": ch.finish_reason,
        "text": _text(ch.message)[:120],
        "completion_tokens": r.usage.completion_tokens,
    }


def check_stream(c):
    text, usage, finish, n = "", None, None, 0
    for ev in c.oa.chat.completions.create(
        model=c.model,
        messages=[{"role": "user", "content": "Count from one to five in words."}],
        max_tokens=300,
        stream=True,
        stream_options={"include_usage": True},
    ):
        n += 1
        if ev.usage:
            usage = ev.usage
        for ch in ev.choices:
            d = ch.delta
            text += (d.content or "") + (getattr(d, "reasoning_content", None) or "")
            finish = ch.finish_reason or finish
    if not text.strip():
        raise AssertionError(f"empty stream ({n} events)")
    if usage is None:
        raise AssertionError("no usage chunk")
    if finish is None:
        raise AssertionError("no finish_reason")
    return {"events": n, "finish": finish, "text": text[:120]}


def check_tools(c):
    import openai

    note = {}
    try:
        r = c.oa.chat.completions.create(
            model=c.model,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[TOOL],
            tool_choice="required",
            max_tokens=1500,
        )
    except openai.BadRequestError as e:
        # By design: a model whose tool grammar cannot compile refuses to promise a forced call.
        # Then `auto` must still produce a parsed call.
        if "Cannot guarantee forced tool_choice" not in str(e):
            raise
        note["forced_tool_choice"] = (
            "refused with 400 (documented: grammar cannot compile)"
        )
        r = c.oa.chat.completions.create(
            model=c.model,
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the get_weather tool.",
                }
            ],
            tools=[TOOL],
            tool_choice="auto",
            max_tokens=1500,
        )
    calls = r.choices[0].message.tool_calls or []
    if not calls:
        raise AssertionError(
            f"no tool call: finish={r.choices[0].finish_reason} {_text(r.choices[0].message)[:200]!r}"
        )
    args = json.loads(calls[0].function.arguments)
    if calls[0].function.name != "get_weather" or "city" not in args:
        raise AssertionError(f"bad call {calls[0].function}")
    return {"name": calls[0].function.name, "args": args, **note}


def check_vision(c):
    from route_checks import _png

    url = "data:image/png;base64," + base64.b64encode(_png(224)).decode()
    r = c.oa.chat.completions.create(
        model=c.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {
                        "type": "text",
                        "text": "What is the main colour of this image? Answer with one word.",
                    },
                ],
            }
        ],
        max_tokens=600,
    )
    ans = _text(r.choices[0].message).lower()
    if "red" not in ans:
        raise AssertionError(f"colour answer {ans[:200]!r}")
    return {"answer": ans[:120]}


def run_cell(name, fn, c, cells):
    t0 = time.monotonic()
    try:
        cells[name] = {"status": "pass", "notes": fn(c)}
    except BaseException as e:  # noqa: BLE001
        cells[name] = {
            "status": "fail",
            "detail": f"{type(e).__name__}: {str(e)[:600]}",
            "trace": traceback.format_exc()[-800:],
        }
    cells[name]["seconds"] = round(time.monotonic() - t0, 1)
    print(
        "cell",
        name,
        cells[name]["status"],
        cells[name].get("detail", "")[:200],
        flush=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-vision", action="store_true")
    a = ap.parse_args(argv)
    from m3sweep_jobs import routes_make_ctx, start_server, write

    name = Path(a.model).name
    res = {
        "kind": "model_family",
        "model": name,
        "path": a.model,
        "cells": {},
        "complete": False,
    }
    cells = res["cells"]
    srv = None
    t0 = time.monotonic()
    try:
        srv = start_server(a.model, "family", ["YUNSHU_VLM_APC_DISK=0"])
        cells["load"] = {"status": "pass", "seconds": round(time.monotonic() - t0, 1)}
        ctx = routes_make_ctx(srv, "", "vlm")
        card = ctx.http.get(f"/v1/models/{srv.model_id}").json()
        caps = (card.get("capabilities") or {}) if isinstance(card, dict) else {}
        res["card"] = {
            "capabilities": caps,
            "context": card.get("max_input_tokens") or card.get("context_length"),
        }
        run_cell("chat", check_chat, ctx, cells)
        run_cell("stream", check_stream, ctx, cells)
        run_cell("tools", check_tools, ctx, cells)
        img = (caps.get("image_input") or {}).get("supported") if caps else None
        if a.no_vision or not img:
            cells["vision"] = {
                "status": "untested",
                "detail": "the model card does not advertise image input"
                if not a.no_vision
                else "skipped by flag",
            }
        else:
            run_cell("vision", check_vision, ctx, cells)
    except BaseException as e:  # noqa: BLE001
        cells.setdefault(
            "load", {"status": "fail", "detail": f"{type(e).__name__}: {str(e)[:600]}"}
        )
        traceback.print_exc()
    finally:
        if srv:
            res["server_log_tail"] = srv.log_tail(30)
            srv.kill()
    for n in CELLS:
        cells.setdefault(n, {"status": "untested", "detail": "load failed"})
    res["failures"] = judge(cells)
    res["complete"] = True
    res["pass"] = not res["failures"]
    write(a.out, res)
    print(
        "RESULT", name, "PASS" if res["pass"] else "FAIL", res["failures"], flush=True
    )
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
