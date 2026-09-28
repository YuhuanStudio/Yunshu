"""Cross-engine performance + capability matrix over any OpenAI-compatible server.

One script, same requests, for Yunshu and every reference engine (oMLX, Splash,
MTPLX, mlx-vlm server, mlx-lm server). Measures what a user feels — socket
first-content, decode rate, prefix reuse over growing chats and edited long
documents, image turns — and checks the capabilities an acceleration must not
break: typed tool arguments (stream + non-stream), JSON schema, stop sequences,
length truncation, and recovery after a mid-stream disconnect. Optionally samples
the server's process-tree physical footprint after every case and after idle.

Every output file starts with a ``meta`` row (engine label, checkpoint, versions,
git SHA, flags) so a result is never detached from its environment. Never resolves
or downloads models; the server must already be running.

    python scripts/research/bench_engine_matrix.py --url http://127.0.0.1:18764 \
        --model Qwen3.8-27B --engine yunshu-apc --checkpoint /Volumes/.../Jundot/... \
        --pid 12345 --output docs/research/runs/2026-09-28-matrix/yunshu-apc.jsonl
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.parse
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DOC_LINE = (
    "Section {i}: The municipal archive logs routine maintenance for building "
    "{b}, including inspections of wiring, plumbing, and the elevator shaft.\n"
)


def long_document(lines: int) -> str:
    return "".join(DOC_LINE.format(i=i, b=100 + i % 37) for i in range(lines))


def image_url(red_left: bool, size: int = 256) -> str:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size // 2), "blue")
    box = (
        (0, 0, size // 2 - 1, size // 2 - 1)
        if red_left
        else (size // 2, 0, size - 1, size // 2 - 1)
    )
    ImageDraw.Draw(im).rectangle(box, fill="red")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get a weather forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "metric": {"type": "boolean"},
            },
            "required": ["city", "days", "metric"],
        },
    },
}

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "population_millions": {"type": "number"},
        "is_capital": {"type": "boolean"},
        "landmarks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 3,
        },
    },
    "required": ["city", "country", "population_millions", "is_capital", "landmarks"],
    "additionalProperties": False,
}


class Client:
    def __init__(self, url: str, model: str, no_think: str, timeout: float):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.model = model
        self.no_think = no_think
        self.timeout = timeout

    def body(self, messages, max_tokens, **extra):
        b = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if self.no_think in ("kwargs", "both"):
            b["chat_template_kwargs"] = {"enable_thinking": False}
        if self.no_think in ("flag", "both"):
            b["enable_thinking"] = False
        b.update(extra)
        return b

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def stream(self, body, abort_after_chunks: int | None = None):
        body = dict(body, stream=True, stream_options={"include_usage": True})
        r = {
            "first_content_s": None,
            "first_event_s": None,
            "content": "",
            "reasoning": "",
            "tool_calls": {},
            "finish_reason": None,
            "usage": None,
            "done": False,
            "chunks": 0,
        }
        t0 = time.perf_counter()
        conn = self._conn()
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            r["http_status"] = resp.status
            if resp.status != 200:
                r["error_body"] = resp.read().decode(errors="replace")[:2000]
                return r
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    r["done"] = True
                    break
                ev = json.loads(data)
                now = time.perf_counter() - t0
                if r["first_event_s"] is None:
                    r["first_event_s"] = now
                if ev.get("error"):
                    r["stream_error"] = ev["error"]
                if ev.get("usage"):
                    r["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    txt = d.get("content")
                    if txt:
                        if r["first_content_s"] is None:
                            r["first_content_s"] = now
                        r["content"] += txt
                        r["chunks"] += 1
                    rs = d.get("reasoning_content") or d.get("reasoning")
                    if rs:
                        r["reasoning"] += rs
                    for tc in d.get("tool_calls") or []:
                        if r["first_content_s"] is None:
                            r["first_content_s"] = now
                        slot = r["tool_calls"].setdefault(
                            tc.get("index", 0), {"name": "", "arguments": ""}
                        )
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        r["finish_reason"] = ch["finish_reason"]
                if abort_after_chunks is not None and r["chunks"] >= abort_after_chunks:
                    r["aborted_client_side"] = True
                    break
        finally:
            conn.close()
        r["wall_s"] = time.perf_counter() - t0
        r["tool_calls"] = list(r["tool_calls"].values())
        return r

    def complete(self, body):
        t0 = time.perf_counter()
        conn = self._conn()
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(dict(body, stream=False)),
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode(errors="replace")
        finally:
            conn.close()
        r = {"http_status": resp.status, "wall_s": time.perf_counter() - t0}
        try:
            j = json.loads(raw)
        except json.JSONDecodeError:
            r["error_body"] = raw[:2000]
            return r
        if resp.status != 200:
            r["error_body"] = raw[:2000]
            return r
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        r.update(
            content=msg.get("content") or "",
            finish_reason=ch.get("finish_reason"),
            usage=j.get("usage"),
            tool_calls=[
                {
                    "name": (t.get("function") or {}).get("name"),
                    "arguments": (t.get("function") or {}).get("arguments"),
                }
                for t in msg.get("tool_calls") or []
            ],
        )
        return r


def decode_tps(r):
    u = r.get("usage") or {}
    n = u.get("completion_tokens")
    if not n or r.get("first_content_s") is None or n < 8:
        return None
    span = r["wall_s"] - r["first_content_s"]
    return round((n - 1) / span, 2) if span > 0 else None


def cached_tokens(r):
    u = r.get("usage") or {}
    d = u.get("prompt_tokens_details") or {}
    return d.get("cached_tokens", u.get("cached_tokens"))


def check_tool(r):
    calls = r.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("name") != "get_weather":
        return False, f"calls={calls!r}"
    try:
        args = (
            json.loads(calls[0]["arguments"])
            if isinstance(calls[0]["arguments"], str)
            else calls[0]["arguments"]
        )
    except Exception as e:  # noqa: BLE001
        return False, f"bad args json {e}"
    ok = (
        isinstance(args.get("city"), str)
        and "taipei" in args["city"].lower()
        and args.get("days") == 3
        and type(args.get("days")) is int
        and args.get("metric") is True
    )
    leak = "<tool_call" in (r.get("content") or "") or "<function" in (
        r.get("content") or ""
    )
    return ok and not leak, f"args={args!r} leak={leak} finish={r.get('finish_reason')}"


def check_schema(text):
    t = (text or "").strip()
    try:
        obj = json.loads(t)
    except Exception as e:  # noqa: BLE001
        return False, f"not json: {e}: {t[:120]!r}"
    try:
        import jsonschema

        jsonschema.validate(obj, SCHEMA)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        return False, f"schema: {str(e)[:160]}"
    ok = "tokyo" in str(obj.get("city", "")).lower() and obj.get("is_capital") is True
    return ok, f"obj={json.dumps(obj, ensure_ascii=False)[:200]}"


def build_cases(doc_lines: int, chat_turns: int):
    doc = long_document(doc_lines)
    ask = "\n\nQuestion: according to the final line, what is the access code? Reply with the code only."
    cases = []
    cases.append(
        (
            "short_cold",
            "perf",
            [{"role": "user", "content": "Reply with only the word ORCHID."}],
            16,
            "ORCHID",
        )
    )
    cases.append(
        (
            "short_warm",
            "perf",
            [{"role": "user", "content": "Reply with only the word ORCHID."}],
            16,
            "ORCHID",
        )
    )
    cases.append(
        (
            "decode_code",
            "perf",
            [
                {
                    "role": "user",
                    "content": "Write a detailed Python module implementing an LRU cache class with get, put, delete, resize and __len__, with docstrings and type hints. Output code only.",
                }
            ],
            512,
            "class",
        )
    )
    for name, code in (
        ("doc_cold", "ALPHA"),
        ("doc_repeat", "ALPHA"),
        ("doc_edited_tail", "COBALT"),
    ):
        cases.append(
            (
                name,
                "perf",
                [
                    {
                        "role": "user",
                        "content": doc
                        + f"Final line: the access code is {code}."
                        + ask,
                    }
                ],
                16,
                code,
            )
        )
    # Growing chat over a shared document: every turn resends the whole history,
    # which is what any OpenAI client does. Assistant replies are fixed so all
    # engines see identical prompts.
    codes = [
        "RIVER",
        "MAPLE",
        "OCEAN",
        "EMBER",
        "CEDAR",
        "LUNAR",
        "PRISM",
        "SOLAR",
        "TIDAL",
        "AMBER",
        "FROST",
        "CORAL",
    ]
    hist = [
        {"role": "system", "content": "You are a precise assistant."},
        {
            "role": "user",
            "content": long_document(max(doc_lines // 4, 40))
            + "\nI will give you status codes; keep track of the latest one.",
        },
        {"role": "assistant", "content": "Understood."},
    ]
    for t in range(chat_turns):
        c = codes[t % len(codes)]
        msgs = hist + [
            {
                "role": "user",
                "content": f"Turn {t + 1}: the status code is now {c}. What is the current status code? Reply with the code only.",
            }
        ]
        cases.append((f"chat_turn_{t + 1:02d}", "perf", msgs, 12, c))
        hist = msgs + [{"role": "assistant", "content": c}]
    img_q = (
        "Which half of the image is red? Reply with exactly one word: left or right."
    )
    cases.append(
        (
            "image_left",
            "perf",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url(True)}},
                        {"type": "text", "text": img_q},
                    ],
                }
            ],
            8,
            "left",
        )
    )
    followup = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url(True)}},
                {"type": "text", "text": img_q},
            ],
        },
        {"role": "assistant", "content": "left"},
        {
            "role": "user",
            "content": "What color is the other half? Reply with one word.",
        },
    ]
    cases.append(("image_followup", "perf", followup, 8, "blue"))
    cases.append(
        (
            "image_right",
            "perf",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url(False)}},
                        {"type": "text", "text": img_q},
                    ],
                }
            ],
            8,
            "right",
        )
    )
    # Same image + question again: prefix reuse across image turns.
    cases.append(
        (
            "image_right_repeat",
            "perf",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url(False)}},
                        {"type": "text", "text": img_q},
                    ],
                }
            ],
            8,
            "right",
        )
    )
    # Two different images in one turn; the answer depends on the second one.
    two_q = (
        "There are two images. In the SECOND image, which half is red? "
        "Reply with exactly one word: left or right."
    )
    cases.append(
        (
            "image_two",
            "perf",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url(True)}},
                        {"type": "image_url", "image_url": {"url": image_url(False)}},
                        {"type": "text", "text": two_q},
                    ],
                }
            ],
            8,
            "right",
        )
    )
    return cases


def sample_memory(pid):
    if not pid:
        return None
    try:
        from process_memory import process_tree_memory

        m = process_tree_memory(pid)
        return {
            "footprint_gib": round(m["physical_footprint_sum_bytes"] / 2**30, 3),
            "rss_gib": round(m["rss_sum_bytes"] / 2**30, 3),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True, help="label, e.g. yunshu-apc, omlx-mtp")
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="path/revision actually loaded by the server",
    )
    ap.add_argument("--pid", type=int, help="server root pid for footprint sampling")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--doc-lines", type=int, default=260, help="~8.4K tokens at 260")
    ap.add_argument("--chat-turns", type=int, default=12)
    ap.add_argument(
        "--no-think", choices=["kwargs", "flag", "both", "none"], default="both"
    )
    ap.add_argument("--only", nargs="*", help="case-name prefixes to run")
    ap.add_argument("--skip", nargs="*", default=[], help="case-name prefixes to skip")
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    cli = Client(args.url, args.model, args.no_think, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.open("a")

    def emit(row):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        short = {
            k: v for k, v in row.items() if k not in ("content", "reasoning", "raw")
        }
        print(json.dumps(short, ensure_ascii=False)[:600], flush=True)

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        sha = None
    emit(
        {
            "kind": "meta",
            "engine": args.engine,
            "model": args.model,
            "checkpoint": args.checkpoint,
            "url": args.url,
            "pid": args.pid,
            "yunshu_git": sha,
            "host": platform.node(),
            "macos": platform.mac_ver()[0],
            "no_think": args.no_think,
            "doc_lines": args.doc_lines,
            "chat_turns": args.chat_turns,
            "note": args.note,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "env_flags": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("YUNSHU_", "OMLX_", "SPLASH_", "MTPLX_"))
            },
            "memory_start": sample_memory(args.pid),
        }
    )

    def want(name):
        if any(name.startswith(s) for s in args.skip):
            return False
        return not args.only or any(name.startswith(s) for s in args.only)

    for name, kind, msgs, limit, expected in build_cases(
        args.doc_lines, args.chat_turns
    ):
        if not want(name):
            continue
        try:
            r = cli.stream(cli.body(msgs, limit))
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e)}
        text = (r.get("content") or "").strip()
        ok = (
            (expected in text)
            if name == "decode_code"
            else text.strip(" .\n\"'`").lower() == expected.lower()
        )
        emit(
            {
                "kind": kind,
                "case": name,
                "ok": ok,
                "expected": expected,
                **r,
                "decode_tps": decode_tps(r),
                "cached_tokens": cached_tokens(r),
                "memory": sample_memory(args.pid),
            }
        )

    # ── capability gates ──
    tool_msgs = [
        {
            "role": "user",
            "content": "Get the weather forecast for Taipei for the next 3 days in metric units.",
        }
    ]
    for mode in ("nonstream", "stream"):
        name = f"tool_{mode}"
        if not want(name):
            continue
        body = cli.body(tool_msgs, 256, tools=[TOOL], tool_choice="auto")
        try:
            r = cli.complete(body) if mode == "nonstream" else cli.stream(body)
            ok, why = check_tool(r)
        except Exception as e:  # noqa: BLE001
            r, ok, why = {"error": repr(e)}, False, "exception"
        emit(
            {
                "kind": "capability",
                "case": name,
                "ok": ok,
                "why": why,
                **r,
                "memory": sample_memory(args.pid),
            }
        )

    schema_msgs = [
        {
            "role": "user",
            "content": "Give facts about Tokyo as JSON: city, country, population_millions, is_capital, and 2-3 landmarks.",
        }
    ]
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "city_facts", "schema": SCHEMA, "strict": True},
    }
    for mode in ("nonstream", "stream"):
        name = f"schema_{mode}"
        if not want(name):
            continue
        body = cli.body(schema_msgs, 256, response_format=rf)
        try:
            r = cli.complete(body) if mode == "nonstream" else cli.stream(body)
            ok, why = check_schema(r.get("content"))
        except Exception as e:  # noqa: BLE001
            r, ok, why = {"error": repr(e)}, False, "exception"
        emit(
            {
                "kind": "capability",
                "case": name,
                "ok": ok,
                "why": why,
                **r,
                "memory": sample_memory(args.pid),
            }
        )

    if want("image_schema"):
        img_schema = {
            "type": "object",
            "properties": {
                "red_half": {"type": "string", "enum": ["left", "right"]},
                "other_color": {"type": "string"},
            },
            "required": ["red_half", "other_color"],
            "additionalProperties": False,
        }
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url(True)}},
                    {
                        "type": "text",
                        "text": "Which half is red, and what color is the other half? Answer as JSON.",
                    },
                ],
            }
        ]
        body = cli.body(
            msgs,
            96,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "halves", "schema": img_schema, "strict": True},
            },
        )
        try:
            r = cli.stream(body)
            obj = json.loads((r.get("content") or "").strip())
            ok = (
                obj.get("red_half") == "left"
                and "blue" in str(obj.get("other_color", "")).lower()
            )
            why = f"obj={obj!r}"
        except Exception as e:  # noqa: BLE001
            r = r if "r" in dir() else {}
            ok, why = (
                False,
                f"{e!r} http={r.get('http_status')} body={str(r.get('error_body'))[:200]}",
            )
        emit(
            {
                "kind": "capability",
                "case": "image_schema",
                "ok": ok,
                "why": why,
                **r,
                "memory": sample_memory(args.pid),
            }
        )

    if want("logprobs"):
        body = cli.body(
            [{"role": "user", "content": "Reply with only the word ORCHID."}],
            8,
            logprobs=True,
            top_logprobs=3,
        )
        r = cli.complete(body)
        raw = None
        try:
            conn = http.client.HTTPConnection(cli.host, cli.port, timeout=cli.timeout)
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(dict(body, stream=False)),
                {"Content-Type": "application/json"},
            )
            raw = json.loads(conn.getresponse().read())
            conn.close()
            content = ((raw.get("choices") or [{}])[0].get("logprobs") or {}).get(
                "content"
            ) or []
            ok = bool(content) and all(
                isinstance(e.get("logprob"), (int, float))
                and e["logprob"] <= 0
                and len(e.get("top_logprobs") or []) == 3
                for e in content
            )
            why = f"{len(content)} entries, first={content[0] if content else None}"
        except Exception as e:  # noqa: BLE001
            ok, why = False, repr(e)
        emit(
            {
                "kind": "capability",
                "case": "logprobs",
                "ok": ok,
                "why": str(why)[:300],
                **r,
            }
        )

    if want("stop_sequence"):
        r = cli.stream(
            cli.body(
                [
                    {
                        "role": "user",
                        "content": "Count from 1 to 20, separated by commas and spaces. Output only the numbers.",
                    }
                ],
                96,
                stop=[", 6"],
            )
        )
        t = r.get("content") or ""
        ok = (
            ", 6" not in t
            and t.strip().startswith("1")
            and "5" in t
            and r.get("finish_reason") == "stop"
        )
        emit(
            {
                "kind": "capability",
                "case": "stop_sequence",
                "ok": ok,
                "why": f"{t!r} finish={r.get('finish_reason')}",
                **r,
            }
        )

    if want("length_finish"):
        r = cli.stream(
            cli.body(
                [
                    {
                        "role": "user",
                        "content": "Write a long essay about the history of the printing press.",
                    }
                ],
                20,
            )
        )
        u = r.get("usage") or {}
        ok = r.get("finish_reason") == "length" and u.get("completion_tokens") in (
            19,
            20,
            21,
        )
        emit(
            {
                "kind": "capability",
                "case": "length_finish",
                "ok": ok,
                "why": f"finish={r.get('finish_reason')} completion_tokens={u.get('completion_tokens')}",
                **r,
            }
        )

    if want("disconnect_recovery"):
        long_msgs = [
            {
                "role": "user",
                "content": "Write a very long story about a lighthouse keeper. At least 2000 words.",
            }
        ]
        a = cli.stream(cli.body(long_msgs, 2048), abort_after_chunks=8)
        t0 = time.perf_counter()
        b = cli.stream(
            cli.body(
                [{"role": "user", "content": "Reply with only the word COBALT."}], 16
            )
        )
        ok = (b.get("content") or "").strip(" .").upper() == "COBALT"
        emit(
            {
                "kind": "capability",
                "case": "disconnect_recovery",
                "ok": ok,
                "why": f"after abort next wall={b.get('wall_s'):.3f}s first={b.get('first_content_s')}",
                "aborted": {
                    k: a.get(k) for k in ("chunks", "first_content_s", "wall_s")
                },
                "recovery_wall_s": round(time.perf_counter() - t0, 3),
                "content": b.get("content"),
                "memory": sample_memory(args.pid),
            }
        )

    if args.idle_s > 0 and args.pid:
        time.sleep(args.idle_s)
        emit(
            {
                "kind": "idle",
                "case": f"idle_{int(args.idle_s)}s",
                "memory": sample_memory(args.pid),
            }
        )

    rows = [
        json.loads(line)
        for line in args.output.read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("kind") != "meta" and r.get("case")]
    chat = [
        r["first_content_s"]
        for r in rows
        if r["case"].startswith("chat_turn_")
        and r["case"] != "chat_turn_01"
        and r.get("first_content_s")
    ]
    summ = {
        "kind": "summary",
        "engine": args.engine,
        "ok": sum(1 for r in rows if r.get("ok")),
        "total": sum(1 for r in rows if "ok" in r),
        "failed": [r["case"] for r in rows if r.get("ok") is False],
        "chat_warm_ttft_median": round(statistics.median(chat), 3) if chat else None,
        "decode_tps": next(
            (r.get("decode_tps") for r in rows if r["case"] == "decode_code"), None
        ),
    }
    for c in (
        "short_cold",
        "short_warm",
        "doc_cold",
        "doc_repeat",
        "doc_edited_tail",
        "image_left",
        "image_followup",
    ):
        row = next((r for r in rows if r["case"] == c), None)
        if row:
            summ[f"{c}_ttft"] = row.get("first_content_s") and round(
                row["first_content_s"], 3
            )
    peaks = [
        r["memory"]["footprint_gib"]
        for r in rows
        if isinstance(r.get("memory"), dict) and "footprint_gib" in r["memory"]
    ]
    if peaks:
        summ["footprint_max_gib"] = max(peaks)
    emit(summ)


if __name__ == "__main__":
    main()
