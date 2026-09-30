"""Time-bounded realistic soak over any OpenAI-compatible server.

Mixes what a real user does for an hour: questions over a pool of long documents
(bigger than a sane prefix-cache budget, so eviction happens), three growing chat
threads, images of varying size, tool calls, JSON schema, thinking with sampling,
long generations and mid-stream disconnects, with idle pauses. Every request is
checked for a correct answer where one exists; process-tree physical footprint is
sampled every few seconds; after the run the server idles and memory is sampled
again to see whether it returns toward the start.

    python scripts/research/soak_realistic.py --url http://127.0.0.1:18764 \
        --model Qwen3.8 --pid 1234 --minutes 60 --output runs/soak.jsonl
"""

import argparse
import base64
import http.client
import json
import random
import statistics
import sys
import threading
import time
import urllib.parse
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_memory import process_tree_memory  # noqa: E402

CODES = [
    "AMBER",
    "BIRCH",
    "CEDAR",
    "DELTA",
    "EMBER",
    "FROST",
    "GLINT",
    "HAZEL",
    "IVORY",
    "JADE",
    "KOALA",
    "LUNAR",
    "MAPLE",
    "NOVA",
    "ONYX",
    "PRISM",
]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather forecast",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"],
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "population_millions": {"type": "number"},
        "is_national_capital": {"type": "boolean"},
    },
    "required": ["city", "population_millions", "is_national_capital"],
    "additionalProperties": False,
}
CITIES = [
    ("Tokyo", True),
    ("Osaka", False),
    ("Paris", True),
    ("Lyon", False),
    ("Berlin", True),
    ("Munich", False),
]
# Osaka, Lyon and Munich are regional capitals: the schema asks for the national one.


def image_url(red_left, size):
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


def document(idx, lines):
    rnd = random.Random(idx)
    body = "".join(
        f"Record {idx}-{i}: inspection of unit {rnd.randint(100, 999)} on floor {rnd.randint(1, 40)} "
        f"found {rnd.choice(['no issues', 'minor wear', 'a loose cable', 'dust buildup'])}.\n"
        for i in range(lines)
    )
    return body, CODES[idx % len(CODES)]


class Client:
    def __init__(self, url, model, timeout):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.model, self.timeout = (
            u.hostname,
            u.port or 80,
            model,
            timeout,
        )

    def chat(self, messages, max_tokens, *, think=False, abort_after=None, **extra):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": think,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        body.setdefault("temperature", 0)
        body.update(extra)
        r = {
            "content": "",
            "reasoning": "",
            "tools": {},
            "finish": None,
            "usage": None,
            "first": None,
            "chunks": 0,
        }
        t0 = time.perf_counter()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            r["status"] = resp.status
            if resp.status != 200:
                r["error"] = resp.read().decode(errors="replace")[:400]
                return r
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("error"):
                    r["error"] = str(ev["error"])[:400]
                r["usage"] = ev.get("usage") or r["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if (d.get("content") or d.get("tool_calls")) and r["first"] is None:
                        r["first"] = time.perf_counter() - t0
                    r["content"] += d.get("content") or ""
                    r["reasoning"] += (
                        d.get("reasoning_content") or d.get("reasoning") or ""
                    )
                    for tc in d.get("tool_calls") or []:
                        slot = r["tools"].setdefault(
                            tc.get("index", 0), {"name": "", "arguments": ""}
                        )
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if d.get("content"):
                        r["chunks"] += 1
                    if ch.get("finish_reason"):
                        r["finish"] = ch["finish_reason"]
                if abort_after is not None and r["chunks"] >= abort_after:
                    r["aborted"] = True
                    break
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)[:400]
        finally:
            conn.close()
        r["wall"] = time.perf_counter() - t0
        return r


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--idle-every-min", type=float, default=10)
    ap.add_argument("--idle-s", type=float, default=60)
    ap.add_argument("--final-idle-s", type=float, default=120)
    ap.add_argument("--footprint-stop-gib", type=float, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--note", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")
    lock = threading.Lock()
    stop = threading.Event()

    def emit(row):
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    def mem():
        try:
            m = process_tree_memory(a.pid)
            return round(m["physical_footprint_sum_bytes"] / 2**30, 3)
        except Exception:  # noqa: BLE001
            return None

    def sampler():
        while not stop.is_set():
            emit({"kind": "mem", "t": round(time.time(), 1), "footprint_gib": mem()})
            stop.wait(5)

    rnd = random.Random(a.seed)
    cli = Client(a.url, a.model, 1200)
    docs = [document(i, rnd.choice([80, 160, 320, 640, 960])) for i in range(12)]
    threads = [[{"role": "system", "content": "You are concise."}] for _ in range(3)]
    thread_codes = [None, None, None]
    kinds = (
        ["doc"] * 25
        + ["chat"] * 20
        + ["image"] * 12
        + ["tool"] * 10
        + ["schema"] * 8
        + ["think"] * 10
        + ["disconnect"] * 7
        + ["long"] * 8
    )
    start_mem = mem()
    emit(
        {
            "kind": "meta",
            "url": a.url,
            "model": a.model,
            "pid": a.pid,
            "minutes": a.minutes,
            "seed": a.seed,
            "note": a.note,
            "start_footprint_gib": start_mem,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    threading.Thread(target=sampler, daemon=True).start()
    t_end = time.time() + a.minutes * 60
    next_idle = time.time() + a.idle_every_min * 60
    n = 0
    last_note = time.time()
    while time.time() < t_end:
        if time.time() - last_note >= 60:  # progress on stdout (queue stall detection)
            last_note = time.time()
            print(f"soak {n} requests, footprint {mem()} GiB", flush=True)
        if time.time() >= next_idle:
            emit({"kind": "idle", "seconds": a.idle_s, "footprint_gib": mem()})
            time.sleep(a.idle_s)
            next_idle = time.time() + a.idle_every_min * 60
        kind = rnd.choice(kinds)
        ok, expect = None, None
        if kind == "doc":
            i = rnd.randrange(len(docs))
            body, code = docs[i]
            r = cli.chat(
                [
                    {
                        "role": "user",
                        "content": body
                        + f"\nThe access code for this archive is {code}. "
                        "What is the access code? Reply with the code only.",
                    }
                ],
                16,
            )
            expect, ok = code, code in r["content"].upper()
        elif kind == "chat":
            j = rnd.randrange(3)
            if thread_codes[j] is None or rnd.random() < 0.3:
                thread_codes[j] = rnd.choice(CODES)
                msg = f"Remember: my locker code is now {thread_codes[j]}. Acknowledge briefly."
                threads[j].append({"role": "user", "content": msg})
                r = cli.chat(threads[j], 40)
                ok = r.get("status") == 200 and bool(r["content"].strip())
            else:
                threads[j].append(
                    {
                        "role": "user",
                        "content": "What is my locker code? Reply with the code only.",
                    }
                )
                r = cli.chat(threads[j], 16)
                expect, ok = thread_codes[j], thread_codes[j] in r["content"].upper()
            threads[j].append({"role": "assistant", "content": r["content"] or "OK"})
            if len(threads[j]) > 80:
                threads[j] = threads[j][:1] + threads[j][-40:]
        elif kind == "image":
            left = rnd.random() < 0.5
            size = rnd.choice([256, 512, 768, 1024])
            r = cli.chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url(left, size)},
                            },
                            {
                                "type": "text",
                                "text": "Which half is red? Reply with exactly one word: left or right.",
                            },
                        ],
                    }
                ],
                8,
            )
            expect = "left" if left else "right"
            ok = r["content"].strip(" .\n").lower() == expect
        elif kind == "tool":
            city, days = rnd.choice(CITIES)[0], rnd.randint(1, 7)
            r = cli.chat(
                [
                    {
                        "role": "user",
                        "content": f"Get the weather for {city} for {days} days.",
                    }
                ],
                256,
                tools=[TOOL],
                tool_choice="auto",
            )
            calls = list(r["tools"].values())
            try:
                args = json.loads(calls[0]["arguments"]) if calls else {}
            except Exception:  # noqa: BLE001
                args = {}
            ok = (
                bool(calls)
                and args.get("city", "").lower() == city.lower()
                and args.get("days") == days
            )
        elif kind == "schema":
            city, cap = rnd.choice(CITIES)
            r = cli.chat(
                [{"role": "user", "content": f"Give facts about {city} as JSON."}],
                200,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "c", "schema": SCHEMA},
                },
            )
            try:
                obj = json.loads(r["content"])
                ok = (
                    obj.get("city", "").lower() == city.lower()
                    and obj.get("is_national_capital") is cap
                )
            except Exception:  # noqa: BLE001
                ok = False
        elif kind == "think":
            x, y = rnd.randint(11, 99), rnd.randint(11, 99)
            r = cli.chat(
                [
                    {
                        "role": "user",
                        "content": f"What is {x}*{y}? Answer with just the number.",
                    }
                ],
                1500,
                think=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
            expect = str(x * y)
            ok = expect in r["content"].replace(",", "")
        elif kind == "disconnect":
            r = cli.chat(
                [
                    {
                        "role": "user",
                        "content": "Write a very long story about a lighthouse.",
                    }
                ],
                2000,
                abort_after=rnd.randint(2, 30),
            )
            ok = r.get("aborted", False) or r.get("status") == 200
        else:  # long
            r = cli.chat(
                [
                    {
                        "role": "user",
                        "content": "Write a detailed technical article about how SSDs work.",
                    }
                ],
                1024,
            )
            ok = r.get("status") == 200 and len(r["content"]) > 1000
        n += 1
        u = r.get("usage") or {}
        emit(
            {
                "kind": "req",
                "n": n,
                "type": kind,
                "ok": bool(ok),
                "expect": expect,
                "got": (r["content"] or "")[:80],
                "status": r.get("status"),
                "error": r.get("error"),
                "first_s": r.get("first") and round(r["first"], 3),
                "wall_s": round(r.get("wall", 0), 3),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                ),
                "finish": r.get("finish"),
                "footprint_gib": mem(),
            }
        )
        fp = mem()
        if fp is not None and fp > a.footprint_stop_gib:
            emit(
                {
                    "kind": "abort",
                    "reason": f"footprint {fp} GiB > {a.footprint_stop_gib}",
                }
            )
            break
    time.sleep(a.final_idle_s)
    stop.set()
    rows = [
        json.loads(line) for line in a.output.read_text().splitlines() if line.strip()
    ]
    reqs = [r for r in rows if r.get("kind") == "req"]
    mems = [
        r["footprint_gib"]
        for r in rows
        if r.get("kind") == "mem" and r.get("footprint_gib")
    ]
    by = {}
    for r in reqs:
        by.setdefault(r["type"], []).append(r)
    summary = {
        "kind": "summary",
        "requests": len(reqs),
        "ok": sum(r["ok"] for r in reqs),
        "errors": sum(1 for r in reqs if r.get("error") and r["type"] != "disconnect"),
        "start_footprint_gib": start_mem,
        "max_footprint_gib": max(mems) if mems else None,
        "end_footprint_gib": mem(),
        "per_type": {},
    }
    for k, rs in by.items():
        walls = sorted(r["wall_s"] for r in rs)
        firsts = sorted(r["first_s"] for r in rs if r.get("first_s"))
        summary["per_type"][k] = {
            "n": len(rs),
            "ok": sum(r["ok"] for r in rs),
            "wall_p50": round(statistics.median(walls), 3),
            "wall_p95": round(walls[int(0.95 * (len(walls) - 1))], 3),
            "first_p50": round(statistics.median(firsts), 3) if firsts else None,
        }
    emit(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
