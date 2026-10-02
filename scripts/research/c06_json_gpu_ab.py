"""Greedy structured-output A/B on a real checkpoint: in-house vs llguidance mask.

Starts one server per engine (YUNSHU_JSON_SCHEMA_ENGINE), sends the same ~30
json_schema chat requests greedy, and reports per-prompt output digests, how many
are identical, and decode time.  Where both masks accept the same tokens the
outputs must be identical; a difference is listed with both texts.

    c06_json_gpu_ab.py --checkpoint $M --src python --out run.json

Servers run in their own process with an isolated HOME, ports 18990-18999, and are
killed with SIGKILL on exit (see tool_grammar_replay.Server).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from c06_json_ab import OPENAI  # noqa: E402
from tool_grammar_replay import Server  # noqa: E402

TASKS = [
    "Make up a realistic example about planning a weekend trip.",
    "Make up a realistic example about a small bakery inventory.",
]
SKIP = {"top_string_enum"}  # the in-house engine cannot enforce a root enum


def requests_for() -> list[tuple[str, str, dict]]:
    out = []
    for name, schema in OPENAI.items():
        if name in SKIP:
            continue
        for i, task in enumerate(TASKS):
            out.append((f"{name}#{i}", task, schema))
    return out


def ask(server, label, task, schema, max_tokens):
    body = {
        "model": server.model,
        "messages": [
            {"role": "user", "content": task + " Answer with JSON only."},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "x", "strict": True, "schema": schema},
        },
    }
    req = urllib.request.Request(
        server.url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    dt = time.perf_counter() - t0
    text = res["choices"][0]["message"]["content"] or ""
    return {
        "label": label,
        "text": text,
        "digest": hashlib.sha256(text.encode()).hexdigest()[:16],
        "completion_tokens": res["usage"]["completion_tokens"],
        "finish": res["choices"][0]["finish_reason"],
        "seconds": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default="c06_gpu.json")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--engines", default="inhouse,llguidance")
    ap.add_argument("--logdir", default="c06-gpu-logs")
    args = ap.parse_args()

    from jsonschema import Draft202012Validator

    reqs = requests_for()
    runs: dict[str, list[dict]] = {}
    for engine in args.engines.split(","):
        log = Path(args.logdir) / f"{engine}.log"
        server = Server(
            args.checkpoint, args.src, {"YUNSHU_JSON_SCHEMA_ENGINE": engine}, log
        )
        try:
            ask(
                server,
                "warm",
                "Say hi.",
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                },
                32,
            )
            rows = []
            for label, task, schema in reqs:
                row = ask(server, label, task, schema, args.max_tokens)
                try:
                    row["valid"] = Draft202012Validator(schema).is_valid(
                        json.loads(row["text"])
                    )
                except ValueError:
                    row["valid"] = False
                rows.append(row)
                print(
                    engine,
                    label,
                    row["digest"],
                    row["completion_tokens"],
                    row["valid"],
                    flush=True,
                )
            runs[engine] = rows
        finally:
            server.kill()
    report = {"runs": runs}
    engines = list(runs)
    if len(engines) == 2:
        a, b = (runs[e] for e in engines)
        same = [x["digest"] == y["digest"] for x, y in zip(a, b, strict=True)]
        report["identical"] = sum(same)
        report["total"] = len(same)
        report["differences"] = [
            {"label": x["label"], engines[0]: x["text"], engines[1]: y["text"]}
            for x, y, s in zip(a, b, same, strict=True)
            if not s
        ]
        for e in engines:
            toks = sum(r["completion_tokens"] for r in runs[e])
            secs = sum(r["seconds"] for r in runs[e])
            report[f"{e}_tok_per_s"] = toks / secs
            report[f"{e}_valid"] = sum(r["valid"] for r in runs[e])
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("runs", "differences")},
            indent=1,
        )
    )
    print(f"differences: {[d['label'] for d in report.get('differences', [])]}")


if __name__ == "__main__":
    main()
