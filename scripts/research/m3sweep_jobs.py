"""Job bodies for scripts/dev/m3sweep (run through the gpuq M3 lane; correctness only, no timing).

    wire   --model M --out F     real server + the SDK wire-contract matrix (tests/unit/wire_clients.py)
    agent  --model M --out F     covaudit session (tool loop, cache growth) + concurrent identity vs solo
    units  --model M... --out F  unit tests that skip on a machine without the small checkpoints

Every subcommand writes one JSON file (rewritten as it goes) that ends with `"complete": true`
and `"pass": bool`; any exception, missing piece or mismatch is a failure (exit 1). The servers
use ports 18990-18996 (covaudit_session.free_port) and are always kill -9'd.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

STOP_CHARS = ["e", "a", "o", " "]
TOOL_DIALECTS = ("chat", "messages", "responses", "ollama_chat")
SCHEMA_DIALECTS = ("chat", "responses", "ollama_chat", "ollama_generate")
STREAM_USAGE_DIALECTS = ("chat", "completions", "messages", "responses")


# Case: (name, run kwargs, dialects). Greedy-independent: invariants only, never exact text.
def cases():
    d_all = (
        "chat",
        "completions",
        "messages",
        "responses",
        "ollama_chat",
        "ollama_generate",
    )
    return [
        ("basic", dict(max_tokens=32), d_all),
        ("truncate", dict(max_tokens=3), d_all),
        ("stop", dict(max_tokens=48, stop=STOP_CHARS), d_all),
        (
            "tool_required",
            dict(tools=True, tool_choice="required", max_tokens=200),
            ("chat", "messages", "responses"),
        ),
        (
            "tool_named",
            dict(tools=True, tool_choice="get_weather", max_tokens=200),
            ("chat", "messages", "responses"),
        ),
        (
            "tool_none",
            dict(tools=True, tool_choice="none", max_tokens=32),
            ("chat", "messages", "responses"),
        ),
        ("tool_auto", dict(tools=True, max_tokens=200), TOOL_DIALECTS),
        (
            "tool_serial",
            dict(tools=True, tool_choice="required", parallel=False, max_tokens=200),
            ("chat", "messages", "responses"),
        ),
        ("schema", dict(schema=True, max_tokens=64), SCHEMA_DIALECTS),
    ]


def _ok_json(text, key="a"):
    try:
        v = json.loads(text)
    except ValueError:
        return False
    return isinstance(v, dict) and isinstance(v.get(key), int)


def check_case(name, kw, dialect, stream, out):
    """Problems (strings) in one dialect result; pure, unit-tested with fake `Out`s."""
    bad = []
    mt = kw.get("max_tokens")
    if out.finish not in ("stop", "length", "tool_calls"):
        bad.append(f"finish {out.finish!r}")
    for f in ("prompt", "completion"):
        v = getattr(out, f)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            bad.append(f"usage.{f}={v!r}")
    if isinstance(out.prompt, int) and out.prompt <= 0:
        bad.append("prompt tokens 0")
    if isinstance(out.completion, int) and mt is not None and out.completion > mt:
        bad.append(f"completion {out.completion} > max_tokens {mt}")
    if out.reasoning is not None and isinstance(out.completion, int):
        if out.reasoning > out.completion:
            bad.append(f"reasoning {out.reasoning} > completion {out.completion}")
    if (
        out.cached is not None
        and isinstance(out.prompt, int)
        and out.cached > out.prompt
    ):
        bad.append(f"cached {out.cached} > prompt {out.prompt}")
    empty = not (out.text or out.thinking or out.tools)
    if name == "basic" and empty:
        bad.append("empty answer")
    if name == "truncate":
        if out.finish != "length":
            bad.append(f"max_tokens=3 finished {out.finish!r}, want length")
        if out.completion not in (None, 3) and isinstance(out.completion, int):
            if out.completion != 3:
                bad.append(f"completion {out.completion} != 3")
    if name == "stop" and out.finish == "stop":
        hit = [s for s in STOP_CHARS if s in out.text]
        if hit:
            bad.append(f"stop sequence leaked into text: {hit} {out.text[:40]!r}")
    if name in ("tool_required", "tool_named", "tool_serial"):
        if not out.tools:
            bad.append("forced tool call missing")
        for n, a in out.tools:
            if (
                n != "get_weather"
                or not isinstance(a, dict)
                or not isinstance(a.get("city"), str)
            ):
                bad.append(f"bad tool call {n!r} {a!r}")
        if out.tools and out.finish != "tool_calls":
            bad.append(f"tool call finished {out.finish!r}")
        if name == "tool_serial" and len(out.tools) > 1:
            bad.append(f"parallel_tool_calls=false gave {len(out.tools)} calls")
    if name == "tool_none" and out.tools:
        bad.append("tool_choice none produced a tool call")
    if name == "tool_auto":
        for n, a in out.tools:
            if n != "get_weather" or not isinstance(a, dict):
                bad.append(f"bad tool call {n!r} {a!r}")
    if name == "schema" and out.finish == "stop" and not _ok_json(out.text):
        bad.append(f"schema answer is not {{a:int}} JSON: {out.text[:60]!r}")
    return bad


def compare_stream(name, dialect, a, b):
    """Stream and non-stream runs of the same request must count the prompt alike."""
    if dialect not in STREAM_USAGE_DIALECTS or name in ("tool_auto",):
        return []
    if a.prompt is not None and b.prompt is not None and a.prompt != b.prompt:
        return [f"prompt tokens stream {b.prompt} != non-stream {a.prompt}"]
    return []


ERROR_REQUESTS = [
    (
        "chat unknown model",
        "/v1/chat/completions",
        {"model": "nope-xyz", "messages": [{"role": "user", "content": "hi"}]},
        "openai",
    ),
    ("chat no messages", "/v1/chat/completions", {"model": "{m}"}, "openai"),
    (
        "chat bad max_tokens",
        "/v1/chat/completions",
        {
            "model": "{m}",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": -5,
        },
        "openai",
    ),
    (
        "messages unknown model",
        "/v1/messages",
        {
            "model": "nope-xyz",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
        "anthropic",
    ),
    (
        "messages no max_tokens",
        "/v1/messages",
        {"model": "{m}", "messages": [{"role": "user", "content": "hi"}]},
        "anthropic",
    ),
    (
        "responses unknown model",
        "/v1/responses",
        {"model": "nope-xyz", "input": "hi"},
        "openai",
    ),
    ("completions no prompt", "/v1/completions", {"model": "{m}"}, "openai"),
]


def check_error(label, status, body, shape):
    if not 400 <= status < 500:
        return [f"{label}: status {status}, want 4xx"]
    if not isinstance(body, dict) or not isinstance(body.get("error"), (dict, str)):
        return [f"{label}: body without error object: {str(body)[:80]}"]
    e = body["error"]
    if shape == "anthropic":
        if (
            body.get("type") != "error"
            or not isinstance(e, dict)
            or not e.get("type")
            or not e.get("message")
        ):
            return [f"{label}: not the Anthropic error shape: {str(body)[:100]}"]
    elif isinstance(e, dict) and not e.get("message"):
        return [f"{label}: error without message: {str(body)[:100]}"]
    return []


def write(path, d):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))


def start_server(model, label, sets=()):
    from covaudit_session import Srv

    home = Path(os.environ.get("HOME", "/tmp")) / f"m3sweep-{label}"
    cand = Path(sys.executable).parent / "yunshu"
    if "COVAUDIT_BIN" not in os.environ and cand.exists():
        os.environ["COVAUDIT_BIN"] = str(cand)
    log = home / "server.log"
    srv = Srv(model, str(ROOT / "python"), home, log, list(sets))
    srv.wait_ready()
    return srv


def cmd_wire(a):
    import anthropic
    import httpx
    import openai

    sys.path.insert(0, str(ROOT))
    from tests.unit import wire_clients as wc

    res = {
        "kind": "wire",
        "model": a.model,
        "rows": [],
        "failures": [],
        "complete": False,
    }
    srv = None
    try:
        srv = start_server(a.model, "wire", ["YUNSHU_VLM_APC_DISK=0"])
        wc.MODEL = srv.model_id
        http = httpx.Client(base_url=srv.url, timeout=300)
        cl = wc.Clients.__new__(wc.Clients)
        cl.http = http
        cl.oa = openai.OpenAI(
            base_url=srv.url + "/v1", api_key="x", max_retries=0, timeout=300
        )
        cl.an = anthropic.Anthropic(
            base_url=srv.url, api_key="x", max_retries=0, timeout=300
        )
        for name, kw, dialects in cases():
            for dialect in dialects:
                outs = {}
                for stream in (False, True):
                    tag = f"{name}/{dialect}/{'stream' if stream else 'json'}"
                    try:
                        o = wc.run(cl, dialect, stream=stream, **kw)
                        outs[stream] = o
                        bad = check_case(name, kw, dialect, stream, o)
                        row = {
                            "case": tag,
                            "finish": o.finish,
                            "prompt": o.prompt,
                            "completion": o.completion,
                            "tools": len(o.tools),
                        }
                    except Exception as e:  # noqa: BLE001  fail closed, keep going
                        bad = [f"{type(e).__name__}: {str(e)[:200]}"]
                        row = {"case": tag}
                    row["problems"] = bad
                    res["rows"].append(row)
                    res["failures"] += [f"{tag}: {b}" for b in bad]
                    print(f"{tag} {'FAIL ' + str(bad) if bad else 'ok'}", flush=True)
                if len(outs) == 2:
                    for b in compare_stream(name, dialect, outs[False], outs[True]):
                        res["failures"].append(f"{name}/{dialect}: {b}")
                        print(f"{name}/{dialect} FAIL {b}", flush=True)
            write(a.out, res)
        for label, path, body, shape in ERROR_REQUESTS:
            body = json.loads(json.dumps(body).replace("{m}", srv.model_id))
            headers = {"x-api-key": "x", "anthropic-version": "2023-06-01"}
            try:
                r = http.post(path, json=body, headers=headers)
                try:
                    j = r.json()
                except ValueError:
                    j = r.text
                bad = check_error(label, r.status_code, j, shape)
            except Exception as e:  # noqa: BLE001
                bad = [f"{label}: {type(e).__name__}: {e}"]
            res["failures"] += bad
            print(f"error {label} {'FAIL ' + str(bad) if bad else 'ok'}", flush=True)
        # the server must still be alive and answering after every case
        if srv.proc.poll() is not None:
            res["failures"].append(f"server died rc={srv.proc.returncode}")
        elif httpx.get(srv.url + "/health/ready", timeout=10).status_code != 200:
            res["failures"].append("server not ready after the matrix")
    except BaseException as e:  # noqa: BLE001
        res["failures"].append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if srv:
            res["server_log_tail"] = srv.log_tail(30)
            srv.kill()
    res["complete"] = True
    res["pass"] = not res["failures"] and len(res["rows"]) > 0
    write(a.out, res)
    print(
        "RESULT",
        "PASS" if res["pass"] else "FAIL",
        len(res["failures"]),
        "failures",
        flush=True,
    )
    for f in res["failures"]:
        print("FAIL:", f, flush=True)
    return 0 if res["pass"] else 1


def run_sub(cmd, label):
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    tail = (p.stdout + p.stderr)[-1500:]
    print(tail, flush=True)
    return {"step": label, "rc": p.returncode, "tail": tail}


def cmd_agent(a):
    res = {
        "kind": "agent",
        "model": a.model,
        "steps": [],
        "failures": [],
        "complete": False,
    }
    out = Path(a.out)
    d = out.parent / (out.stem + "-work")
    d.mkdir(parents=True, exist_ok=True)
    py, src = sys.executable, str(ROOT / "python")
    cand = Path(sys.executable).parent / "yunshu"
    if "COVAUDIT_BIN" not in os.environ and cand.exists():
        os.environ["COVAUDIT_BIN"] = str(cand)
    home = os.environ.get("HOME", str(d))
    steps = [
        (
            "session-messages",
            [
                py,
                str(HERE / "covaudit_session.py"),
                "run",
                "--model",
                a.model,
                "--src",
                src,
                "--out",
                str(d / "session.jsonl"),
                "--home",
                f"{home}/m3sweep-sess",
                "--turns",
                "3",
                "--file-tokens",
                "1200",
                "--max-tokens",
                "160",
            ],
        ),
        (
            "conc-identity",
            [
                py,
                str(HERE / "covaudit_conc.py"),
                "run",
                "--model",
                a.model,
                "--src",
                src,
                "--out",
                str(d / "conc.jsonl"),
                "--long-tokens",
                "2500",
                "--mid-tokens",
                "800",
            ],
        ),
    ]
    for label, cmd in steps:
        s = run_sub(cmd, label)
        res["steps"].append(s)
        if s["rc"] != 0:
            res["failures"].append(f"{label} rc={s['rc']}")
        write(a.out, res)
    res["complete"] = True
    res["pass"] = not res["failures"]
    write(a.out, res)
    print("RESULT", "PASS" if res["pass"] else "FAIL", flush=True)
    return 0 if res["pass"] else 1


UNIT_FILES = [
    "tests/unit/test_model_card.py",
    "tests/unit/test_model_card_gateway.py",
    "tests/unit/test_json_schema_llguidance.py",
    "tests/unit/test_tool_call_grammar.py",
    "tests/unit/test_cfg_constraint_support.py",
    "tests/unit/test_json_engine_choice.py",
]


def parse_pytest_summary(text):
    """(passed, failed, skipped) from pytest's last line; None when there is no summary."""
    import re

    last = [
        ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)
    ]
    if not last:
        return None
    g = lambda k: int(m.group(1)) if (m := re.search(rf"(\d+) {k}", last[-1])) else 0  # noqa: E731
    return g("passed"), g("failed") + g("error"), g("skipped")


def skipped_models(text, models):
    """SKIPPED lines that name a checkpoint the job supplied (a skip there is a lane defect)."""
    names = [Path(m).name for m in models]
    return [
        ln[:160]
        for ln in text.splitlines()
        if ln.startswith("SKIPPED") and any(n in ln for n in names)
    ]


def cmd_units(a):
    res = {"kind": "units", "models": a.model, "failures": [], "complete": False}
    roots = {str(Path(m).parent) for m in a.model}
    if len(roots) != 1:
        raise SystemExit(f"models must share one root: {roots}")
    env = dict(os.environ, YUNSHU_TEST_MODELS=roots.pop())
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "-p",
        "no:cacheprovider",
        *UNIT_FILES,
    ]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    text = p.stdout + p.stderr
    print(text[-4000:], flush=True)
    summ = parse_pytest_summary(text)
    res["summary"] = summ
    res["tail"] = text[-3000:]
    missed = skipped_models(text, a.model)
    if missed:
        res["failures"].append(
            f"tests skipped for a model the sweep supplied: {missed}"
        )
    if p.returncode != 0 or summ is None or summ[1]:
        res["failures"].append(f"pytest rc={p.returncode} summary={summ}")
    elif summ[0] == 0:
        res["failures"].append("no test ran")
    res["complete"] = True
    res["pass"] = not res["failures"]
    write(a.out, res)
    print("RESULT", "PASS" if res["pass"] else "FAIL", flush=True)
    return 0 if res["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("wire", "agent", "units"):
        p = sub.add_parser(n)
        p.add_argument("--out", required=True)
        p.add_argument(
            "--model",
            required=(n != "units"),
            action="append" if n == "units" else "store",
        )
    a = ap.parse_args(argv)
    return {"wire": cmd_wire, "agent": cmd_agent, "units": cmd_units}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
