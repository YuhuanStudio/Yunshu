"""Long Claude Code-style tool session against one server arm (coverage audit).

The conversation is deterministic except for the model's own replies: the user asks to Read
`--turns` files one call per turn (each file is `--file-tokens` tokens with a needle) and to
finish with `CODES: ...`. Every request echoes the model's previous reply (text + tool_use, as a
client does) and appends the next tool_result, so request k+1 extends request k exactly like a real
agent loop. Checks per request: a valid `Read` tool_use for the expected file (parsed JSON), no
tool markup in text, a complete stream; on the last request all needles recalled; and the
cached tokens grow (request k+1 reuses >= 85% of request k's prompt).

    run      --model M --src WORKTREE/python --out arm.jsonl [--turns 6] [--file-tokens 8000]
             [--replay bodies.jsonl] [--only-last]   # replay: send exactly these request bodies
    restart  --model M --src ... --out r.json         # SSD tier + idle: round trip across a SIGTERM
    compare  a.jsonl b.jsonl                          # outputs equal request by request
    judge    arm.jsonl

The server runs with an isolated HOME (empty APC disk tier), is always kill -9'd, uses a port in
18990-18996, and a failed validation exits nonzero (fail closed).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MAIN = Path(
    os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
)
WORDS = [
    "amber",
    "birch",
    "cedar",
    "delta",
    "ember",
    "frost",
    "glint",
    "hazel",
    "ivory",
    "jade",
    "kelp",
    "lotus",
    "maple",
    "nectar",
    "onyx",
    "pearl",
    "quartz",
    "raven",
    "sage",
    "tulip",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
]

TOOLS = [
    {
        "name": "Bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file from the working tree.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace a string in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]
SYSTEM = (
    "You are a coding agent working in a repository through tools. Read files with Read, run "
    "commands with Bash, change code with Edit. Be concise. Follow the user's instructions exactly.\n"
    + "".join(
        f"Rule {i}: keep changes minimal, keep the existing style, never invent file contents.\n"
        for i in range(1, 60)
    )
)


def needle(i: int) -> str:
    return f"{WORDS[i % len(WORDS)].upper()}-{(i * 7919) % 9000 + 1000}"


def file_text(i: int, tokens: int) -> str:
    """Deterministic python-ish module of about `tokens` tokens (3 chars per token) with a needle."""
    rng = random.Random(1000 + i)
    lines = [f'SECRET_CODE = "{needle(i)}"  # module {i}']
    n = 0
    while n < tokens * 3:
        a, b = rng.randrange(100), rng.choice(WORDS)
        ln = f"def {b}_{rng.randrange(10**6)}(x, y):\n    return (x * {a}) + y  # {b} {a}"
        lines.append(ln)
        n += len(ln) + 1
    return "\n".join(lines)


def path_of(i: int) -> str:
    return f"src/pkg/mod_{i}.py"


def first_body(
    n_files: int, model: str, max_tokens: int = 200, thinking: bool = False
) -> dict:
    ask = (
        f"Read the files {path_of(1)} through {path_of(n_files)} with the Read tool, one call per turn "
        "and nothing else. After the last file reply with one line `CODES: ` followed by the "
        "SECRET_CODE of each file in order, separated by spaces."
    )
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "system": SYSTEM,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": ask}],
    }
    if (
        not thinking
    ):  # a client that leaves thinking on (Codex, Claude Code default) omits the key
        body["thinking"] = {"type": "disabled"}
    return body


def next_body(prev: dict, reply: dict, k: int, file_tokens: int) -> dict:
    """Request k+1 = request k + the model's own reply (as a client echoes it) + file k's text."""
    blocks = []
    if reply["text"].strip():
        blocks.append({"type": "text", "text": reply["text"]})
    call = next((c for c in reply["tool_calls"] if c["input"] is not None), None)
    if (
        call is None
    ):  # no usable tool call: the script's own call keeps the session going (judge fails it)
        call = {"id": f"toolu_s{k}", "name": "Read", "input": {"file_path": path_of(k)}}
    tid = call.get("id") or f"toolu_{k}"
    blocks.append(
        {"type": "tool_use", "id": tid, "name": call["name"], "input": call["input"]}
    )
    result = {
        "type": "tool_result",
        "tool_use_id": tid,
        "content": file_text(k, file_tokens),
    }
    return dict(
        prev,
        messages=prev["messages"]
        + [
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": [result]},
        ],
    )


def parse_sse(lines) -> dict:
    """Fold an Anthropic SSE stream into text, tool_calls, stop_reason, usage. Pure (unit-tested)."""
    text, tools, usage, stop, ended, thinking = "", {}, {}, None, False, ""
    ev = None
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.rstrip("\r\n")
        if raw.startswith("event:"):
            ev = raw[6:].strip()
            continue
        if not raw.startswith("data:"):
            continue
        d = json.loads(raw[5:])
        t = d.get("type", ev)
        if t == "message_start":
            usage.update(d["message"].get("usage") or {})
        elif t == "content_block_start":
            b = d["content_block"]
            if b["type"] == "tool_use":
                tools[d["index"]] = {"name": b["name"], "id": b.get("id"), "json": ""}
        elif t == "content_block_delta":
            x = d["delta"]
            if x["type"] == "text_delta":
                text += x["text"]
            elif x["type"] == "thinking_delta":
                thinking += x.get("thinking", "")
            elif x["type"] == "input_json_delta":
                tools[d["index"]]["json"] += x["partial_json"]
        elif t == "message_delta":
            stop = d["delta"].get("stop_reason") or stop
            usage.update(d.get("usage") or {})
        elif t == "message_stop":
            ended = True
        elif t == "error":
            raise RuntimeError(f"stream error event: {d}")
    calls = []
    for i in sorted(tools):
        try:
            args = json.loads(tools[i]["json"] or "{}")
        except ValueError:
            args = None
        calls.append(
            {
                "name": tools[i]["name"],
                "id": tools[i]["id"],
                "input": args,
                "raw": tools[i]["json"],
            }
        )
    return {
        "text": text,
        "thinking": thinking,
        "tool_calls": calls,
        "stop_reason": stop,
        "usage": usage,
        "ended": ended,
    }


def total_prompt(u: dict) -> int:
    return sum(
        int(u.get(k) or 0)
        for k in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )


def judge_rows(rows: list, turns: int, require_cache: bool = True) -> list:
    """Reasons an arm's rows fail (empty = PASS). Request k (1..turns) must call Read on file k,
    request turns+1 must answer with every needle. Pure (unit-tested)."""
    bad = []
    by = {r["req"]: r for r in rows if r.get("kind") == "req"}
    if not by:
        return ["no request rows"]
    for k, r in sorted(by.items()):
        tag = f"req{k}"
        if not r.get("ended"):
            bad.append(f"{tag}: stream did not end with message_stop")
            continue
        if "<tool_call>" in r["text"] or "<function=" in r["text"]:
            bad.append(f"{tag}: tool markup leaked into text")
        if k <= turns:
            ok = [
                c
                for c in r["tool_calls"]
                if c["name"] == "Read"
                and isinstance(c["input"], dict)
                and path_of(k) in str(c["input"].get("file_path"))
            ]
            if not ok:
                bad.append(
                    f"{tag}: no valid Read tool_use for {path_of(k)}: {r['tool_calls'][:1]} text={r['text'][:60]!r}"
                )
            if r["stop_reason"] != "tool_use":
                bad.append(f"{tag}: stop_reason {r['stop_reason']}")
        else:
            miss = [
                needle(i) for i in range(1, turns + 1) if needle(i) not in r["text"]
            ]
            if miss:
                bad.append(f"{tag}: needles missing {miss}: {r['text'][:120]!r}")
    if require_cache:
        for k in range(2, max(by) + 1):
            if k in by and k - 1 in by:
                want = int(0.85 * total_prompt(by[k - 1]["usage"]))
                got = int(by[k]["usage"].get("cache_read_input_tokens") or 0)
                if got < want:
                    bad.append(
                        f"req{k}: cached {got} < {want} (85% of request {k - 1}'s prompt)"
                    )
    return bad


def _calls(r: dict) -> list:
    """Tool calls without the random ids (and anything else a tool stamps on the row)."""
    return [(c["name"], c.get("raw")) for c in r["tool_calls"]]


def compare_rows(a: list, b: list) -> list:
    """Mismatches between two arms' outputs on the requests both have."""
    ia = {r["req"]: r for r in a if r.get("kind") == "req"}
    ib = {r["req"]: r for r in b if r.get("kind") == "req"}
    keys = sorted(set(ia) & set(ib))
    out = []
    for k in keys:
        x, y = ia[k], ib[k]
        if (x["text"], _calls(x)) != (y["text"], _calls(y)):
            out.append(
                f"req{k}: {x['text'][:60]!r}/{x['tool_calls'][:1]} vs {y['text'][:60]!r}/{y['tool_calls'][:1]}"
            )
    if not keys:
        out.append("no common requests")
    return out


def free_port() -> int:
    for p in range(18990, 18997):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        return p
    raise RuntimeError("no free port in 18990-18996")


def load_timeout(model: str) -> float:
    """Readiness budget: 90 s + 6 s per GiB of weight files (a 58 GiB mmap model: ~440 s)."""
    try:
        gib = sum(f.stat().st_size for f in Path(model).glob("*.safetensors")) / 2**30
    except OSError:
        gib = 0.0
    return 90 + 6 * gib


def load_failure(log_tail: str) -> str | None:
    """The first line of a server log that says the model did not load (pure; unit-tested)."""
    for ln in log_tail.splitlines():
        if "FATAL" in ln or "load failed" in ln or ln.startswith("Traceback"):
            return ln[:300]
    return None


class Srv:
    def __init__(
        self,
        model: str,
        src: str | None,
        home: Path,
        log: Path,
        sets=(),
        models_dir: str | None = None,
        token: str | None = None,
    ):
        """`models_dir` serves in multi-model mode (`--models-dir`; `model` then only sizes the
        load-time budget); `token` is the bearer token the readiness probe presents."""
        self.token = token
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = log
        self.model_path = model
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE", "CODEX"))
        }
        home.mkdir(parents=True, exist_ok=True)
        env.update(HOME=str(home), HF_HUB_OFFLINE="1", NO_PROXY="127.0.0.1")
        if src:
            env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            os.environ.get("COVAUDIT_BIN") or str(MAIN / ".venv/bin/yunshu"),
            "serve",
            *(["--models-dir", models_dir] if models_dir else ["-m", model]),
            "--port",
            str(self.port),
        ]
        for s in sets:
            cmd += ["--set", s]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as lf:
            self.proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

    def log_tail(self, n: int = 50) -> str:
        try:
            return "".join(self.log.read_text(errors="replace").splitlines(True)[-n:])
        except OSError:
            return "(no server log)"

    def wait_ready(self, timeout: float | None = None):
        """Block until the model is loaded and served. Fails at once, with the server's last 50
        log lines, when the process exits or logs a load failure; the timeout defaults to
        90 s + 6 s per GiB of weights; progress lines are printed every 30 s."""
        timeout = timeout or load_timeout(self.model_path)
        t0 = time.monotonic()
        last = 0.0
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited rc={self.proc.returncode}; log tail:\n{self.log_tail()}"
                )
            fatal = load_failure(self.log_tail(200))
            if fatal:
                raise RuntimeError(
                    f"server failed to load the model: {fatal}\n{self.log_tail()}"
                )
            try:
                with urllib.request.urlopen(self.url + "/health/ready", timeout=3) as r:
                    ready = bool(json.load(r).get("ready"))
                mreq = urllib.request.Request(self.url + "/v1/models")
                if self.token:
                    mreq.add_header("Authorization", f"Bearer {self.token}")
                with urllib.request.urlopen(mreq, timeout=3) as r:
                    data = json.load(r)["data"]
                if ready and data:
                    self.model_id = data[0]["id"]
                    print(
                        f"server ready after {time.monotonic() - t0:.0f}s", flush=True
                    )
                    return
            except Exception:
                pass
            if time.monotonic() - last >= 30:
                last = time.monotonic()
                tail = [x for x in self.log_tail(5).splitlines() if "GET /" not in x][
                    -1:
                ]
                print(
                    f"waiting for the model {last - t0:.0f}s / {timeout:.0f}s: {tail}",
                    flush=True,
                )
            time.sleep(2)
        raise RuntimeError(
            f"server not ready after {timeout:.0f}s; log tail:\n{self.log_tail()}"
        )

    def kill(self):
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=30)

    def stop(self, timeout=180) -> bool:
        """Graceful SIGTERM (what a service stop sends); True when it exited in time."""
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
        return True


API = "messages"  # set from --api: messages (Claude Code) | chat (opencode) | responses (Codex)


def send(url: str, body: dict, timeout=900) -> dict:
    if API != "messages":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from covaudit_wire import send_api

        return send_api(url, body, API, timeout)
    req = urllib.request.Request(
        url + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": "k",
            "anthropic-version": "2023-06-01",
        },
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = parse_sse(r)
    res["wall_s"] = round(time.monotonic() - t0, 2)
    return res


def log_row(out: Path, k: int, res: dict) -> dict:
    row = {"kind": "req", "req": k, "prompt_tokens": total_prompt(res["usage"]), **res}
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"req{k} prompt={row['prompt_tokens']} cached={res['usage'].get('cache_read_input_tokens')} "
        f"stop={res['stop_reason']} wall={res['wall_s']}s text={res['text'][:50]!r} "
        f"tools={[(c['name'], c['input']) for c in res['tool_calls']]}",
        flush=True,
    )
    return row


def cmd_run(a) -> int:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    bodies_path = out.with_suffix(".bodies.jsonl")
    home = Path(a.home or f"/Volumes/P5Plus/yunshu-build/covaudit/home-{out.stem}")
    srv = Srv(a.model, a.src, home, out.with_suffix(".server.log"), a.set)
    rc = 0
    try:
        srv.wait_ready()
        if a.replay:
            seq = [json.loads(x) for x in Path(a.replay).read_text().splitlines()]
            seq = [(i + 1, dict(b, model=srv.model_id)) for i, b in enumerate(seq)]
            if a.only_last:
                seq = seq[-1:]
            for k, body in seq:
                if not log_row(out, k, send(srv.url, body))["ended"]:
                    print("FAIL: stream ended without message_stop", file=sys.stderr)
                    return 2
        else:
            body = first_body(a.turns, srv.model_id, a.max_tokens, a.thinking)
            for k in range(1, a.turns + 2):
                with bodies_path.open("a") as f:
                    f.write(json.dumps(body) + "\n")
                res = send(srv.url, body)
                if not log_row(out, k, res)["ended"]:
                    print("FAIL: stream ended without message_stop", file=sys.stderr)
                    return 2
                if k <= a.turns:
                    body = next_body(body, res, k, a.file_tokens)
    except BaseException as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        rc = 2
    finally:
        srv.kill()
    if rc == 0:
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        bad = judge_rows(rows, a.turns, require_cache=not (a.only_last or a.replay))
        for b in bad:
            print("JUDGE FAIL:", b)
        print("RESULT", "FAIL" if bad else "PASS")
        rc = 1 if bad else 0
    return rc


def judge_restart(p1: dict, idle: dict, p2: dict, stopped: bool, turns: int) -> list:
    """Reasons a restart / idle round trip fails (pure; unit-tested). Each arg is the last request's result."""
    bad = []
    if not stopped:
        bad.append("server did not exit within the SIGTERM grace period")
    for tag, r in (("first", p1), ("after idle", idle), ("after restart", p2)):
        miss = [
            needle(i) for i in range(1, turns + 1) if needle(i) not in r.get("text", "")
        ]
        if not r.get("ended") or miss:
            bad.append(f"{tag}: wrong or incomplete reply {r.get('text', '')[:60]!r}")
    for tag, r, frac in (("after idle", idle, 0.95), ("after restart", p2, 0.7)):
        got, tot = (
            int(r["usage"].get("cache_read_input_tokens") or 0),
            total_prompt(r["usage"]),
        )
        if got < frac * tot:
            bad.append(
                f"{tag}: cached {got} of {tot} prompt tokens (< {int(frac * 100)}%)"
            )
        if r.get("text") != p1.get("text"):
            bad.append(f"{tag}: output differs from the first reply")
    return bad


def cmd_restart(a) -> int:
    import shutil

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    home = Path(f"/Volumes/P5Plus/yunshu-build/covaudit/home-{out.stem}")
    shutil.rmtree(home, ignore_errors=True)
    rc, srv = 2, None
    try:
        srv = Srv(a.model, a.src, home, out.with_suffix(".server1.log"), a.set)
        srv.wait_ready()
        body = first_body(a.turns, srv.model_id, a.max_tokens, a.thinking)
        for k in range(1, a.turns + 2):
            res = send(srv.url, body)
            print(
                f"req{k} prompt={total_prompt(res['usage'])} cached={res['usage'].get('cache_read_input_tokens')} text={res['text'][:40]!r}",
                flush=True,
            )
            if k <= a.turns:
                body = next_body(body, res, k, a.file_tokens)
        p1, last = res, body
        time.sleep(a.idle)
        idle = send(srv.url, last)
        stopped = srv.stop()
        srv.kill()
        srv = Srv(a.model, a.src, home, out.with_suffix(".server2.log"), a.set)
        srv.wait_ready()
        p2 = send(srv.url, last)
        print(
            f"after restart cached={p2['usage'].get('cache_read_input_tokens')} prompt={total_prompt(p2['usage'])}",
            flush=True,
        )
        bad = judge_restart(p1, idle, p2, stopped, a.turns)
        out.write_text(json.dumps({"p1": p1, "idle": idle, "p2": p2, "bad": bad}))
        for b in bad:
            print("JUDGE FAIL:", b)
        print("RESULT", "FAIL" if bad else "PASS")
        rc = 1 if bad else 0
    except BaseException as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        if srv:
            srv.kill()
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "restart"):
        r = sub.add_parser(name)
        r.add_argument("--model", required=True)
        r.add_argument("--src")
        r.add_argument("--out", required=True)
        r.add_argument("--home")
        r.add_argument("--turns", type=int, default=6 if name == "run" else 3)
        r.add_argument(
            "--file-tokens", type=int, default=8000 if name == "run" else 6000
        )
        r.add_argument("--set", action="append", default=[])
        r.add_argument("--max-tokens", type=int, default=200)
        r.add_argument("--thinking", action="store_true")
        r.add_argument(
            "--api", choices=["messages", "chat", "responses"], default="messages"
        )
        if name == "run":
            r.add_argument("--replay")
            r.add_argument("--only-last", action="store_true")
        else:
            r.add_argument("--idle", type=float, default=45)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    j = sub.add_parser("judge")
    j.add_argument("a")
    j.add_argument("--turns", type=int, default=6)
    a = ap.parse_args(argv)
    global API
    API = getattr(a, "api", "messages")

    def rd(p):
        return [json.loads(x) for x in Path(p).read_text().splitlines()]

    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "restart":
        return cmd_restart(a)
    if a.cmd == "compare":
        bad = compare_rows(rd(a.a), rd(a.b))
        print("\n".join(bad) or "IDENTICAL")
        return 1 if bad else 0
    bad = judge_rows(rd(a.a), a.turns)
    print("\n".join(bad) or "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
