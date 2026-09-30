"""Replay captured agent request bodies and grade every tool call.

For each body (an Anthropic /v1/messages request JSON, e.g. a Claude Code turn)
the script sends it ``--n`` times to a fresh Yunshu server (streaming, like the
agent does) and records, per reply:

- ``stop_reason`` and the block types;
- ``malformed``: a tool_use whose name is not a request tool, whose input does
  not validate against that tool's ``input_schema``, or that has no input at all;
- ``leaked``: call markup (``<tool_call>``, ``<function=``, ``<parameter=``,
  ``</function>``) in a text or thinking block that is not a tool_use;
- ``dropped``: the model wrote a tool-call start marker but no tool_use came out
  (counted from the server log, ``tool call unreadable`` / dropped warnings, and
  from the reply: marker in the raw text but stop_reason != tool_use);
- decode speed: output tokens after the first / time after the first.

The full replies are written to ``--out`` (JSONL) so two runs can be diffed for
token-for-token identity (greedy: ``--temperature 0``).

    tool_grammar_replay.py --checkpoint $M --bodies b1.json b2.json --n 8 \\
        --label grammar-on --env YUNSHU_TOOL_GRAMMAR=1 --out runs/x.jsonl

Server: own session, isolated HOME, port from 18990-18999, killed with SIGKILL
on exit.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MARKERS = ("<tool_call>", "</tool_call>", "<function=", "<parameter=", "</function>")
PORTS = range(18990, 19000)


def free_port() -> int:
    for port in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port in 18990-18999")


class Server:
    def __init__(self, checkpoint: str, src: str | None, env: dict, log: Path):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = log
        py = os.environ.get("YUNSHU_PY")
        binary = os.environ.get("YUNSHU_BIN") or (
            str(Path(py).with_name("yunshu")) if py else "yunshu"
        )
        home = Path(
            os.environ.get("REPLAY_HOME")
            or Path(tempfile.gettempdir()) / "yunshu-replay-home"
        )
        home.mkdir(parents=True, exist_ok=True)
        e = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE", "CODEX", "YUNSHU_"))
        }
        e.update(HOME=str(home), HF_HUB_OFFLINE="1", NO_PROXY="127.0.0.1", **env)
        if src:
            e["PYTHONPATH"] = src + os.pathsep + e.get("PYTHONPATH", "")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [binary, "serve", "-m", checkpoint, "--port", str(self.port)],
            stdout=log.open("ab"),
            stderr=subprocess.STDOUT,
            env=e,
        )
        # A stopped / timed-out job must not leave the server holding the GPU.
        atexit.register(self.kill)
        signal.signal(signal.SIGTERM, lambda *_: (self.kill(), sys.exit(143)))
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early rc={self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=3) as r:
                    if r.status == 200:
                        self.model = json.load(r)["data"][0]["id"]
                        return
            except Exception:
                time.sleep(2)
        raise RuntimeError("server not ready")

    def kill(self):
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=30)


def stream_reply(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url + "/v1/messages",
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "x-api-key": "k",
            "anthropic-version": "2023-06-01",
        },
    )
    blocks: dict[int, dict] = {}
    stop = None
    out_tokens = 0
    t0 = time.perf_counter()
    t_first = t_last = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        event = None
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                data = json.loads(line[5:])
                if event == "content_block_start":
                    cb = data["content_block"]
                    blocks[data["index"]] = {
                        "type": cb["type"],
                        "text": "",
                        "input": "",
                        "name": cb.get("name"),
                    }
                elif event == "content_block_delta":
                    d = data["delta"]
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    b = blocks[data["index"]]
                    b["text"] += d.get("text", "") + d.get("thinking", "")
                    b["input"] += d.get("partial_json", "")
                elif event == "message_delta":
                    stop = data["delta"].get("stop_reason")
                    out_tokens = (data.get("usage") or {}).get(
                        "output_tokens", out_tokens
                    )
                elif event == "error":
                    return {"error": data}
    return {
        "blocks": [blocks[i] for i in sorted(blocks)],
        "stop": stop,
        "out_tokens": out_tokens,
        "s": round(time.perf_counter() - t0, 2),
        "ttft_s": round((t_first or t0) - t0, 2),
        "decode_s": round((t_last - t_first), 3) if t_first and t_last else 0.0,
    }


def grade(reply: dict, tools: list[dict]) -> dict:
    import jsonschema

    by_name = {t["name"]: t for t in tools}
    malformed = leaked = 0
    calls = []
    for b in reply.get("blocks", []):
        if b["type"] == "tool_use":
            problems = []
            tool = by_name.get(b["name"])
            try:
                args = json.loads(b["input"] or "{}")
            except ValueError:
                args = None
                problems.append("input is not JSON")
            if tool is None:
                problems.append("unknown tool")
            elif isinstance(args, dict):
                try:
                    jsonschema.validate(args, tool.get("input_schema") or {})
                except jsonschema.ValidationError as exc:
                    problems.append("schema: " + exc.message[:80])
            elif args is not None:
                problems.append("input is not an object")
            if problems:
                malformed += 1
            calls.append({"name": b["name"], "problems": problems})
        elif any(m in b["text"] for m in MARKERS):
            leaked += 1
    text_marker = any(
        "<tool_call>" in b["text"]
        for b in reply.get("blocks", [])
        if b["type"] != "tool_use"
    )
    dropped = int(text_marker and reply.get("stop") != "tool_use")
    tps = None
    if reply.get("decode_s") and reply.get("out_tokens", 0) > 1:
        tps = round((reply["out_tokens"] - 1) / reply["decode_s"], 2)
    return {
        "malformed": malformed,
        "leaked": leaked,
        "dropped": dropped,
        "calls": calls,
        "decode_tps": tps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", help="python/ directory to run instead of the checkout")
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE for the server")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--log", default=str(Path(tempfile.gettempdir()) / "replay-server.log")
    )
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    srv = Server(a.checkpoint, a.src, env, Path(a.log))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    totals = dict(replies=0, tool_use=0, malformed=0, leaked=0, dropped=0, errors=0)
    speeds = []
    try:
        with out.open("a") as fh:
            for f in a.bodies:
                body = json.loads(Path(f).read_text())
                body["stream"] = True
                if a.temperature is not None:
                    body["temperature"] = a.temperature
                if a.max_tokens is not None:
                    body["max_tokens"] = a.max_tokens
                for i in range(a.n):
                    reply = stream_reply(srv.url, body)
                    if "error" in reply:
                        totals["errors"] += 1
                        print(Path(f).name, i, "ERROR", reply["error"], flush=True)
                        continue
                    g = grade(reply, body["tools"])
                    totals["replies"] += 1
                    totals["tool_use"] += reply["stop"] == "tool_use"
                    for k in ("malformed", "leaked", "dropped"):
                        totals[k] += g[k]
                    if g["decode_tps"]:
                        speeds.append(g["decode_tps"])
                    row = dict(
                        label=a.label, body=Path(f).name, i=i, reply=reply, grade=g
                    )
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(
                        Path(f).name,
                        i,
                        reply["stop"],
                        [b["type"] for b in reply["blocks"]],
                        f"out={reply['out_tokens']} tps={g['decode_tps']} ttft={reply['ttft_s']}s",
                        f"malformed={g['malformed']} leaked={g['leaked']} dropped={g['dropped']}",
                        flush=True,
                    )
        speeds.sort()
        med = speeds[len(speeds) // 2] if speeds else None
        print(
            "SUMMARY", a.label, json.dumps(totals), "decode_tps_median", med, flush=True
        )
    finally:
        srv.kill()


if __name__ == "__main__":
    sys.exit(main())
