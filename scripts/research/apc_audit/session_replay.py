"""Replay synthetic long agent sessions against a fresh Yunshu server (GPU: run through gpuq).

The system prompt and tool list come from a captured agent request (--template, a Chat
Completions body); every session then grows by assistant tool calls + tool results made of
real source text, the way a coding agent's history grows. Per request it records prompt /
cached tokens, TTFT, wall time and the server's memory.

Scenarios
  multi     N sessions grow round-robin to --target tokens (--step per turn), --gap-s idle
            between rounds, then each session sends one more request (revisit).
  tiers     for each --lengths: a cold request, then the same prompt + a small suffix
            (a cache hit: RAM, or SSD when the RAM tier is too small), greedy text saved.
  restart   for each --lengths: a cold request, a graceful server stop (timed: the shutdown spill),
            a new server on the same disk tier, then the same prompt + suffix (tier ssd).
  identity  the `tiers` requests only, for output comparison across server configurations.

    session_replay.py --checkpoint M --template body.json --scenario multi --sessions 3 \
        --target 30000 --step 3000 --env YUNSHU_VLM_APC_DISK_DIR=/Volumes/P5Plus/tmp/apc \
        --out results.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agentic"))

REPO = HERE.parents[2]


def source_text(session: int, chars: int) -> str:
    """Distinct real source text per session (rotating file order)."""
    files = sorted((REPO / "python").rglob("*.py"))
    files = [f for f in files if f.stat().st_size > 2000]
    rot = (session * 37) % len(files)
    files = files[rot:] + files[:rot]
    out, n = [], 0
    for f in files:
        t = f.read_text(errors="ignore")
        out.append(f"# {f.relative_to(REPO)}\n{t}\n")
        n += len(t)
        if n >= chars:
            break
    return "".join(out)[:chars]


class Session:
    def __init__(
        self,
        idx: int,
        template: dict,
        tok,
        step: int,
        max_tokens: int,
        sizes: list[int] | None = None,
    ):
        self.idx = idx
        self.sizes = sizes
        self.tok = tok
        self.step = step
        self.base = [dict(m) for m in template["messages"][:2]]
        # Distinct first user message (agents differ per task).
        self.base[1] = {
            "role": "user",
            "content": f"Session {idx}: review the engine source and report bugs "
            f"in module group {idx}. Read files one by one.",
        }
        self.tools = template["tools"]
        self.template = template
        text = source_text(idx, max_tokens * 6 + 400_000)
        self.ids = tok.encode(text, add_special_tokens=False)
        self.turn = 0
        self.max_tokens_total = max_tokens

    def chunk(self, j: int) -> str:
        if self.sizes:
            lo = sum(self.sizes[:j])
            return self.tok.decode(self.ids[lo : lo + self.sizes[j]])
        return self.tok.decode(self.ids[j * self.step : (j + 1) * self.step])

    def messages(self, turns: int) -> list[dict]:
        m = list(self.base)
        for j in range(turns):
            cid = f"call_{self.idx}_{j}"
            m.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps(
                                    {"filePath": f"/work/s{self.idx}/file_{j}.py"}
                                ),
                            },
                        }
                    ],
                }
            )
            m.append({"role": "tool", "tool_call_id": cid, "content": self.chunk(j)})
        return m

    def body(self, turns: int, max_tokens: int = 8) -> dict:
        b = {k: v for k, v in self.template.items() if k not in ("stream_options",)}
        b.update(
            messages=self.messages(turns),
            stream=False,
            max_tokens=max_tokens,
            temperature=0,
        )
        return b


def post(url: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    r["_wall_s"] = time.time() - t0
    return r


def rss_gib(pid: int) -> float:
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return round(int(out) / 1048576, 2)
    return -1.0


def kv_stats(url: str) -> dict | None:
    with contextlib.suppress(Exception):
        return json.load(urllib.request.urlopen(url + "/debug/kv-cache", timeout=10))
    return None


def summarize(r: dict) -> dict:
    x = r.get("x_yunshu") or {}
    u = r.get("usage") or {}
    return dict(
        prompt=u.get("prompt_tokens"),
        cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens")
        or x.get("cached_tokens"),
        ttft_ms=x.get("ttft_ms"),
        prefill_ms=x.get("prefill_ms"),
        total_ms=x.get("total_ms"),
        wall_s=round(r.get("_wall_s", 0), 2),
        cache=x.get("cache"),
        text=(r["choices"][0]["message"].get("content") or "")[:400],
        reasoning=(r["choices"][0]["message"].get("reasoning_content") or "")[:200],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument(
        "--scenario", choices=["multi", "tiers", "identity", "restart"], default="multi"
    )
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--target", type=int, default=30000)
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--gap-s", type=float, default=0)
    ap.add_argument("--lengths", default="10000,30000")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/tmp/apcaudit-server.log")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--deadline-min", type=float, default=17)
    a = ap.parse_args()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    t_start = time.time()
    template = json.loads(Path(a.template).read_text())

    from mlx_lm.utils import load_tokenizer
    from servers import Server, free_ports

    tok = load_tokenizer(Path(a.checkpoint))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (port,) = free_ports(1)
    srv = Server("yunshu", a.checkpoint, port, Path(a.log)).start()
    pid = srv.proc.pid

    def emit(rec: dict) -> None:
        rec.update(
            label=a.label, t=round(time.time() - t_start, 1), rss_gib=rss_gib(pid)
        )
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "text"}), flush=True)

    try:
        emit(dict(kind="meta", args=vars(a), stats=kv_stats(srv.url)))
        if a.scenario == "multi":
            sess = [
                Session(i, template, tok, a.step, a.target) for i in range(a.sessions)
            ]
            steps = max(1, (a.target - 7500) // a.step)
            for k in range(1, steps + 1):
                for s in sess:
                    if time.time() - t_start > a.deadline_min * 60:
                        raise TimeoutError("deadline")
                    r = summarize(post(srv.url, s.body(k, a.max_new)))
                    emit(
                        dict(
                            kind="turn",
                            session=s.idx,
                            step=k,
                            **r,
                            stats=kv_stats(srv.url),
                        )
                    )
                if a.gap_s:
                    time.sleep(a.gap_s)
            for s in sess:  # revisit after everyone else ran
                r = summarize(post(srv.url, s.body(steps, a.max_new)))
                emit(
                    dict(
                        kind="revisit",
                        session=s.idx,
                        step=steps,
                        **r,
                        stats=kv_stats(srv.url),
                    )
                )
            for s in sess:  # and one more growth step
                r = summarize(post(srv.url, s.body(steps + 1, a.max_new)))
                emit(dict(kind="revisit+1", session=s.idx, step=steps + 1, **r))
            emit(dict(kind="final", stats=kv_stats(srv.url)))
        elif a.scenario == "restart":
            for L in [int(x) for x in a.lengths.split(",")]:
                s = Session(
                    200 + L // 1000,
                    template,
                    tok,
                    0,
                    L,
                    sizes=[max(L - 8000, 500), 300],
                )
                cold = summarize(post(srv.url, s.body(1, 64)))
                emit(dict(kind="cold", length=L, **cold))
                t0 = time.time()
                srv.proc.terminate()
                with contextlib.suppress(Exception):
                    srv.proc.wait(timeout=300)
                emit(dict(kind="stop", length=L, stop_s=round(time.time() - t0, 1)))
                srv.kill()
                (port,) = free_ports(1)
                srv = Server("yunshu", a.checkpoint, port, Path(a.log)).start()
                pid = srv.proc.pid
                hit = summarize(post(srv.url, s.body(2, 64)))
                emit(dict(kind="hit-after-restart", length=L, **hit))
                hit2 = summarize(post(srv.url, s.body(2, 64)))
                emit(dict(kind="hit-repeat", length=L, **hit2))
        else:
            lengths = [int(x) for x in a.lengths.split(",")]
            for L in lengths:
                # one big tool result sized so the prompt is ~L tokens, then a small
                # second tool result (the agent's next turn extends the same history)
                s = Session(
                    100 + L // 1000,
                    template,
                    tok,
                    0,
                    L,
                    sizes=[max(L - 8000, 500), 300],
                )
                cold = summarize(post(srv.url, s.body(1, 64)))
                emit(dict(kind="cold", length=L, **cold, stats=kv_stats(srv.url)))
                b2 = s.body(2, 64)
                hit = summarize(post(srv.url, b2))
                emit(dict(kind="hit", length=L, **hit, stats=kv_stats(srv.url)))
                hit2 = summarize(post(srv.url, b2))
                emit(dict(kind="hit-repeat", length=L, **hit2))
        emit(dict(kind="complete"))
    finally:
        srv.kill()
    # fail closed: a run that did not reach its last line wrote incomplete results
    lines = out.read_text().splitlines() if out.exists() else []
    if not lines or json.loads(lines[-1]).get("kind") != "complete":
        print("session_replay: results incomplete", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
