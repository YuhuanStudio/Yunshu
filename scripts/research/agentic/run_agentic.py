"""Agentic coding benchmark: real coding-agent CLIs drive a local model server and write code.

    run_agentic.py run --serve yunshu --checkpoint $M --agent opencode --tasks all --repeat 3 \
        --engine-label yunshu-default --output docs/research/runs/DATE-agentic/runs.jsonl
    run_agentic.py run --engine-url http://127.0.0.1:18990 --engine-label x ...   (server already up)
    run_agentic.py summarize docs/research/runs/DATE-agentic/*.jsonl
    run_agentic.py check-tasks          # reference solutions pass, starting repos fail
    run_agentic.py list-tasks

Every agent talks to a recording proxy (proxy.py) that forwards to the engine, so the same code
measures every engine: per-request TTFT, decode tok/s, prompt / completion / cached tokens, API
errors and malformed tool calls. Hidden tests decide pass/fail. Runs are resumable: a
(task, repeat) already in --output for the same agent and engine label is skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import agents  # noqa: E402
import tasks as tasklib  # noqa: E402
from proxy import RecordingProxy  # noqa: E402

BUILD = agents.BUILD


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def git_sha() -> str:
    src = os.environ.get("AGENTIC_YUNSHU_SRC")
    main = os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
    where = str(Path(src).parent) if src else main
    return sh(["git", "-C", where, "rev-parse", "--short", "HEAD"]) or "unknown"


def hw() -> dict:
    return dict(
        chip=sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        mem_gib=round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30),
        os=platform.platform(),
    )


def done_keys(path: Path, label: str, agent: str) -> set[tuple[str, int]]:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (
                r.get("type") == "run"
                and r.get("engine") == label
                and r.get("agent") == agent
            ):
                out.add((r["task"], r["repeat"]))
    return out


class MemSampler(threading.Thread):
    def __init__(self, pid: int | None, period: float = 2.0):
        super().__init__(daemon=True)
        self.pid, self.period = pid, period
        self.peak = 0
        self.last = 0
        self.stop_ev = threading.Event()

    def run(self):
        if not self.pid:
            return
        from process_memory import process_tree_memory

        while not self.stop_ev.is_set():
            try:
                m = process_tree_memory(self.pid)
                v = max(m["physical_footprint_sum_bytes"], m["rss_sum_bytes"])
                self.last = v
                self.peak = max(self.peak, v)
            except Exception:
                pass
            self.stop_ev.wait(self.period)

    def finish(self) -> tuple[float, float]:
        self.stop_ev.set()
        return round(self.peak / 2**30, 2), round(self.last / 2**30, 2)


def init_repo(workdir: Path, env: dict):
    e = {
        **env,
        "GIT_AUTHOR_NAME": "bench",
        "GIT_AUTHOR_EMAIL": "b@x.invalid",
        "GIT_COMMITTER_NAME": "bench",
        "GIT_COMMITTER_EMAIL": "b@x.invalid",
    }
    for c in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "initial state"]):
        subprocess.run(["git", *c], cwd=workdir, env=e, capture_output=True)


TRIM = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "ttft_s",
    "decode_tok_s",
    "total_s",
    "status",
    "finish_reason",
    "tool_calls",
    "malformed_tool_calls",
    "leaked_tool_markup",
    "stream",
    "n_tools",
    "n_messages",
    "kind",
    "model",
)


def run_one(task, agent, args, engine_url, model, server_pid, rep, out_dir) -> dict:
    run_id = f"{args.engine_label}-{agent}-{task.id}-r{rep}"
    run_dir = BUILD / "work" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    workdir = run_dir / "work"
    task.stage(workdir)
    envbase = agents.base_env(run_dir / "gitenv")
    init_repo(workdir, envbase)
    art = out_dir / "artifacts" / run_id
    art.mkdir(parents=True, exist_ok=True)
    px = RecordingProxy(
        engine_url,
        port=args.proxy_port,
        bodies_dir=art / "bodies",
        save_bodies=args.save_bodies,
    ).start()
    mem = MemSampler(server_pid)
    mem.start()
    launch = agents.prepare(
        agent,
        run_dir,
        workdir,
        px.url,
        model,
        task.prompt,
        context_len=args.context,
        max_output=args.max_output,
    )
    timeout = int(task.timeout_s or args.timeout_min * 60)
    t0 = time.time()
    timed_out = False
    hb_stop = threading.Event()

    def heartbeat():  # keeps the queue's stall detector fed during long agent runs
        while not hb_stop.wait(30):
            g = [r for r in px.snapshot() if r.get("kind")]
            print(
                f"[agentic]   {run_id} running {time.time() - t0:.0f}s, {len(g)} requests",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    with (
        (run_dir / "agent.stdout").open("wb") as so,
        (run_dir / "agent.stderr").open("wb") as se,
    ):
        p = subprocess.Popen(
            launch.cmd,
            env=launch.env,
            cwd=launch.cwd,
            stdout=so,
            stderr=se,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, 9)
            rc = p.wait()
    wall = time.time() - t0
    hb_stop.set()
    for n in ("agent.stdout", "agent.stderr"):
        shutil.copy2(run_dir / n, art / n)
    time.sleep(0.6)  # let the proxy record the final request
    recs = px.snapshot()
    inflight = px.inflight()
    px.stop()
    peak, last = mem.finish()
    passed, test_out = task.grade(workdir)
    gen = [r for r in recs if r.get("kind")]
    other_err = [r for r in recs if not r.get("kind") and r.get("status", 0) >= 400]
    api_err = [r for r in gen if r.get("error")]
    pt = sum(r.get("prompt_tokens") or 0 for r in gen)
    ct = sum(r.get("completion_tokens") or 0 for r in gen)
    cached = sum(r.get("cached_tokens") or 0 for r in gen)
    t_start = min((r["t_wall"] for r in recs), default=t0)
    reqs = []
    for r in gen:
        d = {k: r[k] for k in TRIM if k in r}
        d["t"] = round(r["t_wall"] - t0, 2)
        if r.get("sampling"):
            d["sampling"] = r["sampling"]
        if r.get("x_yunshu") is not None:
            d["x_yunshu"] = r["x_yunshu"]
        reqs.append(d)
    reqs += inflight
    if api_err:
        (art / "api_errors.json").write_text(json.dumps(api_err, indent=1))
    (art / "grade.txt").write_text(test_out)
    tail = (art / "agent.stdout").read_bytes()[-1500:].decode(errors="replace")
    row = dict(
        type="run",
        engine=args.engine_label,
        agent=agent,
        task=task.id,
        kind=task.kind,
        repeat=rep,
        passed=passed,
        wall_s=round(wall, 1),
        agent_exit=rc,
        timed_out=timed_out,
        first_request_delay_s=round(t_start - t0, 2),
        requests=len(gen),
        prompt_tokens=pt,
        completion_tokens=ct,
        cached_tokens=cached,
        cache_hit_ratio=round(cached / pt, 4) if pt else None,
        max_prompt_tokens=max((r.get("prompt_tokens") or 0 for r in gen), default=0),
        api_errors=len(api_err),
        other_http_errors=len(other_err),
        malformed_tool_calls=sum(r.get("malformed_tool_calls", 0) for r in gen),
        leaked_tool_markup=sum(1 for r in gen if r.get("leaked_tool_markup")),
        tool_calls=sum(r.get("tool_calls", 0) for r in gen),
        server_peak_gib=peak,
        server_end_gib=last,
        artifacts=str(art.relative_to(out_dir)),
        grade_tail=test_out[-400:],
        agent_tail=tail[-400:],
        request_log=reqs,
    )
    if not args.keep_work:
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        shutil.copytree(workdir, art / "final-work", dirs_exist_ok=True)
    return row


def cmd_run(args):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = BUILD / "tasks-cache"
    cache.mkdir(parents=True, exist_ok=True)
    tasks = tasklib.select(args.tasks, tasklib.all_tasks(cache))
    plan = [(t, r) for r in range(1, args.repeat + 1) for t in tasks]
    plan = [x for i, x in enumerate(plan) if i % args.shard_n == args.shard_i]
    todo = [
        (t, r)
        for t, r in plan
        if (t.id, r) not in done_keys(out, args.engine_label, args.agent)
    ]
    print(f"[agentic] {len(plan)} planned, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    server = None
    t_job = time.time()
    try:
        if args.serve:
            from servers import Server, free_ports

            sp, pp = free_ports(2)
            args.proxy_port = pp
            server = Server(
                args.serve,
                args.checkpoint,
                sp,
                out.parent / "logs" / f"server-{args.engine_label}-{args.agent}.log",
                extra=args.server_arg,
            )
            server.start()
            engine_url, model, pid = server.url, server.model_id, server.proc.pid
            print(
                f"[agentic] server {args.serve} ready in {server.ready_s:.0f}s as {model!r}",
                flush=True,
            )
        else:
            engine_url, pid = args.engine_url.rstrip("/"), args.server_pid
            model = args.model
            if not model:
                import urllib.request

                with urllib.request.urlopen(engine_url + "/v1/models") as r:
                    model = json.load(r)["data"][0]["id"]
            args.proxy_port = args.proxy_port or 0
        if not out.exists() or out.stat().st_size == 0:
            meta = dict(
                type="meta",
                engine=args.engine_label,
                serve=args.serve,
                checkpoint=args.checkpoint,
                model=model,
                git_sha=git_sha(),
                agent=args.agent,
                agent_version=agents.agent_version(args.agent),
                flags=args.server_arg,
                yunshu_env={
                    k: v for k, v in os.environ.items() if k.startswith("YUNSHU_")
                },
                context_len=args.context,
                timeout_min=args.timeout_min,
                repeat=args.repeat,
                polyglot_commit=tasklib.dataset_commit(),
                hw=hw(),
                time=time.strftime("%FT%T"),
                sampling="agent defaults (see per-request sampling)",
            )
            with out.open("a") as f:
                f.write(json.dumps(meta) + "\n")
        for task, rep in todo:
            if time.time() - t_job > args.budget_min * 60:
                print(
                    "[agentic] job budget reached; remaining runs resume in a later job",
                    flush=True,
                )
                break
            if server and server.proc.poll() is not None:
                raise RuntimeError("server died")
            row = run_one(
                task, args.agent, args, engine_url, model, pid, rep, out.parent
            )
            with out.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[agentic] {task.id} r{rep}: {'PASS' if row['passed'] else 'FAIL'} "
                f"{row['wall_s']}s reqs={row['requests']} cache={row['cache_hit_ratio']} "
                f"err={row['api_errors']} malformed={row['malformed_tool_calls']} "
                f"peak={row['server_peak_gib']}GiB",
                flush=True,
            )
    finally:
        if server:
            server.kill()
    return 0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def cmd_summarize(args):
    rows = []
    for pat in args.files:
        for f in glob.glob(pat):
            for line in Path(f).read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "run":
                    rows.append(r)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["engine"], r["agent"]), []).append(r)
    print(
        "| engine | agent | runs | pass | rate (Wilson 95%) | wall med / p90 (s) | req/run | cache hit | "
        "decode tok/s med | ttft med (s) | api err | malformed | leaked | peak GiB |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (eng, ag), rs in sorted(groups.items()):
        n, k = len(rs), sum(1 for r in rs if r["passed"])
        lo, hi = wilson(k, n)
        walls = [r["wall_s"] for r in rs]
        pt = sum(r["prompt_tokens"] for r in rs)
        cached = sum(r["cached_tokens"] for r in rs)
        dec = [
            q["decode_tok_s"]
            for r in rs
            for q in r["request_log"]
            if q.get("decode_tok_s")
        ]
        ttft = [
            q["ttft_s"]
            for r in rs
            for q in r["request_log"]
            if q.get("ttft_s") is not None
        ]
        med = lambda v: f"{statistics.median(v):.1f}" if v else "-"  # noqa: E731
        print(
            f"| {eng} | {ag} | {n} | {k} | {k / n:.0%} ({lo:.0%}-{hi:.0%}) | {med(walls)} / {pct(walls, 0.9):.0f} | "
            f"{statistics.mean(r['requests'] for r in rs):.1f} | {cached / pt if pt else 0:.1%} | {med(dec)} | "
            f"{med(ttft)} | {sum(r['api_errors'] for r in rs)} | {sum(r['malformed_tool_calls'] for r in rs)} | "
            f"{sum(r['leaked_tool_markup'] for r in rs)} | {max(r['server_peak_gib'] for r in rs):.1f} |"
        )
    print("\nPer task pass count (passed/runs):\n")
    tasks = sorted({r["task"] for r in rows})
    keys = sorted(groups)
    print("| task | " + " | ".join(f"{e}/{a}" for e, a in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for t in tasks:
        cells = []
        for key in keys:
            rs = [r for r in groups[key] if r["task"] == t]
            cells.append(f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")


def cmd_check_tasks(args):
    cache = BUILD / "tasks-cache"
    cache.mkdir(parents=True, exist_ok=True)
    all_t = tasklib.all_tasks(cache)
    bad = 0
    for t in tasklib.select(args.tasks, all_t):
        w = BUILD / "check" / t.id
        t.stage(w)
        before, _ = t.grade(w)
        ref = t.src / "reference"
        if ref.exists():
            shutil.copytree(ref, w, dirs_exist_ok=True)
            if (w / "apply.py").exists():  # scripted reference (generated repos)
                (w / "apply.py").unlink()
                subprocess.run(
                    [tasklib.PYTHON, str(ref / "apply.py"), str(w)], check=True
                )
        after, tail = t.grade(w)
        ok = (not before) and after
        bad += not ok
        print(
            f"{'OK ' if ok else 'BAD'} {t.id}: start={'pass' if before else 'fail'} reference={'pass' if after else 'fail'}"
        )
        if not ok:
            print(tail[-600:])
        shutil.rmtree(w, ignore_errors=True)
    return 1 if bad else 0


def cmd_list(args):
    cache = BUILD / "tasks-cache"
    cache.mkdir(parents=True, exist_ok=True)
    for t in tasklib.all_tasks(cache).values():
        print(f"{t.id:32s} {t.kind:8s} {t.title}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine-url")
    r.add_argument("--serve", choices=["yunshu", "tensorfold"])
    r.add_argument("--checkpoint", default=os.environ.get("M"))
    r.add_argument("--server-arg", action="append", default=[])
    r.add_argument("--server-pid", type=int)
    r.add_argument("--model")
    r.add_argument("--engine-label", required=True)
    r.add_argument("--agent", required=True, choices=["opencode", "claude", "codex"])
    r.add_argument("--tasks", default="all")
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--output", required=True)
    r.add_argument("--timeout-min", type=float, default=20)
    r.add_argument(
        "--budget-min",
        type=float,
        default=35,
        help="stop starting new runs after this many minutes",
    )
    r.add_argument("--shard-i", type=int, default=0)
    r.add_argument("--shard-n", type=int, default=1)
    r.add_argument("--context", type=int, default=131072)
    r.add_argument("--max-output", type=int, default=32768)
    r.add_argument("--proxy-port", type=int, default=0)
    r.add_argument("--save-bodies", action="store_true")
    r.add_argument("--keep-work", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("files", nargs="+")
    c = sub.add_parser("check-tasks")
    c.add_argument("--tasks", default="all")
    sub.add_parser("list-tasks")
    a = ap.parse_args()
    if a.cmd == "run" and not (a.engine_url or a.serve):
        ap.error("run needs --engine-url or --serve")
    sys.exit(
        {
            "run": cmd_run,
            "summarize": cmd_summarize,
            "check-tasks": cmd_check_tasks,
            "list-tasks": cmd_list,
        }[a.cmd](a)
        or 0
    )


if __name__ == "__main__":
    main()
