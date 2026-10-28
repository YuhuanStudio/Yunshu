"""Turn-2 TTFT cases on the real 27B server (run through gpuq --quiet).

Cases per rep (fresh session text each): imm = follow-up right after turn 1 ends;
gap = follow-up after --gap seconds idle (rehearsal had time); warm = the warmprobe shape
(identical prompt re-sent, then the follow-up). Saves the server log segment of every
request next to the rows (DEBUG log shows the rehearsal verdicts)."""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("/Volumes/P5Plus/yunshu-build/lead/turn2gap")
CASES = ("imm", "gap", "warm")


def parse_cases(s):
    cases = [c for c in s.split(",") if c]
    bad = [c for c in cases if c not in CASES]
    if bad or not cases:
        raise ValueError(f"unknown cases {bad}")
    return cases


def follow_body(t, srv, text, first_text):
    f = t.req(srv.model, text, 16)
    f["messages"] += [
        {"role": "assistant", "content": first_text.strip()},
        {"role": "user", "content": "Continue with the next part, at the same length."},
    ]
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", default="/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
    p.add_argument("--arm", default="base")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--kinds", default="code,prose")
    p.add_argument("--cases", default="imm,gap,warm")
    p.add_argument("--gap", type=float, default=4.0)
    p.add_argument(
        "--profile", choices=("none", "plain", "sync", "call"), default="none"
    )
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cases = parse_cases(a.cases)
    if a.dry_run:
        print("dry-run ok", cases)
        return
    import tfbench as t

    if a.profile != "none":
        popen = subprocess.Popen
        prof = str(Path(a.tree) / "scripts/research/coldttft_profile.py")
        callprof = str(Path(a.tree) / "scripts/research/callprof.py")

        def launch(command, *args, **kwargs):
            if command and Path(command[0]).name == "yunshu":
                if a.profile == "call":
                    command = [sys.executable, callprof, *command[1:]]
                else:
                    flag = "--serve-plain" if a.profile == "plain" else "--serve"
                    command = [sys.executable, prof, flag, *command[1:]]
            return popen(command, *args, **kwargs)

        subprocess.Popen = launch
    t.YUNSHU_SRC = a.tree + "/python"
    t.OUT = OUT / "work"
    env = {
        "YUNSHU_VLM_DRAFT": t.D,
        "YUNSHU_VLM_APC_DISK": "0",
        "YUNSHU_VLM_APC_DISK_DIR": f"/Volumes/Micron/yunshu-apc/turn2gap-{a.arm}",
        "YUNSHU_LOG_LEVEL": "DEBUG",
        **dict(e.split("=", 1) for e in a.env),
    }
    srv = t.start_server("yunshu", env, f"turn2gap-{a.arm}")
    out = OUT / f"res-{a.arm}.jsonl"

    def seg(since):
        with open(srv.log, "rb") as f:
            f.seek(since)
            return f.read().decode(errors="replace")

    def step(case, kind, rep, phase, body):
        off = srv.log.stat().st_size
        res = t.send(srv.url, body)
        time.sleep(0.2)
        text = res.pop("_text", "")
        log = seg(off)
        (OUT / f"log-{a.arm}-{case}-{kind}-{rep}-{phase}.txt").write_text(log)
        row = dict(arm=a.arm, case=case, kind=kind, rep=rep, phase=phase, **res)
        with out.open("a") as o:
            o.write(json.dumps(row) + "\n")
        lat = ((res.get("xy") or {}).get("latency") or {}).get("durations_ms")
        print(
            json.dumps(
                dict(
                    case=case,
                    kind=kind,
                    rep=rep,
                    phase=phase,
                    ttft_s=res.get("ttft_s"),
                    pt=res.get("pt"),
                    cached=res.get("cached"),
                    lat=lat,
                )
            ),
            flush=True,
        )
        return text

    try:
        for kind in a.kinds.split(","):
            base = t.decode_prompt(kind, a.ctx, True)
            for r in range(a.reps):
                for case in cases:
                    text = f"[session {uuid.uuid4().hex}]\n" + base
                    first = step(case, kind, r, "turn1", t.req(srv.model, text, 512))
                    if case == "gap":
                        time.sleep(a.gap)
                    elif case == "warm":
                        time.sleep(1.0)
                        step(case, kind, r, "warm", t.req(srv.model, text, 16))
                    step(case, kind, r, "turn2", follow_body(t, srv, text, first))
                    time.sleep(2.0)
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
