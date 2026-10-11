"""yv command line: ab / status / wait / gate / suites."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import gate as gate_mod
from . import runner
from .core import REPO, InfraError, resolve_arm
from .gate import local_env
from .suites import STAGES, SUITES


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="yv",
        description="Yunshu verification: preflight, smoke, identity, apc, quality, speed, memory -- one verdict.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ab = sub.add_parser("ab", help="verify a candidate against a base")
    ab.add_argument("--base", required=True, help="git ref or checkout directory")
    ab.add_argument("--cand", required=True, help="git ref or checkout directory")
    ab.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="server env, both arms",
    )
    ab.add_argument(
        "--cand-env",
        action="append",
        default=[],
        metavar="K=V",
        help="server env, candidate only",
    )
    ab.add_argument(
        "--base-env",
        action="append",
        default=[],
        metavar="K=V",
        help="server env, base only",
    )
    ab.add_argument(
        "--quick",
        action="store_true",
        help="suite quick: identity + speed on 1K/8K, quality 200; base arm cached",
    )
    ab.add_argument(
        "--reuse-base-speed",
        dest="reuse_base_speed",
        action="store_true",
        help="reuse cached base speed reps (noisier A/B)",
    )
    ab.add_argument(
        "--suite",
        default=None,
        help=f"{sorted(SUITES)} or comma list of {list(STAGES)}",
    )
    ab.add_argument(
        "--label", required=True, help="worker-topic; names the run directory"
    )
    ab.add_argument(
        "--model",
        default=local_env().get("M", ""),
        help="checkpoint directory (default: 27B from local.env)",
    )
    ab.add_argument(
        "--model-name", default="", help="'model' field for paired quality requests"
    )
    ab.add_argument(
        "--engaged",
        action="append",
        default=[],
        metavar="SPEC",
        help="smoke must see the candidate path: log:REGEX | field:KEY=REGEX | spec:MODE",
    )
    ab.add_argument("--ctx", help="override suite contexts, e.g. 1024,8192")
    ab.add_argument("--reps", type=int, help="speed reps (default from suite, 3)")
    ab.add_argument("--mmlu-n", type=int, dest="mmlu_n", help="paired quality items")
    ab.add_argument(
        "--mem-sizes",
        dest="mem_sizes",
        help="memory stage context sizes, e.g. 8192,32768",
    )
    ab.add_argument("--mem-reps", type=int, dest="mem_reps")
    ab.add_argument(
        "--speed-tol",
        type=float,
        dest="speed_tol",
        help="percent regression tolerance (default 2)",
    )
    ab.add_argument(
        "--spec-off",
        dest="spec_off",
        action="store_true",
        default=None,
        help="also check spec on == off",
    )
    ab.add_argument(
        "--spec-modes",
        dest="spec_modes",
        help="identity under each speculative method: default,mtp,dflash (comma list)",
    )
    ab.add_argument("--no-apc-hit-required", action="store_true")
    ab.add_argument(
        "--mem-gb",
        type=float,
        default=0,
        help="gpuq memory per job (default 60 for 27B, else 14)",
    )
    ab.add_argument("--priority", type=int, default=0)
    ab.add_argument(
        "--detach",
        action="store_true",
        help="return at once; follow with yv status / yv wait",
    )
    st = sub.add_parser("status", help="state of a run")
    st.add_argument("run", help="run directory or name")
    wt = sub.add_parser("wait", help="block until a run has a verdict")
    wt.add_argument("run")
    gt = sub.add_parser(
        "gate", help="release gate through gpuq with per-stage persistence"
    )
    gt.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="skip stages that passed on this commit (default)",
    )
    gt.add_argument("--fresh", action="store_true", help="rerun every stage")
    gt.add_argument("--stages", default=",".join(gate_mod.DEFAULT_STAGES))
    gt.add_argument("--priority", type=int, default=0)
    gt.add_argument(
        "--ref", help="verify this pinned commit instead of the current tree"
    )
    gt.add_argument(
        "--label-prefix",
        default=None,
        help="gpuq job owner prefix (default: YV_LABEL_PREFIX, GPUQ_OWNER, then infra)",
    )
    gt.add_argument("--root", help="isolated gate install/cache root")
    sub.add_parser("suites", help="list suites")
    return ap


def pinned_spec(arm) -> str:
    """What this caller resolved: the checkout directory as an absolute path, else the commit."""
    p = Path(arm.spec)
    if p.is_dir() and (p / ".git").exists():
        return str(arm.path)
    return arm.commit


def pin_arms(argv: list, base, cand) -> list:
    """`--detach` re-runs yv in REPO, so a relative spec (HEAD, a branch checked out elsewhere,
    ".") would resolve against the main checkout: on 2026-10-06 a longgap run named for its
    candidate (`--cand HEAD` from the worktree) verified main against itself for 18 h. Hand the
    child the commit / absolute directory this caller resolved."""
    out, pins = [], {"--base": pinned_spec(base), "--cand": pinned_spec(cand)}
    i = 0
    while i < len(argv):
        x = argv[i]
        flag = x.split("=", 1)[0]
        if flag in pins:
            out.append(f"{flag}={pins[flag]}")
            i += 1 if "=" in x else 2
            continue
        out.append(x)
        i += 1
    return out


def verifier_scripts() -> Path:
    """Detached runs retain this verifier implementation, including new stages."""
    return Path(__file__).resolve().parents[1]


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "suites":
            for k, v in SUITES.items():
                print(f"{k:10s} {v['stages']} ctx={v.get('ctx')}")
            return 0
        if a.cmd == "status":
            print(runner.status_text(runner.load_run(a.run)))
            return 0
        if a.cmd == "wait":
            return runner.wait_run(runner.load_run(a.run))
        if a.cmd == "gate":
            return gate_mod.run_gate(
                [s for s in a.stages.split(",") if s],
                resume=not a.fresh,
                priority=a.priority,
                repo=resolve_arm("cand", a.ref).path if a.ref else None,
                label_prefix=a.label_prefix,
                extra_env={"GATE_ROOT": a.root} if a.root else None,
                log=lambda m: print(f"[yv] {m}", flush=True),
            )
        if a.cmd == "ab":
            a.suite = a.suite or ("quick" if a.quick else None)
            if not a.suite:
                print("yv: --suite or --quick is required")
                return 2
            if not a.model or not Path(a.model).is_dir():
                print(f"yv: --model {a.model!r} is not a directory")
                return 2
            if a.detach:
                cand = resolve_arm("cand", a.cand)
                base = resolve_arm("base", a.base)
                rd = runner.run_dir_for(a.label, cand)
                rd.mkdir(parents=True, exist_ok=True)
                args = pin_arms([x for x in argv if x != "--detach"], base, cand)
                log = open(rd / "yv.log", "a")  # noqa: SIM115
                p = subprocess.Popen(
                    [sys.executable, "-m", "verify", *args],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=dict(os.environ, PYTHONPATH=str(verifier_scripts())),
                    cwd=str(REPO),
                )
                print(
                    f"detached pid {p.pid}; run dir {rd}\nfollow: yv status {rd}   yv wait {rd}"
                )
                return 0
            return runner.run_ab(a)
    except InfraError as e:
        print(f"yv: infra error: {e}")
        return 2
    return 2
