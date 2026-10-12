"""Where a short request's prefill time goes (27B): traced HTTP requests per draft mode.

Run only through gpuq. Each arm starts a server whose launcher installs the first-token
trace (no layer sync), sends sequential short requests, then prints per-event totals before
the first visible token. Arms: ``mtp`` (checkpoint MTP head), ``default`` (auto DFlash drafter),
``off`` (no drafting).

    python scripts/research/hostpath_short.py --arm mtp --out runs/short-mtp.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROF = """
import cProfile, io, pstats, itertools, pathlib
_d = pathlib.Path(%r); _d.mkdir(parents=True, exist_ok=True)
_n = itertools.count()
def _prof(cls, name):
    inner = getattr(cls, name)
    def run(self, *a, **k):
        pr = cProfile.Profile()
        try:
            return pr.runcall(inner, self, *a, **k)
        finally:
            b = io.StringIO(); pstats.Stats(pr, stream=b).sort_stats('cumtime').print_stats(14)
            (_d / f"{name}-{next(_n)}.txt").write_text(b.getvalue())
    setattr(cls, name, run)
_prof(_Coordinator, 'store_checkpoint')
_prof(_Coordinator, 'observe_cache')
"""


def summarize(rows):
    """Per-event-name median over requests of summed duration, plus the first-visible time."""
    names = {}
    for row in rows:
        per = {}
        for ev in row["events"]:
            per.setdefault(ev["name"], []).append(ev["duration_ms"])
        for name, xs in per.items():
            names.setdefault(name, []).append((sum(xs), len(xs)))
    out = {
        name: dict(
            ms_med=round(statistics.median(s for s, _ in v), 2),
            calls=statistics.median(c for _, c in v),
        )
        for name, v in names.items()
    }
    return dict(
        first_visible_ms_med=round(
            statistics.median(r["first_visible_ms"] for r in rows), 1
        ),
        events=out,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("mtp", "default", "off"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument(
        "--prof", action="store_true", help="cProfile APC store/observe calls"
    )
    p.add_argument(
        "--reps",
        type=int,
        nargs="*",
        default=[1],
        help="sentence repeats = prompt lengths",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    sys.path.insert(0, str(HERE))
    import hostpath_probe as hp
    import tfbench as t

    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    t.OUT = a.out.parent / a.out.stem
    t.YUNSHU_SRC = str(HERE.parents[1] / "python")
    trace = a.out.with_suffix(".trace.jsonl")
    launcher = a.out.with_suffix(".launcher.py")
    launcher.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(HERE)!r})\n"
        f"import http_first_token_profile as p\ntr = p.install({str(trace)!r})\n"
        "import importlib\nimport mlx.core as mx\nar = importlib.import_module('mlx_vlm.generate.ar')\n"
        "from yunshu_engine.apc_manager import _Coordinator\n"
        "tr.wrap(mx, 'clear_cache', 'mx.clear_cache')\n"
        "tr.wrap(mx, 'async_eval', 'mx.async_eval')\n"
        "tr.wrap(ar.PromptProcessingBatch, '_store_apc_exact_checkpoints')\n"
        "tr.wrap(_Coordinator, 'observe_cache')\n"
        + (PROF % str(a.out.with_suffix(".prof")) if a.prof else "")
        + "from yunshu_cli import main\nmain()\n"
    )
    launcher.chmod(0o700)
    t.YUNSHU_BIN = str(launcher)
    env = {"YUNSHU_VLM_APC_DISK": "0"}
    if a.arm != "default":
        env["YUNSHU_VLM_DRAFT"] = {"mtp": "mtp", "off": "off"}[a.arm]
    server = None
    try:
        server = t.Srv("yunshu", env, a.out.stem)
        text = "Explain in two sentences why the sky is blue."
        hp.stream(server.port, hp.body_for(server.model, text, 16))  # warmup
        time.sleep(1)
        out = []
        for rep in a.reps:
            body_text = " ".join([text] * rep)
            trace.unlink(missing_ok=True)  # warm requests only
            for i in range(a.n):
                hp.stream(
                    server.port,
                    hp.body_for(server.model, hp.salted(body_text, f"r{rep}s{i}"), 16),
                )
                time.sleep(
                    1
                )  # the trace requires the previous ASGI call to have finished
            rows = [
                r
                for r in map(json.loads, trace.read_text().splitlines())
                if r.get("success")
            ]
            if len(rows) != a.n:
                raise RuntimeError(f"traced {len(rows)} of {a.n} requests")
            out.append(dict(arm=a.arm, reps=rep, n=a.n, **summarize(rows)))
            ev = out[-1]["events"]
            print(
                rep,
                out[-1]["first_visible_ms_med"],
                {
                    k: ev[k]["ms_med"]
                    for k in (
                        "PromptProcessingBatch.prompt_step",
                        "PromptProcessingBatch.generate",
                        "model",
                    )
                    if k in ev
                },
                flush=True,
            )
        a.out.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    finally:
        if server is not None:
            server.kill()
        with contextlib.suppress(OSError):
            launcher.unlink()


if __name__ == "__main__":
    main()
