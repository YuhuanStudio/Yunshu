"""Sampled-decode probe: gapboard's 8K prose warm cell under several sampling settings.

Reuses the gapboard harness (gb_cell) with a private plan, so the engine, prompts and request
shape are the board's. Usage: sampled_cell.py --label L --out DIR --src WORKTREE [--stages a,b]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/Volumes/P5Plus/yunshu-build/claude-alt/gapboard")
import gb_cell as g  # noqa: E402

SETS = {
    "greedy": {"temperature": 0},
    "s07p95": {"temperature": 0.7, "top_p": 0.95, "seed": 1234},
    "s07p1": {"temperature": 0.7, "top_p": 1.0, "seed": 1234},
    "s07p95np": {
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 1234,
        "presence_penalty": 0.0,
    },
    "s03p95": {"temperature": 0.3, "top_p": 0.95, "seed": 1234},
    "s10p95": {"temperature": 1.0, "top_p": 0.95, "seed": 1234},
}


def plan(names, ctx):
    st = [g.stage("prime", "decode", *g.greedy((ctx,), "prose", "cold"))]
    for n in names:
        st.append(
            g.stage(
                "samp-" + n,
                "decode",
                *g.greedy((ctx,), "prose", "warm"),
                "--request-extra",
                json.dumps(SETS[n]),
            )
        )
    return dict(ctx=ctx + 4096, parallel=1, stages=st)


if __name__ == "__main__":
    argv = sys.argv[1:]
    names = ["greedy", "s07p95", "s07p1", "s03p95", "s07p95", "greedy"]
    ctx = 8192
    if "--stages" in argv:
        i = argv.index("--stages")
        names = argv[i + 1].split(",")
        del argv[i : i + 2]
    if "--ctx" in argv:
        i = argv.index("--ctx")
        ctx = int(argv[i + 1])
        del argv[i : i + 2]
    g.PLANS["S"] = plan(names, ctx)
    extra = dict(
        kv.split("=", 1)
        for kv in os.environ.get("SAMPLED_SERVER_ENV", "").split()
        if "=" in kv
    )
    if extra:  # the harness scrubs the shell env; server settings must go through its extra_env
        from pathlib import Path

        opt = {
            a: argv[i + 1]
            for i, a in enumerate(argv[:-1])
            if a in ("--src", "--out", "--label")
        }
        src = Path(opt["--src"]).resolve()
        scratch = Path(opt["--out"]).resolve() / f"scratch-{opt['--label']}"
        os.environ.update(
            TFB_YUNSHU_SRC=str(src / "python"),
            TFB_OUT=str(scratch),
            TFB_PORT_LAST="18996",
            TFB_EXACT_PROMPTS="1",
            TMPDIR=str(scratch / "tmp"),
        )
        os.environ.setdefault(
            "TFB_WORK", "/Volumes/P5Plus/yunshu-build/snapshot014/corpus"
        )
        sys.path.insert(0, str(src / "scripts" / "research"))
        sys.path.insert(0, str(src / "scripts" / "dev"))
        import tfbench as tb

        _start = tb.start_server
        tb.start_server = lambda engine, extra_env, tag, **kw: _start(
            engine, {**extra_env, **extra}, tag, **kw
        )
    if os.environ.get(
        "SAMPLED_KEEP_LOGS"
    ):  # copy server logs out of the scratch dir gb_cell deletes
        import shutil as _sh
        from pathlib import Path as _P

        _rm = g.shutil.rmtree

        def _keep(path, *a, **kw):
            for log in (_P(path) / "out").glob("*.log"):
                _sh.copy(log, _P(os.environ["SAMPLED_KEEP_LOGS"]) / log.name)
            _rm(path, *a, **kw)

        g.shutil.rmtree = _keep
    sys.argv = [sys.argv[0], "--plan", "S", "--engine", "yunshu-main"] + argv
    g.main()
