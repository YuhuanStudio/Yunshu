#!/usr/bin/env python3
"""Fake gpuq for the yv unit tests: `submit` runs the command at once and records a job json."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

jobs = Path(os.environ["FAKE_GPUQ_DIR"])
jobs.mkdir(parents=True, exist_ok=True)
cmd = sys.argv[1]
if cmd == "submit":
    args = sys.argv[2:]
    i = args.index("--")
    opts, argv = args[:i], args[i + 1 :]
    label = opts[opts.index("--label") + 1]
    quiet = "--quiet" in opts
    n = len(list(jobs.glob("*.json"))) + 1
    jid = f"{n:04d}-{label}"
    log = jobs / f"{jid}.log"
    pat = os.environ.get("FAKE_GPUQ_REFUSE")
    if pat and re.search(pat, label):
        print("preflight failed: doomed", file=sys.stderr)
        sys.exit(1)
    with log.open("w") as f:
        rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode
    cont = bool(
        quiet
        and os.environ.get("FAKE_GPUQ_CONTENDED")
        and re.search(os.environ["FAKE_GPUQ_CONTENDED"], label)
        and "-a1-" in label
    )
    (jobs / f"{jid}.json").write_text(
        json.dumps(
            {
                "id": jid,
                "label": label,
                "state": "done" if rc == 0 else "failed",
                "rc": rc,
                "contended": cont,
                "cmd": argv,
            }
        )
    )
    print(jid)
elif cmd == "cancel":
    p = jobs / f"{sys.argv[2]}.json"
    if p.exists():
        d = json.loads(p.read_text())
        d["state"] = "cancelled"
        p.write_text(json.dumps(d))
elif cmd == "log":
    p = jobs / f"{sys.argv[2]}.log"
    print(p.read_text() if p.exists() else "")
elif cmd == "wait":
    pass
