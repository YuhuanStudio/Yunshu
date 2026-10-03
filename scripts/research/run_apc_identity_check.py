"""Fail-closed wrapper of the existing APC identity probe (gpuq only)."""

import json
import sys
from pathlib import Path

import apc_restore_identity as probe
from cache_probe_source import freeze

source = freeze(sys.argv[sys.argv.index("--out") + 1], probe.t)
probe.main()
path = Path(sys.argv[sys.argv.index("--out") + 1])
record = json.loads(path.read_text().splitlines()[-1])
ok = all(
    record[k]["tokens_equal"] and record[k]["logprobs_equal"]
    for k in ("partial", "full")
)
with path.open("a") as f:
    f.write(json.dumps({"complete": True, "ok": ok, "source": source}) + "\n")
if not ok:
    raise SystemExit(1)
