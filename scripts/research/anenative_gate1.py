"""Cold TTFT of one arm (GPU-only or the Splash-style ANE FFN split) at one context, via tfbench.

Metrics only. The arm decides YUNSHU_ANE_FFN_SPLIT; the program cache and calibration live in a
fixed directory so every rep recalls the same share. Evidence lines of the server log (engagement
of the split, the share, the counters) are appended to the receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["off", "splash"], required=True)
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--kind", default="prose")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/Volumes/P5Plus/yunshu-build/codex/anenative-ffncache"))
    p.add_argument("--decode-tokens", default="32")
    a = p.parse_args()

    os.environ.update(
        TFB_YUNSHU_SRC=str(ROOT / "python"),
        TFB_OUT=str(a.scratch),
        TFB_PORT_LAST="18996",
    )
    sys.path.insert(0, str(HERE))
    import tfbench as tb

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tag = f"gate1-{a.arm}-{a.ctx}-{a.kind}-r{a.rep}"
    args = [
        "tfbench",
        "--engine",
        "yunshu-main",
        "--part",
        "decode",
        "--rep",
        str(a.rep),
        "--out",
        str(a.out),
        "--tag",
        tag,
        "--ctx-tokens",
        str(a.ctx + 4096),
        "--idle-s",
        "0",
        "--parallel",
        "1",
        "--env",
        "YUNSHU_CONSOLE=0",
        "--env",
        "YUNSHU_VLM_APC_DISK=0",
        "--env",
        "YUNSHU_VLM_DRAFT=" + tb.D,
        "--env",
        f"YUNSHU_ANE_FFN_SPLIT={a.arm}",
        "--env",
        f"YUNSHU_ANE_FFN_CACHE={a.cache}",
        "--only-ctx",
        str(a.ctx),
        "--only-kind",
        a.kind,
        "--decode-tokens",
        a.decode_tokens,
        "--long-ask",
        "--phases",
        "cold",
    ]
    sys.argv = args
    scratch_tmp = a.scratch / "tmp"
    scratch_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch_tmp)
    try:
        tb.main()
    finally:
        lines = []
        for log in (a.scratch / "out").glob("*.log"):
            lines.extend(
                line
                for line in log.read_text(errors="replace").splitlines()
                if re.search(r"Neural Engine|ANE FFN|prompt cache: .*GiB", line)
            )
        with a.out.open("a") as f:
            f.write(json.dumps({"part": "launch_evidence", "arm": a.arm, "lines": lines[-40:]}) + "\n")
        shutil.rmtree(a.scratch, ignore_errors=True)
    print(json.dumps({"complete": True, "arm": a.arm, "ctx": a.ctx, "rep": a.rep}))


if __name__ == "__main__":
    main()
