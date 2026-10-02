"""Measure the storage devices an APC tier could live on and print the break-even against re-prefill.

No model, no GPU: it probes each directory with ``yunshu_engine.apc_storage.probe_device`` (F_NOCACHE
sequential read / write, 4 KiB read latency) and, for a checkpoint of ``STATE + KV * tokens`` bytes,
compares the restore time (raw and zstd-encoded) with the prefill time of the same tokens.

    tier_breakeven.py --dir internal=~/.yunshu/cache/probe --dir tb4=/Volumes/P5Plus/x \
        --sim hdd=180/12 --sim nas=110/3 --prefill-tps 885 --out breakeven.json

Checkpoint sizes of the Qwen3.8-27B (measured by tier_roofline.py): 154 MB of recurrent state per
checkpoint, 64 KiB of attention K/V per token; zstd (byte-plane shuffled) ratio 1.37 on that mix,
chunk-parallel decode about 10 GB/s.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from yunshu_engine.apc_storage import probe_device  # noqa: E402

STATE = 154e6
KV_PER_TOKEN = 65536
ZSTD_RATIO = 1.37
ZSTD_DECODE_BPS = 10e9
PREFIXES = (1024, 8192, 32768, 131072)


def row(name, prof, prefill_tps):
    out = {
        "device": name,
        "read_MBps": round(prof.read_bps / 1e6),
        "write_MBps": round(prof.write_bps / 1e6),
        "latency_ms": round(prof.latency_s * 1e3, 2),
        "prefixes": {},
    }
    for n in PREFIXES:
        size = STATE + KV_PER_TOKEN * n
        raw = prof.latency_s + size / prof.read_bps
        enc = (
            prof.latency_s + size / ZSTD_RATIO / prof.read_bps + size / ZSTD_DECODE_BPS
        )
        pre = n / prefill_tps
        out["prefixes"][n] = {
            "bytes_MB": round(size / 1e6),
            "restore_raw_s": round(raw, 2),
            "restore_zstd_s": round(enc, 2),
            "prefill_s": round(pre, 2),
            "speedup_raw": round(pre / raw, 1),
            "speedup_best": round(pre / min(raw, enc), 1),
            "best": "zstd" if enc < raw else "raw",
        }
    # break-even read bandwidth for a 32K checkpoint
    size = STATE + KV_PER_TOKEN * 32768
    out["breakeven_MBps_32k"] = round(size / (32768 / prefill_tps) / 1e6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", action="append", default=[], help="NAME=PATH (a real directory)"
    )
    ap.add_argument(
        "--sim", action="append", default=[], help="NAME=MBps/ms (simulated device)"
    )
    ap.add_argument("--mb", type=int, default=1024)
    ap.add_argument("--prefill-tps", type=float, default=700.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for spec in a.dir:
        name, _, path = spec.partition("=")
        p = Path(path).expanduser()
        print("probing", name, p, flush=True)
        try:
            prof = probe_device(p, mb=a.mb)
        finally:
            shutil.rmtree(p, ignore_errors=True)
        rows.append(row(name, prof, a.prefill_tps))
        print(json.dumps(rows[-1]), flush=True)
    base = (
        Path(a.dir[0].partition("=")[2]).expanduser()
        if a.dir
        else Path("/tmp/yunshu-sim")
    )
    for spec in a.sim:
        name, _, rest = spec.partition("=")
        bw, _, ms = rest.partition("/")
        p = base.parent / f"sim-{name}"
        prof = probe_device(p, mb=16, sim=(float(bw) * 1e6, float(ms) / 1e3))
        shutil.rmtree(p, ignore_errors=True)
        rows.append(row(f"{name} (simulated {bw} MB/s, {ms} ms)", prof, a.prefill_tps))
        print(json.dumps(rows[-1]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(
        "device               read   write  lat   | restore s (raw/zstd) vs prefill s"
    )
    for r in rows:
        cells = "  ".join(
            f"{n // 1024}K:{v['restore_raw_s']}/{v['restore_zstd_s']} vs {v['prefill_s']}"
            for n, v in r["prefixes"].items()
        )
        print(
            f"{r['device'][:20]:20} {r['read_MBps']:6} {r['write_MBps']:6} {r['latency_ms']:5} | {cells}"
        )


if __name__ == "__main__":
    main()
