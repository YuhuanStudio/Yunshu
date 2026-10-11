"""Greedy output hash + decode speed per YUNSHU_GIL_SWITCH_MS value (1K prose and code, 3 prompts each).

Run only through gpuq: one job per --ms value, then compare the shas and tok/s.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tfbench as t

    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.OUT = a.out.parent / a.out.stem
    t.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    server = t.Srv(
        "yunshu", {"YUNSHU_VLM_APC_DISK": "0", "YUNSHU_GIL_SWITCH_MS": a.ms}, a.out.stem
    )
    rows = []
    try:
        t.send(server.url, t.req(server.model, "Say hi.", 16, seed=1))
        for kind in ("prose", "code"):
            text = t.load_prompt(f"{kind}-1024")
            for k in range(3):
                r = t.send(
                    server.url,
                    t.req(server.model, f"[{k}]\n" + text + t.LONG_ASK, 384, seed=7),
                )
                r.pop("_text")
                rows.append(
                    dict(
                        ms=a.ms,
                        kind=kind,
                        k=k,
                        sha=r["sha"],
                        dec_tps=r["dec_tps"],
                        ttft_s=r["ttft_s"],
                        ct=r["ct"],
                    )
                )
                print(json.dumps(rows[-1]), flush=True)
        if any(r["ct"] < 300 for r in rows):
            raise RuntimeError("short generation")
        a.out.write_text("\n".join(map(json.dumps, rows)) + "\n")
    finally:
        server.kill()


if __name__ == "__main__":
    main()
