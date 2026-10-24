"""Interleaved sampled decode A/B using tfbench's pinned-source server driver.

Run through gpuq --quiet only after identity gates. --dry-run is CPU-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SAMPLING = {
    "greedy": dict(temperature=0, seed=17),
    "sampled": dict(temperature=0.7, top_p=1.0, top_k=0, seed=17),
    "qwen": dict(temperature=1.0, top_p=0.95, top_k=20, seed=17),
}


def arm_order(reps):
    return [
        arm
        for rep in range(reps)
        for arm in (("base", "cand") if rep % 2 == 0 else ("cand", "base"))
    ]


def validate_row(row):
    if row.get("cached") != 0:
        raise RuntimeError("timing arm was an APC hit or lacks cached-token accounting")
    if row.get("ct", 0) < 32 or not row.get("dec_tps"):
        raise RuntimeError("decode arm did not generate enough tokens")
    spec = (row.get("xy") or {}).get("speculative") or {}
    if not spec.get("rounds"):
        raise RuntimeError("missing speculative round accounting")
    return (row["ct"] - 1) / spec["rounds"]


def verify_engagement(arm, sampling, request_log):
    keyed = "DFlash fast verification: keyed=True" in request_log
    if arm == "cand" and sampling != "greedy" and not keyed:
        raise RuntimeError("sampled candidate cell did not engage the fast keyed tree")
    if keyed:
        return "fast-keyed"
    if "DFlash fast verification: keyed=False" in request_log:
        return "fast-greedy"
    return "chain"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-src", type=Path, required=True)
    p.add_argument("--cand-src", type=Path, required=True)
    p.add_argument("--ctx", type=int, nargs="+", default=[16384, 32768, 131072])
    p.add_argument(
        "--kind", nargs="+", choices=["code", "prose"], default=["code", "prose"]
    )
    p.add_argument(
        "--sampling", nargs="+", choices=list(SAMPLING), default=list(SAMPLING)
    )
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.base_src.resolve() == a.cand_src.resolve():
        p.error("base and candidate must differ")
    if a.dry_run:
        print(
            json.dumps(dict(complete=True, order=arm_order(a.reps), sampling=SAMPLING))
        )
        return
    import tfbench as t

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for index, arm in enumerate(arm_order(a.reps)):
        t.YUNSHU_SRC = str((a.base_src if arm == "base" else a.cand_src).resolve())
        label = f"sampledspec-speed-{a.out.stem}-{index}-{arm}"
        apc = Path("/Volumes/Micron/yunshu-apc") / label
        server = t.Srv(
            "yunshu",
            {
                "YUNSHU_VLM_APC_DISK": "0",
                "YUNSHU_VLM_APC_MEMORY_GB": "0",
                "YUNSHU_VLM_APC_DISK_DIR": str(apc),
                "YUNSHU_VLM_DRAFT": t.D,
                "YUNSHU_NEXT_TURN_PREFILL": "off",
            },
            label,
        )
        try:
            t.send(server.url, t.req(server.model, "Say hi.", 16))
            for ctx in a.ctx:
                for kind in a.kind:
                    for sampling in a.sampling:
                        # Each distinct cell starts from a cold server-side APC entry.
                        prompt = (
                            t.decode_prompt(kind, ctx, True)
                            + f"\nSampling case: {sampling}."
                        )
                        log_start = server.log.stat().st_size
                        row = t.send(
                            server.url,
                            t.req(
                                server.model, prompt, a.tokens, extra=SAMPLING[sampling]
                            ),
                        )
                        committed = validate_row(row)
                        with server.log.open("rb") as log:
                            log.seek(log_start)
                            engaged = verify_engagement(
                                arm, sampling, log.read().decode(errors="replace")
                            )
                        result = {
                            k: v
                            for k, v in row.items()
                            if k not in ("head", "_text", "lp_tokens", "lps")
                        }
                        result.update(
                            arm=arm,
                            index=index,
                            ctx=ctx,
                            kind=kind,
                            sampling=sampling,
                            committed_per_round=committed,
                            engaged=engaged,
                        )
                        with a.out.open("a") as out:
                            out.write(json.dumps(result) + "\n")
                        print(json.dumps(result), flush=True)
            server.verify_spec_mode()
        finally:
            server.kill()
            shutil.rmtree(apc, ignore_errors=True)
    with a.out.open("a") as out:
        out.write(json.dumps(dict(complete=True)) + "\n")


if __name__ == "__main__":
    main()
