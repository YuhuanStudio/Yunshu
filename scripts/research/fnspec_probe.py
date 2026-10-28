"""Flash-Next seeded sampled decode, one server arm: speculative (MTP) or spec off.

  fnspec_probe.py --arm on|off --model DIR --out arm.json [--ctx 4096 32768]

One server per job (the 88 GiB oQ3e pack must be the only resident model; gpuq runs a job's
processes to completion before the next job loads). Every cell is a seeded sampled request;
the digest covers text, per-token logprobs and the finish reason, so `fnspec_probe.py
--compare on.json off.json` can state identity. Any incomplete stream fails the job at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLING = {
    "t08": dict(temperature=0.8, top_p=1.0, top_k=0, seed=17),
    "qwen": dict(temperature=1.0, top_p=0.95, top_k=20, seed=17),
}
CELLS = [  # (name, sampling, logprobs)
    ("lp-t08", "t08", True),
    ("lp-qwen", "qwen", True),
    ("nolp-t08", "t08", False),
]
ENGAGED = re.compile(
    r"Exact speculative request engaged: (\w+) constrained=(\w+) logprobs=(\w+)"
)


def digest(row: dict) -> str:
    """Identity of a response: text, per-token logprobs, finish reason."""
    payload = json.dumps(
        [row.get("_text"), row.get("lp_tokens"), row.get("lps"), row.get("finish")],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_row(row: dict, tokens: int, logprobs: bool) -> None:
    if row.get("finish") != "length" or row.get("ct") != tokens:
        raise RuntimeError(
            f"cell did not fill its token budget: {row.get('finish')} ct={row.get('ct')}"
        )
    if logprobs and (not row.get("lps") or len(row["lps"]) < tokens - 1):
        raise RuntimeError("logprob stream is incomplete")


def summarize(row: dict) -> dict:
    spec = (row.get("xy") or {}).get("speculative") or {}
    return dict(
        digest=digest(row),
        ct=row["ct"],
        pt=row["pt"],
        cached=row["cached"],
        ttft_s=row["ttft_s"],
        dec_tps=row["dec_tps"],
        spec=spec,
        accept_per_round=(
            round((row["ct"] - 1) / spec["rounds"], 3) if spec.get("rounds") else None
        ),
    )


def compare(on: dict, off: dict) -> tuple[bool, list[str]]:
    """(all cells identical and spec engaged in the on arm, table lines)."""
    lines = [
        "| cell | digest on | digest off | equal | tok/s on | tok/s off | tok/round | engaged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ok = True
    for key, a in on["cells"].items():
        b = off["cells"].get(key)
        if b is None:
            ok = False
            lines.append(f"| {key} | {a['digest']} | missing | NO | | | | |")
            continue
        engaged = bool(a["spec"].get("rounds"))
        equal = a["digest"] == b["digest"] and a["ct"] == b["ct"]
        ok &= equal and engaged
        lines.append(
            f"| {key} | {a['digest']} | {b['digest']} | {'yes' if equal else 'NO'} | "
            f"{a['dec_tps']} | {b['dec_tps']} | {a['accept_per_round']} | {'yes' if engaged else 'NO'} |"
        )
    return ok, lines


def run(args) -> dict:
    import tfbench as t

    env = {"YUNSHU_VLM_DRAFT": "mtp" if args.arm == "on" else "off"}
    s = t.Srv("yunshu", env, f"fnspec-{args.arm}", model=args.model)
    out = dict(
        arm=args.arm,
        model=args.model,
        mode=getattr(s, "engaged_spec_mode", None),
        cells={},
        tokens=args.tokens,
    )
    try:
        t.send(s.url, t.req(s.model, "Say hi.", 8))
        for ctx in args.ctx:
            prompt = (
                t.load_prompt(f"code-{ctx}")
                if (t.WORK / "prompts" / f"code-{ctx}.txt").exists()
                else t.make_prompt("code", ctx, 1)
            )
            for name, sampling, logprobs in CELLS:
                for rep in range(args.reps):
                    body = t.req(
                        s.model,
                        prompt,
                        args.tokens,
                        extra=dict(SAMPLING[sampling], logprobs=logprobs),
                    )
                    t0 = time.time()
                    row = t.send(s.url, body)
                    check_row(row, args.tokens, logprobs)
                    cell = summarize(row)
                    key = f"{ctx}-{name}-r{rep}"
                    out["cells"][key] = cell
                    print(
                        f"{key}: {cell['dec_tps']} tok/s digest={cell['digest']} spec={cell['spec']} wall={time.time() - t0:.0f}s",
                        flush=True,
                    )
        log = s.log.read_text(errors="replace")
        out["engaged_lines"] = ENGAGED.findall(log)[:6]
        out["reasons"] = sorted(set(re.findall(r"spec_reason=(\w+)", log)))
        out["mem_peak_gib"] = getattr(s.mem, "peak_gib", None)
        out["swap"] = "swap" in log.lower() and "swapped" in log.lower()
    finally:
        s.kill()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--model")
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 32768])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("ON", "OFF"))
    a = ap.parse_args(argv)
    if a.compare:
        ok, lines = compare(*(json.loads(Path(p).read_text()) for p in a.compare))
        print("\n".join(lines))
        print("IDENTICAL+ENGAGED" if ok else "FAIL")
        return 0 if ok else 1
    if not (a.arm and a.model and a.out):
        ap.error("--arm, --model and --out are required")
    out = run(a)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
