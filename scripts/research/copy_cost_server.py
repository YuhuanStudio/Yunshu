"""Research-only CLI wrapper for the request-local copy cost policy.

The curve must come from this backend's probe_wide_verify run. Component
verify timings plus --head-ms initialize the policy; readback-to-readback
wall observations replace that prior. This wrapper does not add a setting.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--cost-curve", type=Path, required=True)
    ap.add_argument("--head-ms", type=float, default=2.7)
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "python",
    )
    args, cli_args = ap.parse_known_args()
    sys.path.insert(0, str(args.source_dir))
    rows = [json.loads(line) for line in args.cost_curve.read_text().splitlines()]
    if not rows[-1].get("complete") or not rows[-1].get("success"):
        raise ValueError("cost curve is incomplete")
    curves = {}
    for row in rows[:-1]:
        if row.get("shape") == "chain-verifier":
            curves.setdefault(row["context"], {})[row["T"]] = (
                row["eval_ms"] + args.head_ms
            )

    from yunshu_engine import mtp_lane
    from yunshu_engine.copy_cost import CopyCosts
    from yunshu_engine.copy_drafter import CopyDrafter

    class CostCopyDrafter(CopyDrafter):
        def extend(self, tokens):
            if not self.ctx:
                context = min(curves, key=lambda n: abs(n - len(tokens)))
                self.costs = CopyCosts(curves[context], curves[context][6])
            super().extend(tokens)

    mtp_lane.CopyDrafter = CostCopyDrafter
    print(
        f"Copy cost policy: {args.cost_curve}, head prior {args.head_ms} ms", flush=True
    )
    from yunshu_cli import main as cli

    sys.argv = ["yunshu", *cli_args]
    return cli()


if __name__ == "__main__":
    raise SystemExit(main())
