"""Research-only CLI wrapper for the request-local copy cost policy.

The curve must come from this backend's probe_wide_verify run. Component
verify timings plus --head-ms initialize the policy; readback-to-readback
wall observations replace that prior. This wrapper does not add a setting.
"""

import argparse
import json
import sys
from pathlib import Path


def make_drafter_class(curves=None):
    """Keep the rejected policy opt-in only in the research process."""
    from yunshu_engine.copy_cost import MAX_PRICED_ROWS, CopyCosts, costs_for_context
    from yunshu_engine.copy_drafter import CopyDrafter

    class CostCopyDrafter(CopyDrafter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_draft = min(self.max_draft, MAX_PRICED_ROWS - 1)

        def extend(self, tokens):
            if not self.ctx:
                if curves is None:
                    self.costs = costs_for_context(len(tokens))
                else:
                    buckets = sorted(curves)
                    context = next(
                        (n for n in buckets if len(tokens) <= n), buckets[-1]
                    )
                    self.costs = CopyCosts(curves[context], curves[context][6])
            super().extend(tokens)

    return CostCopyDrafter


def main():
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument("--cost-curve", type=Path)
    choice.add_argument("--builtin-costs", action="store_true")
    ap.add_argument("--head-ms", type=float, default=2.7)
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "python",
    )
    args, cli_args = ap.parse_known_args()
    sys.path.insert(0, str(args.source_dir))
    curves = None
    if args.cost_curve is not None:
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

    mtp_lane.CopyDrafter = make_drafter_class(curves)
    print(
        f"Copy cost policy: {args.cost_curve or 'builtin research table'}, head prior {args.head_ms} ms",
        flush=True,
    )
    from yunshu_cli import main as cli

    sys.argv = ["yunshu", *cli_args]
    return cli()


if __name__ == "__main__":
    raise SystemExit(main())
