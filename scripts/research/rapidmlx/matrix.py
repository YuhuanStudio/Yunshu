"""Frozen-profile wrapper: interleave Rapid defaults, documented text AR, and Yunshu defaults."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location(
    "head_to_head", SOURCE / "head_to_head.py"
)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
PROFILES = {
    "rapid-default": ("rapid", []),
    "rapid-ar": (
        "rapid",
        [
            "--no-mllm",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--hybrid-cache-entries",
            "8",
            "--no-spec-decode",
        ],
    ),
    "yunshu-default": ("yunshu", []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 8192, 32768])
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES)
    )
    parser.add_argument("--tool-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write(row):
        with args.output.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("output", "requests", "evidence")
                }
            ),
            flush=True,
        )

    if args.dry_run:
        for name in args.profiles:
            py = harness.PYTHONS[PROFILES[name][0]]
            assert Path(py).is_file(), py
        assert (Path(args.model) / "config.json").is_file()
        write(
            {
                "complete": True,
                "dry_run": True,
                "profiles": args.profiles,
                "model": args.model,
            }
        )
        return
    failures = 0
    for rep in range(args.reps):
        order = (
            args.profiles[rep % len(args.profiles) :]
            + args.profiles[: rep % len(args.profiles)]
        )
        for size in args.sizes:
            for profile in order:
                engine, flags = PROFILES[profile]
                arm_dir = args.output.parent / profile
                arm_dir.mkdir(exist_ok=True)
                arm_args = SimpleNamespace(
                    model=args.model,
                    output=arm_dir / "results.jsonl",
                    sizes=args.sizes,
                    tokens=args.tokens,
                    rapid_flags=flags,
                    tool_eval=args.tool_eval,
                    prompt_identity=str(args.output.resolve()),
                )

                def arm_write(row):
                    row["profile"] = profile
                    with arm_args.output.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    write(row)

                try:
                    harness.run_arm(arm_args, engine, rep, size, arm_write)
                except Exception as exc:
                    failures += 1
                    write(
                        {
                            "profile": profile,
                            "engine": engine,
                            "rep": rep,
                            "size": size,
                            "error": repr(exc),
                        }
                    )
    write({"complete": True, "failures": failures})
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
