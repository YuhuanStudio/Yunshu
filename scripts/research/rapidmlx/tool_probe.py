"""Verify actual forced-tool requests after API fixes; GPU work goes through gpuq."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import agent_shapes
import head_to_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), "choose a unique output"
    assert (Path(args.model) / "config.json").is_file()
    assert Path(head_to_head.PYTHONS["yunshu"]).is_file()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        args.output.write_text(json.dumps({"complete": True, "dry_run": True}) + "\n")
        return
    results = []

    def write(row):
        row["device"] = "m5"
        with args.output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "evidence"}), flush=True)
        if row.get("case") == "physical_memory":
            with __import__("urllib.request", fromlist=["urlopen"]).urlopen(
                "http://127.0.0.1:18998/v1/models"
            ) as response:
                model = json.load(response)["data"][0]["id"]
            result = agent_shapes.run("http://127.0.0.1:18998", model)
            raw = args.output.with_suffix(".shapes.json")
            raw.write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            write(
                {
                    "case": "agent_shapes",
                    "passed": result["passed"],
                    "total": result["total"],
                    "raw": str(raw),
                }
            )

    error = None
    try:
        head_to_head.run_arm(
            SimpleNamespace(
                model=args.model,
                output=args.output,
                sizes=[128],
                tokens=64,
                rapid_flags=[],
                tool_eval=False,
            ),
            "yunshu",
            0,
            128,
            write,
        )
    except Exception as exc:
        error = repr(exc)
    success = bool(results) and error is None
    if args.require_all:
        success = success and all(r["passed"] == r["total"] for r in results)
    write(
        {
            "complete": True,
            "success": success,
            "error": error,
            "failures": int(not success),
        }
    )
    raise SystemExit(not success)


if __name__ == "__main__":
    main()
