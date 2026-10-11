"""Run the existing seven capability probes in one bounded gpuq admission.

Each arm retains its own output and return code. Child processes release their
MLX models between arms; failures are recorded without dropping other evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--rerank-reference", required=True)
    p.add_argument("--model-root", default="/Volumes/P5Plus/models")
    p.add_argument(
        "--reference",
        default="/Volumes/P5Plus/yunshu-build/codex/priorfix/gemma-ref/ref.json",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def plan(args):
    scripts = Path(__file__).parent
    directory = Path(args.out).with_suffix("")
    arms = []
    for model in (
        "embeddinggemma-2-bf16",
        "embeddinggemma-2-4bit",
        "embeddinggemma-2-bf16-multishard",
    ):
        arms.append(
            (
                model,
                [
                    sys.executable,
                    str(scripts / "priorfix_embedding_parity.py"),
                    "--model",
                    str(Path(args.model_root) / model),
                    "--reference",
                    args.reference,
                ],
            )
        )
    for kind in ("retrieval", "classifier", "diffusion", "omni"):
        argv = [
            sys.executable,
            str(scripts / "priorfix_runtime_parity.py"),
            "--kind",
            kind,
        ]
        if kind == "retrieval":
            argv += ["--rerank-reference", args.rerank_reference]
        arms.append((kind, argv))
    return [
        (name, argv + ["--out", str(directory / (name + ".jsonl"))])
        for name, argv in arms
    ]


def evidence(path, rc):
    if rc != 0:
        return {"passed": False, "reason": f"child rc={rc}"}
    try:
        result = json.loads(Path(path).read_text().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return {"passed": False, "reason": "missing or malformed child evidence"}
    if not isinstance(result, dict):
        return {"passed": False, "reason": "child evidence must be an object"}
    if (
        result.get("complete") is not True
        or result.get("passed") is not True
        or result.get("device") != "M5"
    ):
        return {
            "passed": False,
            "reason": "incomplete, failed or wrong-device child evidence",
            "result": result,
        }
    return {"passed": True, "result": result}


def admission(args):
    """CPU-check paths and reference syntax before any child can load a model."""
    for path in (args.reference, args.rerank_reference):
        if not Path(path).is_file():
            raise ValueError(f"missing reference: {path}")
    compile(Path(args.rerank_reference).read_text(), args.rerank_reference, "exec")
    json.loads(Path(args.reference).read_text())
    for _, argv in plan(args):
        compile(Path(argv[1]).read_text(), argv[1], "exec")
        if "--model" in argv:
            model = Path(argv[argv.index("--model") + 1])
            if not model.is_dir():
                raise ValueError(f"missing model: {model}")


def run(args, runner=subprocess.run):
    arms = plan(args)
    Path(args.out).with_suffix("").mkdir(parents=True, exist_ok=True)
    results = {}
    for name, argv in arms:
        out = Path(argv[-1])
        out.unlink(missing_ok=True)  # a failed retry cannot accept an earlier result
        completed = runner(argv, check=False)
        results[name] = {
            "rc": completed.returncode,
            "output": str(out),
            **evidence(out, completed.returncode),
        }
        print(
            json.dumps(
                {
                    "arm": name,
                    "rc": completed.returncode,
                    "passed": results[name]["passed"],
                }
            ),
            flush=True,
        )
    return {
        "complete": True,
        "passed": all(row["passed"] for row in results.values()),
        "device": "M5",
        "arms": results,
    }


def main():
    args = parser().parse_args()
    admission(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "arms": plan(args)}))
        return 0
    result = run(args)
    Path(args.out).write_text(json.dumps(result) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
