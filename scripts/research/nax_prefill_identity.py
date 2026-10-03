"""Cold/partial/full APC identity under one research prefill arithmetic ID."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm", choices=("combo", "lane128", "tile128", "production"), required=True
    )
    p.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--kinds", nargs="+", default=["prose", "code"])
    p.add_argument("--model")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    import apc_restore_identity as identity
    import tfbench as t

    original_assert = identity.assert_identity
    tiny = bool(a.model and "27B" not in a.model)
    if tiny:

        def tiny_assert(result):
            # Generic bf16 tiny targets do not install canonical partial
            # checkpoints. Validate full hit and both outputs without claiming
            # that a partial miss exercised restore.
            for mode in ("partial", "full"):
                record = result[mode]
                if not (record["tokens_equal"] and record["logprobs_equal"]):
                    raise RuntimeError("tiny output identity failed")
            if not result["full"]["cached"]:
                raise RuntimeError("tiny full restore did not engage")

        identity.assert_identity = tiny_assert
    if a.model:
        t.M = a.model
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    launcher = a.out.with_suffix(".launcher.py")
    launcher.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0,{str(Path(__file__).parent)!r})\nimport nax_prefill_dispatch as dispatch\ndispatch.install({a.arm!r})\nfrom yunshu_engine.kernels import lane_linear\noriginal_id=lane_linear.prefill_kernel_id\nlane_linear.prefill_kernel_id=lambda: original_id()+'+research-{a.arm}'\nfrom yunshu_cli import main\nmain()\n"
    )
    if a.arm == "production":
        launcher.write_text(
            f"#!{sys.executable}\nfrom yunshu_cli import main\nmain()\n"
        )
    launcher.chmod(0o700)
    t.YUNSHU_BIN = str(launcher)
    t.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    children = []
    original_popen = subprocess.Popen

    def tracked(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    t.subprocess.Popen = tracked
    failures = []
    try:
        with a.out.open("w") as out:
            for ctx in a.contexts:
                for kind in a.kinds:
                    for turn2 in (False, True):
                        target = a.out.with_name(
                            f"{a.out.stem}-{ctx}-{kind}-{'chat' if turn2 else 'suffix'}.jsonl"
                        )
                        t.OUT = a.out.parent / target.stem
                        sys.argv = [
                            "apc_restore_identity",
                            "--ctx",
                            str(ctx),
                            "--kind",
                            kind,
                            "--out",
                            str(target),
                            "--env",
                            "YUNSHU_VLM_DRAFT=off" if tiny else "YUNSHU_VLM_DRAFT=mtp",
                            "--env",
                            "YUNSHU_VLM_APC_DISK=0",
                        ] + (["--turn2"] if turn2 else [])
                        error = None
                        try:
                            identity.main()
                            if a.arm == "production" and not tiny:
                                for side in ("A", "B"):
                                    log = (
                                        t.OUT / "out" / f"server-apcid-{side}-{ctx}.log"
                                    )
                                    if (
                                        "NAX prefill engaged: rows=2048"
                                        not in log.read_text()
                                    ):
                                        raise RuntimeError(
                                            f"production prefill not engaged in {log}"
                                        )
                        except Exception as exc:
                            error = repr(exc)
                        rows = (
                            [json.loads(s) for s in target.read_text().splitlines()]
                            if target.exists()
                            else []
                        )
                        success = (
                            error is None
                            and bool(rows)
                            and rows[-1].get("phase") == "complete"
                            and rows[-1].get("success") is True
                        )
                        record = dict(
                            ctx=ctx,
                            kind=kind,
                            turn2=turn2,
                            success=success,
                            error=error,
                            result=str(target),
                            partial_hit_required=not tiny,
                        )
                        out.write(json.dumps(record) + "\n")
                        out.flush()
                        print(json.dumps(record), flush=True)
                        if not success:
                            failures.append(record)
                        for child in children:
                            if child.poll() is None:
                                child.terminate()
                                child.wait(timeout=30)
            out.write(
                json.dumps(
                    dict(phase="complete", success=not failures, failures=failures)
                )
                + "\n"
            )
    finally:
        identity.assert_identity = original_assert
        t.subprocess.Popen = original_popen
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
        launcher.unlink(missing_ok=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
