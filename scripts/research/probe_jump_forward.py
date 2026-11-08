"""JSON-schema forced-window A/B, through gpuq only; CPU-only --dry-run.

Uses tfbench's existing structured-output schema, SSE accounting and server lifecycle.
Every pair must have identical output digests and token counts. Report client decode
rate (completion tokens minus first / first content through stream completion).
"""

import argparse
import json
import math
import shutil
import statistics
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-src", type=Path, required=True)
    p.add_argument("--cand-src", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", help="Explicit drafter mode/path; tiny pilots use off")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--ctx-chars", type=int, default=32000)
    p.add_argument(
        "--cases",
        nargs="+",
        choices=["schema", "forced_literal"],
        default=["schema", "forced_literal"],
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_records(rows, reps, cases=("schema", "forced_literal")):
    if (
        not rows
        or rows[-1].get("complete") is not True
        or rows[-1].get("success") is not True
    ):
        return False, "missing successful completion"
    cells = [r for r in rows if r.get("phase") == "decode"]
    if len(cells) != 2 * reps * len(cases) or {
        (r.get("arm"), r.get("rep"), r.get("case")) for r in cells
    } != {
        (a, n, case) for a in ("base", "cand") for n in range(reps) for case in cases
    }:
        return False, "missing or duplicate paired repetitions"
    for r in cells:
        if (
            not isinstance(r.get("dec_tps"), (float, int))
            or not math.isfinite(r["dec_tps"])
            or r["dec_tps"] <= 0
        ):
            return False, "invalid decode rate"
        if (
            not r.get("sha")
            or not isinstance(r.get("ct"), int)
            or r["ct"] < 2
            or not r.get("schema_valid")
        ):
            return False, "missing identity or schema evidence"
        if (
            r["arm"] == "cand"
            and r["case"] == "forced_literal"
            and not r.get("engaged")
        ):
            return False, "forced window never engaged"
    for case in cases:
        if len({(r["sha"], r["ct"]) for r in cells if r["case"] == case}) != 1:
            return False, "output identity mismatch"
    return True, ""


def main():
    a = parser().parse_args()
    if a.reps < 1 or a.tokens < 16 or a.ctx_chars < 16:
        raise ValueError("positive reps, tokens >= 16, ctx-chars >= 16 required")
    if a.dry_run:
        print(json.dumps(dict(complete=True, dry_run=True, reps=a.reps)))
        return
    try:
        from . import tfbench as t
        from .conc_probe import filler
    except ImportError:
        import tfbench as t
        from conc_probe import filler
    t.OUT = a.out.parent / (a.out.stem + "-servers")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with a.out.open("x") as output:

        def write(row):
            records.append(row)
            line = json.dumps(row)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        for rep in range(a.reps):
            for arm in ("base", "cand") if rep % 2 == 0 else ("cand", "base"):
                t.YUNSHU_SRC = str(a.base_src if arm == "base" else a.cand_src)
                env = dict(
                    YUNSHU_JUMP_FORWARD="1" if arm == "cand" else "0",
                    YUNSHU_VLM_APC_DISK="0",
                    YUNSHU_VLM_APC_MEMORY_GB="0",
                    YUNSHU_NEXT_TURN_PREFILL="off",
                    YUNSHU_CONSOLE="0",
                )
                if a.draft is not None:
                    env["YUNSHU_VLM_DRAFT"] = a.draft
                server = t.Srv(
                    "yunshu", env, f"jump-{a.out.stem}-{arm}-{rep}", model=a.model
                )
                try:
                    for case in a.cases:
                        body = dict(
                            model=server.model,
                            messages=[
                                dict(role="user", content=filler(a.ctx_chars, 101))
                            ],
                            max_tokens=a.tokens,
                            temperature=0.0,
                        )
                        t.family_apply(body, "json-schema", 0, "off", "none")
                        if case == "forced_literal":
                            schema = {
                                "type": "object",
                                "properties": {
                                    "payload": {
                                        "const": "yunshu_jump_forward_lossless_singleton_structural_continuation_"
                                        * 8
                                    }
                                },
                                "required": ["payload"],
                                "additionalProperties": False,
                            }
                            body["response_format"]["json_schema"]["schema"] = schema
                        log_start = server.log.stat().st_size
                        result = t.send(server.url, body)
                        if case == "schema":
                            problems = t.family_problems(
                                "json-schema", result, "off", "none"
                            )
                        else:
                            import jsonschema

                            jsonschema.validate(json.loads(result["_text"]), schema)
                            problems = []
                        if problems:
                            raise RuntimeError(f"structured output invalid: {problems}")
                        log = server.log.read_bytes()[log_start:].decode(
                            errors="replace"
                        )
                        write(
                            dict(
                                phase="decode",
                                case=case,
                                arm=arm,
                                rep=rep,
                                sha=result["sha"],
                                ct=result["ct"],
                                dec_tps=result["dec_tps"],
                                schema_valid=True,
                                engaged="Round driver forced window engaged:" in log,
                                server_log=str(server.log),
                            )
                        )
                        if (
                            arm == "cand"
                            and case == "forced_literal"
                            and not records[-1]["engaged"]
                        ):
                            raise RuntimeError("forced windows did not engage")
                        if (
                            len(
                                {
                                    (r["sha"], r["ct"])
                                    for r in records
                                    if r["case"] == case
                                }
                            )
                            > 1
                        ):
                            raise RuntimeError("output identity mismatch")
                finally:
                    server.kill()
                    shutil.rmtree(server.home, ignore_errors=True)
        final = dict(complete=True, success=True)
        ok, reason = validate_records([*records, final], a.reps, a.cases)
        if not ok:
            raise RuntimeError(reason)
        final["median_tps"] = {
            case: {
                arm: statistics.median(
                    r["dec_tps"]
                    for r in records
                    if r["arm"] == arm and r["case"] == case
                )
                for arm in ("base", "cand")
            }
            for case in a.cases
        }
        write(final)


if __name__ == "__main__":
    main()
