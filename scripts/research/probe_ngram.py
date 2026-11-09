"""n-gram speculation A/B on the round driver, through gpuq only; CPU-only --dry-run.

One server tree, one model. Per rep and case the same greedy request runs with ``spec_decode`` off
and on (alternating order across reps, a fresh server per rep). ``--expect engaged``: a generic
layer model with trimmable caches; the drafter must engage on the repetitive cases and its text
must match the plain decode for at least ``MIN_MATCH_CHARS`` characters (the guarantee on the
generic layer is the old verifier's top-1 over the window, not bitwise T=1; the exact-equal flag
is reported). ``--expect plain``: a packed layer or a cache that cannot trim; the drafter must not
engage and the two arms must be token-identical (spec on == off).
"""

import argparse
import json
import math
import shutil
import statistics
from pathlib import Path

CASES = ("repeat_short", "repeat_long", "prose_short")
MIN_MATCH_CHARS = 64
REPEAT_ASK = (
    "\n\nNow write the sentence 'The quick brown fox jumps over the lazy dog near the river bank.' "
    "exactly 60 times, one per line, numbered, and nothing else."
)
PROSE_ASK = "\n\nSummarize the text above in a long, detailed essay."
CHARS = {"repeat_short": 1500, "repeat_long": None, "prose_short": 1500}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--expect", choices=["engaged", "plain"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--long-chars", type=int, default=130000)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--dry-run", action="store_true")
    return p


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def validate_records(rows, reps, expect, cases=CASES):
    if (
        not rows
        or rows[-1].get("complete") is not True
        or rows[-1].get("success") is not True
    ):
        return False, "missing successful completion"
    cells = [r for r in rows if r.get("phase") == "decode"]
    keys = {(r.get("arm"), r.get("rep"), r.get("case")) for r in cells}
    if len(cells) != 2 * reps * len(cases) or keys != {
        (a, n, c) for a in ("off", "on") for n in range(reps) for c in cases
    }:
        return False, "missing or duplicate paired repetitions"
    for r in cells:
        if (
            not isinstance(r.get("dec_tps"), (float, int))
            or not math.isfinite(r["dec_tps"])
            or r["dec_tps"] <= 0
            or not isinstance(r.get("ct"), int)
            or r["ct"] < 32
            or not r.get("sha")
        ):
            return False, "invalid decode evidence"
    for case in cases:
        for rep in range(reps):
            off = next(
                r
                for r in cells
                if (r["arm"], r["rep"], r["case"]) == ("off", rep, case)
            )
            on = next(
                r for r in cells if (r["arm"], r["rep"], r["case"]) == ("on", rep, case)
            )
            if off.get("engaged"):
                return False, "the drafter engaged on a request that did not ask for it"
            if expect == "plain":
                if on.get("engaged"):
                    return False, "the drafter engaged where it must stay off"
                if (on["sha"], on["ct"]) != (off["sha"], off["ct"]):
                    return False, "plain row is not identical with spec off"
            else:
                if case.startswith("repeat") and not on.get("engaged"):
                    return False, "the drafter never engaged on a repetitive prompt"
                if int(on.get("match", 0)) < MIN_MATCH_CHARS:
                    return False, "spec on diverged from spec off almost at once"
    return True, ""


def main():
    a = parser().parse_args()
    if a.reps < 1 or a.tokens < 32 or a.long_chars < 16:
        raise ValueError("positive reps, tokens >= 32, long-chars >= 16 required")
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
            t.YUNSHU_SRC = str(a.src / "python")
            env = dict(
                YUNSHU_VLM_APC_DISK="0",
                YUNSHU_VLM_APC_MEMORY_GB="0",
                YUNSHU_NEXT_TURN_PREFILL="off",
                YUNSHU_CONSOLE="0",
                YUNSHU_NGRAM_DEFAULT="0",
            )
            server = t.Srv("yunshu", env, f"ngram-{a.out.stem}-{rep}", model=a.model)
            try:
                for case in a.cases:
                    chars = CHARS[case] or a.long_chars
                    ask = PROSE_ASK if case == "prose_short" else REPEAT_ASK
                    text = filler(chars, 101) + ask
                    texts = {}
                    for arm in ("off", "on") if rep % 2 == 0 else ("on", "off"):
                        body = dict(
                            model=server.model,
                            messages=[dict(role="user", content=text)],
                            max_tokens=a.tokens,
                            temperature=0.0,
                        )
                        if arm == "on":
                            body["spec_decode"] = True
                        start = server.log.stat().st_size
                        result = t.send(server.url, body)
                        log = server.log.read_bytes()[start:].decode(errors="replace")
                        texts[arm] = result["_text"]
                        write(
                            dict(
                                phase="decode",
                                case=case,
                                arm=arm,
                                rep=rep,
                                sha=result["sha"],
                                ct=result["ct"],
                                dec_tps=result["dec_tps"],
                                engaged="n-gram window engaged" in log,
                                server_log=str(server.log),
                            )
                        )
                    on = records[-1] if records[-1]["arm"] == "on" else records[-2]
                    on["match"] = common_prefix(texts["off"], texts["on"])
                    on["exact"] = texts["off"] == texts["on"]
            finally:
                server.kill()
                shutil.rmtree(server.home, ignore_errors=True)
        final = dict(complete=True, success=True)
        ok, reason = validate_records([*records, final], a.reps, a.expect, a.cases)
        if not ok:
            raise RuntimeError(reason)
        final["median_tps"] = {
            case: {
                arm: statistics.median(
                    r["dec_tps"]
                    for r in records
                    if r["arm"] == arm and r["case"] == case
                )
                for arm in ("off", "on")
            }
            for case in a.cases
        }
        final["exact"] = {
            case: all(
                r.get("exact")
                for r in records
                if r["arm"] == "on" and r["case"] == case
            )
            for case in a.cases
        }
        write(final)


if __name__ == "__main__":
    main()
