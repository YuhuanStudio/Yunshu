"""Draft-LM / Gemma-4 assistant speculation A/B on the round driver's generic layer (T4); gpuq only.

One server per rep with the drafter's setting (``--drafter assistant`` sets YUNSHU_GEMMA4_ASSISTANT,
``--drafter draftlm`` sets YUNSHU_DRAFT_MODEL to ``--draft-path``). Per case the same greedy request
runs with ``spec_decode`` off and on (order alternates by rep). Records engaged / drafted / accepted
(from the server's row log line), decode tok/s, the served ``usage.prompt_tokens`` and how long the
two texts agree. ``prose_long`` has a prompt past the sliding window of a rotating-cache model: the
drafter must stay off there and the two arms must then be identical. CPU-only ``--dry-run``.
"""

import argparse
import json
import math
import re
import shutil
import statistics
from pathlib import Path

CASES = ("prose_short", "code_short", "prose_long")
ASKS = {
    "prose_short": "Explain how a refrigerator works, step by step, in detail.",
    "code_short": "Write a Python function that merges overlapping intervals, with a docstring and three unit tests.",
}
ROW_LOG = re.compile(
    r"(?:gemma4-assistant|draft-lm) row: (\d+) windows, (\d+) drafted, (\d+) accepted, (\d+) tokens"
)
MIN_MATCH_CHARS = 64


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--drafter", choices=["assistant", "draftlm"], required=True)
    p.add_argument("--draft-path", default="")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--long-chars", type=int, default=6000)
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


def parse_row_log(log: str):
    """(windows, drafted, accepted, tokens) of the last drafter row in ``log``, or None."""
    found = ROW_LOG.findall(log)
    return tuple(int(x) for x in found[-1]) if found else None


def validate_records(rows, reps, cases=CASES):
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
            or not isinstance(r.get("pt"), int)
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
            if case.endswith("_long"):
                if on.get("engaged"):
                    return False, "the drafter engaged past the sliding window"
                if (on["sha"], on["ct"]) != (off["sha"], off["ct"]):
                    return False, "plain row is not identical with spec off"
            else:
                if not on.get("engaged"):
                    return False, "the drafter never engaged on a short prompt"
                if int(on.get("match", 0)) < MIN_MATCH_CHARS:
                    return False, "spec on diverged from spec off almost at once"
    return True, ""


def main():
    a = parser().parse_args()
    if a.reps < 1 or a.tokens < 32 or a.long_chars < 16:
        raise ValueError("positive reps, tokens >= 32, long-chars >= 16 required")
    if a.drafter == "draftlm" and not a.draft_path:
        raise ValueError("--drafter draftlm needs --draft-path")
    if a.dry_run:
        print(json.dumps(dict(complete=True, dry_run=True, reps=a.reps)))
        return
    import os

    os.environ["TFB_SKIP_ENGAGED"] = "1"
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
            if a.drafter == "assistant":
                env["YUNSHU_GEMMA4_ASSISTANT"] = a.draft_path or ""
            else:
                env["YUNSHU_DRAFT_MODEL"] = a.draft_path
            server = t.Srv("yunshu", env, f"draft-{a.out.stem}-{rep}", model=a.model)
            try:
                for case in a.cases:
                    text = (
                        filler(a.long_chars, 101)
                        + "\n\nSummarize the text above in a long, detailed essay."
                        if case.endswith("_long")
                        else ASKS[case]
                    )
                    texts, rows = {}, {}
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
                        stats = parse_row_log(log)
                        rows[arm] = dict(
                            phase="decode",
                            case=case,
                            arm=arm,
                            rep=rep,
                            sha=result["sha"],
                            ct=result["ct"],
                            pt=result["pt"],
                            dec_tps=result["dec_tps"],
                            engaged=" window engaged" in log
                            and ("gemma4-assistant" in log or "draft-lm" in log),
                            windows=stats[0] if stats else 0,
                            drafted=stats[1] if stats else 0,
                            accepted=stats[2] if stats else 0,
                            server_log=str(server.log),
                        )
                    rows["on"]["match"] = common_prefix(texts["off"], texts["on"])
                    rows["on"]["exact"] = texts["off"] == texts["on"]
                    for arm in ("off", "on"):
                        write(rows[arm])
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
