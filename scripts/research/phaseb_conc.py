"""Phase B HTTP identity/timing cells, reusing conc_probe and sampledspec_conc.

Run via gpuq (speed: --quiet). Output contains hashes, shapes and timings only.
Every deterministic concurrent row is checked against its matched solo request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import statistics
import threading
import time
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-src", type=Path, required=True)
    p.add_argument("--cand-src", type=Path, required=True)
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--mode", choices=["identity", "speed"], default="identity")
    p.add_argument(
        "--ctx-tokens",
        type=int,
        default=None,
        help="served prompt tokens (default 4096)",
    )
    p.add_argument(
        "--ctx-chars",
        type=int,
        default=None,
        help="DEPRECATED alias: characters, converted to tokens",
    )
    p.add_argument(
        "--tokenizer",
        default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp",
        help="tokenizer dir for the length fit",
    )
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument(
        "--draft",
        default=None,
        help="YUNSHU_VLM_DRAFT value: a drafter directory, or mtp (checkpoint MTP head)",
    )
    p.add_argument(
        "--cand-env",
        action="append",
        default=[],
        metavar="K=V",
        help="extra server environment for the cand arm (e.g. YUNSHU_KV_PRECISION=int8)",
    )
    p.add_argument("--join", type=float, default=0.5)
    p.add_argument("--scenarios", nargs="+", default=["burst2", "burst4", "stagger"])
    p.add_argument("--shapes", nargs="+", default=["greedy", "seeded"])
    p.add_argument("--arms", nargs="+", default=["base", "cand"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def plans(scenario, tokens, join):
    if scenario in ("solo", "burst2", "burst4"):
        n = {"solo": 1, "burst2": 2, "burst4": 4}[scenario]
        return [(chr(65 + b), 0.0, tokens, 101 + b) for b in range(n)]
    if scenario == "stagger":
        return [("A", 0.0, tokens, 101), ("B", join, max(16, tokens // 3), 102)]
    raise ValueError(f"unknown scenario: {scenario}")


def resolve_ctx(a):
    """--ctx-tokens (default 4096); the deprecated --ctx-chars is converted with the tokenizer and logged."""
    try:
        from . import prompt_len
    except ImportError:
        import prompt_len
    if a.ctx_chars is not None:
        n = prompt_len.tokens_from_chars(
            prompt_len.load_tokenizer(a.tokenizer), a.ctx_chars
        )
        print(
            f"warning: --ctx-chars {a.ctx_chars} is deprecated; using --ctx-tokens {n}",
            flush=True,
        )
        a.ctx_tokens = n
    elif a.ctx_tokens is None:
        a.ctx_tokens = 4096


def validate(a):
    if a.ctx_tokens is not None and a.ctx_chars is not None:
        raise ValueError("--ctx-chars is a deprecated alias of --ctx-tokens; give one")
    resolve_ctx(a)
    if a.tokens < 4 or a.ctx_tokens < 16 or a.join < 0 or a.rep < 0:
        raise ValueError("invalid request shape")
    if not set(a.shapes) <= {"greedy", "seeded", "agent"} or not a.shapes:
        raise ValueError("unknown or empty sampling shapes")
    if not set(a.arms) <= {"base", "cand", "off"} or not a.arms:
        raise ValueError("unknown or empty arms")
    return [plans(s, a.tokens, a.join) for s in a.scenarios]


def rate(times):
    return (
        (len(times) - 1) / (times[-1] - times[0])
        if len(times) > 1 and times[-1] > times[0]
        else None
    )


def summarize(rows):
    first = min(r["times"][0] for r in rows.values())
    last = max(r["times"][-1] for r in rows.values())
    return dict(
        per_row={
            name: dict(
                ttft=r["ttft"],
                tps=rate(r["times"]),
                tokens=r["usage"].get("completion_tokens"),
                prompt_tokens=r["usage"].get("prompt_tokens"),
                digest=r["digest"],
            )
            for name, r in rows.items()
        },
        aggregate_tps=sum(len(r["times"]) for r in rows.values()) / (last - first)
        if last > first
        else None,
    )


def interval_rates(a, b):
    times = a["times"]
    start, end = b["t0"], b["times"][0]
    return dict(
        # Count over the whole prefill interval: first-to-last arrivals omit
        # stalls at either edge and overstate service during long atoms.
        prefill_tps=sum(start <= t < end for t in times) / (end - start)
        if end > start
        else None,
        overlap_tps=rate([t for t in times if b["times"][0] <= t <= b["times"][-1]]),
        after_tps=rate([t for t in times if t > b["times"][-1]]),
    )


def check_identity(solo, concurrent):
    if solo.keys() != concurrent.keys() or any(
        solo[n]["digest"] != concurrent[n]["digest"] for n in solo
    ):
        raise RuntimeError(
            "concurrent token/logprob digest differs from matched solo request"
        )


def run_cell(server, a, scenario, shape, strict=True):
    try:
        from . import conc_probe, prompt_len
        from . import sampledspec_conc as sampled
    except ImportError:
        import conc_probe
        import prompt_len
        import sampledspec_conc as sampled

    jobs = plans(scenario, a.tokens, a.join)
    tok = prompt_len.load_tokenizer(a.tokenizer)

    def stream(url, b, t_zero, **kw):
        # the label is the served turn-1 length (prompt_len.py); fail the cell on a mismatch
        row = sampled.stream(url, b, t_zero, **kw)
        prompt_len.check_served_length(a.ctx_tokens, row["usage"])
        return row

    def request(salt, ntok):
        prompt = prompt_len.build_text(
            tok,
            conc_probe.filler(a.ctx_tokens * 6, salt),
            a.ctx_tokens,
            suffix="\nReview this design carefully and propose concrete improvements. Reason step by step.",
        )
        body = sampled.body(
            server.model, prompt, ntok, "seeded" if shape == "seeded" else "agent"
        )
        if shape == "greedy":
            body.update(temperature=0.0, logprobs=True)
        if shape == "seeded":
            body["seed"] = 17 + salt
        return body

    solo = {}
    for name, _, ntok, salt in jobs:
        solo[name] = stream(server.url, request(salt, ntok), time.perf_counter())
    t_zero = time.perf_counter()

    started = threading.Event()
    finished = threading.Event()

    def go(job):
        name, delay, ntok, salt = job
        if scenario == "stagger" and name == "B":
            if not started.wait(timeout=1800):
                raise RuntimeError("running row never delivered a token")
            time.sleep(delay)
            if finished.is_set():
                raise RuntimeError(
                    "stagger running row finished before newcomer joined"
                )
        else:
            time.sleep(delay)
        try:
            return name, stream(
                server.url,
                request(salt, ntok),
                t_zero,
                on_first_token=started.set if name == "A" else None,
            )
        finally:
            if name == "A":
                finished.set()
                started.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        concurrent_rows = dict(pool.map(go, jobs))
    same = True
    if shape != "agent":
        # The base arm is today's serving: a concurrent row there is not
        # guaranteed to equal its solo run, so its mismatch is a measurement.
        try:
            check_identity(solo, concurrent_rows)
        except RuntimeError:
            if strict:
                raise
            same = False
    result = dict(
        scenario=scenario,
        shape=shape,
        solo=summarize(solo),
        concurrent=summarize(concurrent_rows),
        identity=shape != "agent" and same,
    )
    if scenario == "stagger":
        result["stagger"] = interval_rates(concurrent_rows["A"], concurrent_rows["B"])
    return result


def speed_verdict(rows, reps=3, tolerance=0.02):
    """Require paired repeated cells; medians decide, missing evidence fails."""
    groups = {}
    for r in rows:
        if "scenario" in r:
            groups.setdefault(
                (r["ctx_tokens"], r["scenario"], r["shape"]), {}
            ).setdefault(r["arm"], {})[r["rep"]] = r
    reasons, numbers = [], []
    if not groups:
        return ["no Phase B speed cells"], []
    for key, arms in sorted(groups.items()):
        if any(set(arms.get(arm, {})) != set(range(reps)) for arm in ("base", "cand")):
            reasons.append(f"{key}: missing paired repetitions")
            continue

        def median(arm, name):
            values = [r["concurrent"][name] for r in arms[arm].values()]
            return (
                statistics.median(values)
                if all(v is not None and v > 0 for v in values)
                else None
            )

        base, cand = median("base", "aggregate_tps"), median("cand", "aggregate_tps")
        if key[1] == "stagger":
            for arm in ("base", "cand"):
                for rep, r in arms[arm].items():
                    if any(
                        r.get("stagger", {}).get(f) is None
                        for f in ("prefill_tps", "overlap_tps", "after_tps")
                    ):
                        reasons.append(
                            f"{key}/{arm}/r{rep}: stagger did not cover prefill,"
                            " overlap and recovery"
                        )
        numbers.append(dict(cell=key, aggregate_tps=[base, cand]))
        if base is None or cand is None:
            reasons.append(f"{key}: missing aggregate rate")
        elif key[1] == "solo" and cand < base * (1 - tolerance):
            reasons.append(f"{key}: solo regression {base} -> {cand}")
        elif key[1] != "solo" and cand <= base:
            reasons.append(
                f"{key}: concurrent aggregate did not improve {base} -> {cand}"
            )
        names = set(_per_row(next(iter(arms["base"].values()))["concurrent"]))
        for name in names:
            rates = {}
            for arm in ("base", "cand"):
                values = [
                    _per_row(r["concurrent"]).get(name, {}).get("tps")
                    for r in arms[arm].values()
                ]
                rates[arm] = (
                    statistics.median(values)
                    if all(v is not None and v > 0 for v in values)
                    else None
                )
            if rates["base"] is None or rates["cand"] is None:
                reasons.append(f"{key}/{name}: missing per-row rate")
            elif rates["cand"] < rates["base"] * (1 - tolerance):
                reasons.append(f"{key}/{name}: per-row regression")
    return reasons, numbers


def _per_row(section):
    """Row records only: the queue stamps device strings next to them."""
    per_row = (section or {}).get("per_row", {})
    return {k: v for k, v in per_row.items() if isinstance(v, dict)}


def validate_records(
    rows,
    arms=("base", "cand"),
    scenarios=("solo", "burst2", "burst4", "stagger"),
    shapes=("greedy", "seeded"),
):
    if (
        not rows
        or rows[-1].get("complete") is not True
        or rows[-1].get("success") is not True
    ):
        return False, "missing final successful completion record"
    cells = [r for r in rows if "scenario" in r]
    expected = {(a, s, h) for a in arms for s in scenarios for h in shapes}
    actual = {(r.get("arm"), r.get("scenario"), r.get("shape")) for r in cells}
    if actual != expected or len(cells) != len(expected):
        return False, "missing or duplicate Phase B cells"
    for r in cells:
        if r["shape"] != "agent" and r.get("arm") != "base":
            solo = _per_row(r.get("solo"))
            concurrent = _per_row(r.get("concurrent"))
            if solo.keys() != concurrent.keys() or any(
                solo[n].get("digest") != concurrent[n].get("digest") for n in solo
            ):
                return False, "concurrent token/logprob digest differs from solo"
        for field in ("solo", "concurrent"):
            per_row = _per_row(r.get(field))
            if len(per_row) != len(plans(r["scenario"], 128, 12)):
                return False, "missing request rows"
            if any(not v.get("digest") for v in per_row.values()):
                return False, "missing token/logprob digest"
    return True, ""


def main():
    a = parser().parse_args()
    validate(a)
    if a.dry_run:
        print(json.dumps(dict(complete=True, dry_run=True)))
        return
    try:
        from . import tfbench as t
        from .sampledspec_conc import arm_order
    except ImportError:
        import tfbench as t
        from sampledspec_conc import arm_order
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.OUT = a.out.parent / (a.out.stem + "-servers")
    reference = {}
    with a.out.open("x") as out:

        def write(row):
            line = json.dumps(row)
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)

        for arm in [x for x in arm_order(a.rep) if x in a.arms]:
            t.YUNSHU_SRC = str(a.base_src if arm == "base" else a.cand_src)
            home = t.OUT / "apc" / arm
            env = dict(
                YUNSHU_VLM_APC_DISK="0",
                YUNSHU_VLM_APC_MEMORY_GB="0",
                YUNSHU_VLM_APC_DISK_DIR=str(home),
                YUNSHU_VLM_DRAFT="off" if arm == "off" else (a.draft or t.D),
                YUNSHU_NEXT_TURN_PREFILL="off",
                YUNSHU_CONSOLE="0",
            )
            if arm == "cand":
                env.update(kv.split("=", 1) for kv in a.cand_env)
            server = t.Srv("yunshu", env, f"phaseb-{a.out.stem}-{arm}", model=a.model)
            try:
                for scenario in a.scenarios:
                    for shape in a.shapes:
                        result = run_cell(
                            server, a, scenario, shape, strict=arm != "base"
                        )
                        if shape != "agent":
                            key = (scenario, shape)
                            # Informational: arms run different kernels, so a
                            # cross-arm digest difference is drift, not a bug;
                            # the gate is each arm's rows == its own solo run.
                            alone = {
                                n: r["digest"]
                                for n, r in _per_row(result["solo"]).items()
                            }
                            seen = [alone]
                            if arm != "base":
                                seen.append(
                                    {
                                        n: r["digest"]
                                        for n, r in _per_row(
                                            result["concurrent"]
                                        ).items()
                                    }
                                )
                            result["cross_arm"] = all(
                                reference.setdefault(key, digests) == digests
                                for digests in seen
                            )
                        result.update(arm=arm, rep=a.rep, ctx_tokens=a.ctx_tokens)
                        write(result)
                if arm != "off":
                    server.verify_spec_mode()
                log = server.log.read_text(errors="replace")
                # the packed-DFlash evidence applies to a DFlash drafter only (not mtp, not off:
                # a family without a drafter, e.g. Gemma-4 on the generic layer)
                if arm == "cand" and a.draft not in ("mtp", "off"):
                    if "Round driver DFlash packed" not in log:
                        raise RuntimeError("candidate did not engage packed DFlash")
                if arm == "cand" and "YUNSHU_KV_PRECISION=int8" in a.cand_env:
                    if "kv=int8" not in log:
                        raise RuntimeError("candidate did not engage int8 KV")
            finally:
                server.kill()
                shutil.rmtree(home, ignore_errors=True)
                shutil.rmtree(server.home, ignore_errors=True)
        write(dict(complete=True, success=True, mode=a.mode, device="m5"))


if __name__ == "__main__":
    main()
