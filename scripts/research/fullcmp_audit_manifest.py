#!/usr/bin/env python3
"""Expected-cell manifest for fullcmp_audit.py (FULLCMP_SPEC.md, "Hard guards"), built from the run inputs.

Inputs, all read here and nothing copied by hand:
  compsync/fullcmp-1103/manifest.jsonl      planned cells (fullcmp_manifest.py)
  compsync/fullcmp-1103/census.json         installed versions, release dates, sha256 pins, weights
  compsync/fullcmp-1103/yunshu_pin.json     pinned Yunshu tree, venv python, packages
  compsync/fullcmp-1103/rep_thresholds.json A/A-derived thresholds (fullcmp_reps.py)
  verify/runs/ctl-1103-*/verdict.json       control runs: A/A must pass, injected delay and injected flip must fail
  <results root>/<engine>/<cell>-rN.jsonl   optional: per-cell rep decisions from fullcmp_reps.scan()

Output: compsync/fullcmp-1103/audit_manifest.json (fullcmp_audit.py reads it as --manifest).

What is audited: decode-part cells (concurrency 1) and conc-part cells (concurrency > 1: one record
per trial, expanded to its requests by fullcmp_audit.conc_request_rows) whose engine has a harness
mapping (bench_engines.MANIFEST_ENGINES). Everything else is listed, never dropped:
  unsupported      stubbed engines (bench_engines.UNSUPPORTED_ENGINES) and rows the manifest marks unsupported
  not_in_harness   rows tfbench cannot run yet (image, thinking, agent, behaviour, sampling, turns)
  not_in_audit     rows tfbench runs under a part the audit does not read (none today)
  unresolved       inputs the builder could not fill; the audit fails on them rather than guessing

Rep policy per cell (fullcmp_reps.py): rep_decision ok (2 reps) | need_rep3 (3 reps) | unstable (3 reps)
| pending (no results yet; the audit reports it as a violation). Without --results-root every cell is
pending, so the manifest is written before the run and re-written after the reps are in.

Fail closed: a broken control, a duplicate cell, or a control that does not show its expected outcome
stops the build with exit status 2.

CLI:
    fullcmp_audit_manifest.py [--batches B1,B2,B4,B5] [--results-root DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_reps as fr  # noqa: E402
import fullcmp_run as frun  # noqa: E402

RUN_DIR = frun.RUN_DIR
VERIFY_RUNS = Path("/Volumes/P5Plus/yunshu-build/verify/runs")
DEFAULT_BATCHES = ("B1", "B2", "B4", "B5")
RUN_ID = frun.RUN_ID

# control runs (verify/runs) that the audit cites as evidence
CONTROL_AA = "ctl-1103-aa-2rep-b70535990697"
CONTROL_SLOW = "ctl-1103-slow2-5d194dcd55bc"
CONTROL_FLIP = "ctl-1103-flip3-7cf5266117ae"
TOK_S_METRICS = ("decode_tps", "followup_decode_tps")

# manifest phase names -> audit phase names (FULLCMP_SCHEMA request.phase)
PHASE_NAMES = {"cold": "cold", "warm": "warm", "fu": "turn2"}

# Per-engine probe for the expected speculative mode of the engines whose expected mode is not yet
# established (expected "unverified" fails closed, or a mode that needs its log line confirmed). Each
# candidate pattern was taken from the engine's own source or binary strings (2026-11-03); a pattern
# is evidence only once a smoke run shows it in that engine's log. "confidence": high = a log line
# printed only when the mode is wired for the request; medium = printed per round or per run, so it
# also needs a count; low = a loaded-state line that is NOT engagement by itself.
EXPECTED_MODE_PROBE = {
    "mlx-serve-26.11.1": {
        "expected_mode": "unverified",
        "probe": (
            "--mtp is the default on Metal (mlx-serve --help: 'The default on Metal and CUDA'). Engaged = "
            "a [mtp-round] log line with drafts>0 and accepted>0 in the cell log, AND /v1/models meta.mtp_loaded "
            "true for the served model. A sidecar-load failure line ('[mtp] failed to load sidecar' or "
            "'sidecar incompatible with target') is a refusal, not engagement."
        ),
        "log_patterns": [
            {
                "regex": r"\[mtp-round\].*drafts=([1-9]\d*).*accepted=([1-9]\d*)",
                "meaning": "a speculative MTP round drafted and accepted tokens",
                "confidence": "medium",
                "source": "mlx-serve binary strings: '[mtp-round] off0= t1= m= drafts= accepted='",
            },
            {
                "regex": r"\[mtp\] failed to load sidecar|sidecar incompatible with target",
                "meaning": "MTP head refused: the cell is not measured in MTP mode",
                "confidence": "high",
                "source": "mlx-serve binary strings",
            },
            {
                "http": "GET /v1/models -> data[].meta.mtp_loaded == true",
                "meaning": "head loaded for the served model (loaded, not yet engaged)",
                "confidence": "low",
                "source": "mlx-serve binary strings: '\"mtp_loaded\":' in the model meta",
            },
        ],
    },
    "kapitsa-0.2.26": {
        "expected_mode": "unverified",
        "probe": (
            "DFlash2 is the default drafter ('--dflash auto ... takes the one downloaded with the model'). "
            "Engaged = the DFlash2 drafter line at load AND a run summary with drafted>0 "
            "('spec: X accepted/drafted over N cycles' with drafted>0) after the cell."
        ),
        "log_patterns": [
            {
                "regex": r"DFlash2 drafter .*loaded in [0-9.]+ ms",
                "meaning": "DFlash2 drafter loaded (loaded, not yet engaged)",
                "confidence": "low",
                "source": "kapitsa binary strings: 'DFlash2 drafter %s: %d layers ... loaded in %.0f ms'",
            },
            {
                "regex": r"spec: [0-9.]+ accepted/drafted over [1-9]\d* cycles",
                "meaning": "speculative cycles ran after the cell (count drafted > 0 separately)",
                "confidence": "medium",
                "source": "kapitsa binary strings: 'spec: %.2f accepted/drafted over %ld cycles'",
            },
            {
                "regex": r"MTP head loaded in [0-9.]+ ms",
                "meaning": "MTP head loaded (not the DFlash2 mode; a mismatch with expected dflash fails)",
                "confidence": "low",
                "source": "kapitsa binary strings",
            },
        ],
    },
    "omlx-0.7.3": {
        "expected_mode": "dflash",
        "probe": (
            "oMLX attaches the DFlash2 drafter only when model settings set dflash_enabled and a draft model "
            "(omlx_model_settings in bench_engines). Engaged = the info line below; the warning lines are "
            "refusals. The 2026-11-03 smoke showed no engagement line (model loaded as type vlm): not engaged."
        ),
        "log_patterns": [
            {
                "regex": r"DFlash enabled for \S+ as a batched drafter, draft=\S+",
                "meaning": "DFlash drafter attached as a batched drafter",
                "confidence": "high",
                "source": "omlx 0.7.3 engine_pool.py:4189 logger.info",
            },
            {
                "regex": r"DFlash enabled for \S+ but no draft model is set|DFlash is not supported for",
                "meaning": "DFlash refused for this model (fails closed)",
                "confidence": "high",
                "source": "omlx 0.7.3 engine_pool.py:4171-4183 logger.warning (line numbers approximate)",
            },
        ],
    },
    "omlx-0.7.3rc1": {
        "expected_mode": "dflash",
        "probe": "same as omlx-0.7.3 (rc1 wheel, same engine_pool branch; re-check its source before the first run)",
        "log_patterns": [
            {
                "regex": r"DFlash enabled for \S+ as a batched drafter, draft=\S+",
                "meaning": "DFlash drafter attached as a batched drafter",
                "confidence": "medium",
                "source": "expected from the 0.7.3 source; not yet grepped in the rc1 wheel",
            },
        ],
    },
}

# manifest engine id -> census weights entry (census._meta.models); None = no pinned weights in census
WEIGHT_GROUP = {
    "yunshu": "qwen3.8-27b-4bit-jundot",
    "omlx-0.7.3": "qwen3.8-27b-4bit-jundot",
    "omlx-0.7.3rc1": "qwen3.8-27b-4bit-jundot",
    "mlx-lm-server": "qwen3.8-27b-4bit-jundot",
    "kapitsa-0.2.26": "qwen3.8-27b-4bit-kapitsa",
    "mlx-vlm-server": "qwen3.8-27b-4bit-jundot",
    "llama-cpp-b12061": "qwen3.8-27b-gguf-unsloth",
    # TensorFold and Splash 1.5.0 load TensorFold/Qwen3.8-27B-MLX-4bit (census key below)
    "tf-1.2.1": "qwen3.8-27b-4bit-tensorfold",
    "tf-1.1.8": "qwen3.8-27b-4bit-tensorfold",
    "splash-1.5.0": "qwen3.8-27b-4bit-tensorfold",
    "mlx-serve-26.11.1": None,
}
# why a None above is unresolved (the audit then has no weight comparison for that engine)
WEIGHT_UNRESOLVED = {
    "mlx-serve-26.11.1": "mlx-serve checkpoint source not recorded in census.json",
}


class ControlError(RuntimeError):
    """A control did not show its expected outcome."""


def _date(iso) -> str | None:
    return str(iso)[:10] if iso and str(iso)[:4].isdigit() else None


def _pin(version, date, sha, kind) -> dict:
    return {
        "version": version,
        "release_date": date,
        "binary_sha256": sha,
        "sha_kind": kind,
    }


def build_pins(census: dict, pin: dict) -> dict:
    """Per-engine pins from census.json (version, release date, sha256); yunshu from yunshu_pin.json."""
    tf, sp, om = census["tensorfold"], census["splash"], census["omlx"]
    ms, ml, kp = census["mlx-serve"], census["mlx_lm_server"], census["kapitsa"]
    return {
        "yunshu": {
            "version": pin["version"],
            "release_date": _date(census["yunshu"]["latest_date"]),
            "git_sha": pin["sha"],
        },
        "tf-1.2.1": _pin(
            "1.2.1",
            _date(tf["latest_date"]),
            tf["sha256"]["1.2.1 tarball"],
            "release archive",
        ),
        # census has no release date for 1.1.8: stays None and is reported unresolved
        "tf-1.1.8": _pin(
            "1.1.8", None, tf["sha256"]["1.1.8 tarball"], "release archive"
        ),
        "splash-1.5.0": _pin(
            "1.5.0",
            _date(sp["latest_date"]),
            sp["sha256"]["1.5.0 tarball"],
            "release archive",
        ),
        "omlx-0.7.3": _pin(
            "0.7.3",
            _date(om["latest_date"]),
            om["sha256"]["0.7.3 wheel cp313"],
            "wheel",
        ),
        "omlx-0.7.3rc1": _pin(
            "0.7.3rc1",
            _date(om["latest_rc_date"]),
            om["sha256"]["0.7.3rc1 wheel cp313"],
            "wheel",
        ),
        "mlx-serve-26.11.1": _pin(
            "26.11.1",
            _date(ms["latest_date"]),
            ms["sha256"]["26.11.1 binary mlx-serve"],
            "binary",
        ),
        "mlx-lm-server": _pin(
            "0.32.2",
            _date(ml["latest_date"]),
            ml["sha256"]["mlx_lm-0.32.2-py3-none-any.whl (PyPI digest)"],
            "wheel",
        ),
        "kapitsa-0.2.26": _pin(
            "0.2.26",
            _date(kp["latest_date"]),
            kp["sha256"]["0.2.26 binary kapitsa"],
            "binary",
        ),
    }


def census_weights(census: dict) -> dict:
    """{census weights key: {file: sha256}} for the weight groups the cells use."""
    models = census["_meta"]["models"]
    out = {}
    for key in sorted(set(v for v in WEIGHT_GROUP.values() if v)):
        out[key] = dict(models[key]["sha256"]) if key in models else None
    return out


def _verdict(root: Path) -> dict:
    path = root / "verdict.json"
    return json.loads(path.read_text())


def _stage(verdict: dict, name: str) -> dict:
    return next(s for s in verdict["stages"] if s["name"] == name)


def load_controls(
    verify_runs: Path = VERIFY_RUNS, thresholds_path: Path | None = None
) -> tuple[dict, dict]:
    """(controls for the audit manifest, evidence per control run). Raises ControlError when a
    control does not show its expected outcome."""
    import hashlib

    evidence = {}
    for d in sorted(verify_runs.glob("ctl-1103-*")):
        if (d / "verdict.json").is_file():
            v = _verdict(d)
            evidence[d.name] = {
                "run_dir": str(d),
                "verdict_sha256": hashlib.sha256(
                    (d / "verdict.json").read_bytes()
                ).hexdigest(),
                "overall": v.get("overall"),
                "failed_stage": v.get("failed_stage"),
            }

    aa_dir = verify_runs / CONTROL_AA
    aa = _verdict(aa_dir)
    if aa.get("overall") != "PASS":
        raise ControlError(f"A/A control {CONTROL_AA} is {aa.get('overall')}, not PASS")
    cells = [
        c
        for c in _stage(aa, "speed")["numbers"]["cells"]
        if c["metric"] in TOK_S_METRICS
    ]
    if not cells:
        raise ControlError("A/A control has no tok/s cells in its speed stage")
    aa_delta = max(abs(float(c["delta_pct"])) for c in cells)
    thresholds = json.loads((thresholds_path or frun.THRESH_PATH).read_text())[
        "thresholds_pct"
    ]
    aa_noise = float(thresholds["decode_tps"])
    if aa_delta > aa_noise:
        raise ControlError(
            f"A/A tok/s delta {aa_delta} % is outside the noise band {aa_noise} %"
        )

    slow = _verdict(verify_runs / CONTROL_SLOW)
    slow_reasons = " ".join(_stage(slow, "speed").get("reasons", []))
    delay_ok = (
        slow.get("overall") == "FAIL"
        and slow.get("failed_stage") == "speed"
        and "decode_tps" in slow_reasons
    )
    if not delay_ok:
        raise ControlError(
            f"injected-delay control {CONTROL_SLOW} was not detected by the speed stage"
        )

    flip = _verdict(verify_runs / CONTROL_FLIP)
    flip_reasons = " ".join(_stage(flip, "identity").get("reasons", []))
    flip_ok = (
        flip.get("overall") == "FAIL"
        and flip.get("failed_stage") == "identity"
        and "digest differs" in flip_reasons
    )
    if not flip_ok:
        raise ControlError(
            f"injected-flip control {CONTROL_FLIP} was not detected by the identity stage"
        )

    controls = {
        "aa_delta_pct": aa_delta,
        "aa_noise_pct": aa_noise,
        "injected_delay_detected": True,
        "injected_flip_detected": True,
        "evidence": {
            "aa": CONTROL_AA,
            "slow": CONTROL_SLOW,
            "flip": CONTROL_FLIP,
            "runs": evidence,
        },
    }
    return controls, evidence


DECISION_OF_STATUS = {
    "ok": ("ok", False),
    "need_rep3": ("need_rep3", True),
    "spread_flag": ("need_rep3", True),  # three reps are present: rep 3 was required
    "unstable": ("unstable", True),
}


def rep_decisions(results_root, thresholds: dict) -> dict:
    """{(engine, cell): (rep_decision, need_rep3)} from fullcmp_reps.scan over raw results.
    Cells with fewer than two reps on disk are absent (the caller reports them as pending)."""
    out = {}
    for engine, cell, _group, record in fr.scan(results_root, thresholds, None):
        out[(engine, cell)] = DECISION_OF_STATUS[record["status"]]
    return out


def build_manifest(
    rows: list,
    census: dict,
    pin: dict,
    thresholds: dict,
    controls: dict,
    decisions: dict,
    batches=DEFAULT_BATCHES,
) -> dict:
    """The audit manifest. Pure: all inputs passed in."""
    pins = build_pins(census, pin)
    unresolved = []
    unsupported, not_in_harness, not_in_audit = [], [], []
    cells, groups, seen = [], {}, set()

    for r in rows:
        if r.get("batch") not in batches:
            continue
        eng, cid = r["engine"], r["id"]
        if (eng, cid) in seen:
            raise SystemExit(f"duplicate cell in manifest: ({eng}, {cid})")
        seen.add((eng, cid))
        entry = {"engine": eng, "cell": cid, "batch": r["batch"]}
        if eng in be.UNSUPPORTED_ENGINES:
            unsupported.append({**entry, "reason": be.stub_reason(eng)})
            continue
        if r.get("status") != "unverified":
            unsupported.append(
                {**entry, "reason": f"status {r.get('status')}: {r.get('reason')}"}
            )
            continue
        if eng not in be.MANIFEST_ENGINES:
            unresolved.append(
                {
                    **entry,
                    "reason": "engine id has no harness mapping (bench_engines.MANIFEST_ENGINES)",
                }
            )
            continue
        part, _extra, why = frun.harness_part(r)
        if part is None:
            not_in_harness.append({**entry, "reason": why})
            continue
        if part not in ("decode", "conc"):
            not_in_audit.append(
                {**entry, "reason": f"{part} part is not in the audit record schema"}
            )
            continue
        if part == "conc":
            # one conc record per trial with n requests each; the planned phase is "conc"
            phases = ["conc"]
            expected = int(r["conc"]) * frun.CONC_TRIALS
        else:
            phases = []
            for p in r["phases"]:
                if p not in PHASE_NAMES:
                    raise SystemExit(f"unknown phase {p!r} in {cid}")
                phases.append(PHASE_NAMES[p])
            expected = len(phases)
        hid = be.tfbench_engine(eng)
        decision, need3 = decisions.get((eng, cid), ("pending", False))
        grp = frun.comparison_group(r)
        cells.append(
            {
                "engine": eng,
                "model": r["model"],
                "column": f"{r['ctx']}/{r['prompt_kind']}/c{r['conc']}/o{r['out_len']}",
                "cell": cid,
                "group": grp,
                "ctx": r["ctx_tokens"],
                "kind": r["prompt_kind"],
                "prompt_id": f"{r['prompt_kind']}-{r['ctx_tokens']}",
                "max_tokens": r["out_len"],
                "phases": phases,
                "expected_records": expected,
                "spec_mode": be.ENGINES[hid].expected_mode,
                "weight_group": WEIGHT_GROUP.get(eng),
                "reps": 2,
                "rep_decision": decision,
                "need_rep3": need3,
            }
        )
        order = groups.setdefault(grp, [])
        if eng not in order:
            order.append(eng)

    for eng, key in sorted(WEIGHT_GROUP.items()):
        if key is None:
            unresolved.append({"engine": eng, "reason": WEIGHT_UNRESOLVED[eng]})
    for eng in sorted({c["engine"] for c in cells}):
        if eng not in pins:
            unresolved.append({"engine": eng, "reason": "no census pin"})
        elif pins[eng].get("release_date") is None:
            unresolved.append(
                {"engine": eng, "reason": "release date not in census.json"}
            )
    if not cells:
        unresolved.append(
            {"engine": "*", "reason": "no audited cells for the requested batches"}
        )

    return {
        "run_id": RUN_ID,
        "reps": 2,
        "pins": pins,
        "yunshu": {"git_sha": pin["sha"], "python": pin["python_executable"]},
        "controls": controls,
        "groups": {g: {"order": o} for g, o in groups.items()},
        "expected_mode_probe": EXPECTED_MODE_PROBE,
        "cells": cells,
        "unsupported": unsupported,
        "not_in_harness": not_in_harness,
        "not_in_audit": not_in_audit,
        "unresolved": unresolved,
        "weights": census_weights(census),
        "thresholds_pct": thresholds,
        "engine_map": {k: be.tfbench_engine(k) for k in sorted(be.MANIFEST_ENGINES)},
        "todo": list(be.TODO_LIST),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batches", default=",".join(DEFAULT_BATCHES))
    ap.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="raw results root for the rep decisions",
    )
    ap.add_argument("--out", type=Path, default=RUN_DIR / "audit_manifest.json")
    a = ap.parse_args(argv)

    rows = frun.load_rows(RUN_DIR / "manifest.jsonl")
    census = json.loads((RUN_DIR / "census.json").read_text())
    pin = json.loads(frun.PIN_PATH.read_text())
    thresholds = json.loads(frun.THRESH_PATH.read_text())["thresholds_pct"]
    try:
        controls, _ = load_controls()
    except (
        ControlError,
        OSError,
        KeyError,
        StopIteration,
        json.JSONDecodeError,
    ) as exc:
        print(f"CONTROLS REFUSED: {exc}")
        return 2
    decisions = {}
    if a.results_root is not None:
        decisions = rep_decisions(
            a.results_root, fr.thresholds_from_file(frun.THRESH_PATH)
        )
    manifest = build_manifest(
        rows, census, pin, thresholds, controls, decisions, tuple(a.batches.split(","))
    )
    a.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    pending = sum(1 for c in manifest["cells"] if c["rep_decision"] == "pending")
    print(
        f"wrote {a.out}: {len(manifest['cells'])} audited cells ({pending} pending rep decision), "
        f"{len(manifest['unsupported'])} unsupported, {len(manifest['not_in_harness'])} not in harness, "
        f"{len(manifest['not_in_audit'])} not in audit, {len(manifest['unresolved'])} unresolved"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
