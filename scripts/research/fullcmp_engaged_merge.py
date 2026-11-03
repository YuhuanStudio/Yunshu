"""Merge the research evidence files into scripts/research/fullcmp_engaged_evidence.json.

Usage: python3 scripts/research/fullcmp_engaged_merge.py <dir with engaged_evidence_*.json> [out]

Keys are the manifest engine ids (fullcmp_manifest.ENGINES minus ollama, lmstudio and the
placeholder). Decisions (fail closed, each printed in the report):
- the expected mode per engine is the one its launch spec or the audit pins configure (ENGINE_MODE);
  an engine with no configured mode gets empty "modes" and can never be verified;
- evidence that is not evaluable (prose, placeholders, dotted-path kinds with a regex, a counter
  that does not prove drafting) is dropped, or moved to "fallback" when it names a failure;
- oMLX's MTP evidence is dropped from its DFlash mode.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fullcmp_engaged as fe  # noqa: E402

SOURCE_FILES = (
    "engaged_evidence_tf_splash.json",
    "engaged_evidence_vlm_llama.json",
    "engaged_evidence_others.json",
)
# research key -> manifest engine id
SOURCE_KEY = {
    "mlx-lm-server-0.32.2": "mlx-lm-server",
    "llama-cpp": "llama-cpp-b12061",
}
# manifest engine id -> configured mode (None: no configured mode, unverifiable)
ENGINE_MODE = {
    "tf-1.2.1": "dflash",
    "tf-1.1.8": "dflash",
    "splash-1.5.0": "dflash",
    "mlx-vlm-server": "dflash",
    "omlx-0.7.3": "dflash",
    "omlx-0.7.3rc1": "dflash",
    "kapitsa-0.2.26": "dflash",  # pins: DFlash2 drafter (--dflash auto); no mode flag in the spec
    "mlx-serve-26.11.1": "mtp",  # default on Metal when the checkpoint ships an MTP head
    "mlx-lm-server": "ar",  # stock mlx-lm: no draft model in argv
    "llama-cpp-b12061": None,  # launch spec sets no --spec-type
}
# evidence that does not prove drafting (checked per engine, dropped with the reason printed)
DENY = {
    "kapitsa-0.2.26": (
        (re.compile(r"chat: prompt"), "a chat line proves a request, not drafting"),
    ),
}
# engines with no real spec-run log on disk (research: source or binary strings only)
SOURCE_ONLY = {"kapitsa-0.2.26", "mlx-serve-26.11.1", "omlx-0.7.3", "omlx-0.7.3rc1",
               "llama-cpp-b12061", "mlx-lm-server"}
FAILURE_MARK = re.compile(r"could not|falls back|incompatible", re.I)
PLACEHOLDER = re.compile(r"^(none|no |_load\(\)|n/a)", re.I)
MTP_ONLY = re.compile(r"MTP|mtp", re.I)
# Yunshu: its own VLM batch runner line (tfbench.engaged_spec_mode_line, fullcmp_engaged)
YUNSHU_ENTRY = {
    "modes": {
        "dflash": {
            "evidence": [
                {
                    "kind": "log_regex",
                    "pattern_or_path": r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b",
                    "source_ref": "scripts/research/tfbench.py engaged_spec_mode_line (yunshu branch)",
                    "real_example": "VLM batch runner: model=... draft=dflash",
                    "negative_example": "draft=off",
                    "per_request": False,
                    "startup_sufficient": True,
                }
            ],
            "fallback": [],
        }
    },
    "log_source": "yunshu server log (runner line)",
    "notes": "yunshu is checked by fullcmp_engaged.yunshu_runner_evidence (its runner line), not by this table",
    "confidence": "source-only",
}


def _evaluable(ev):
    pat = str(ev.get("pattern_or_path", ""))
    if PLACEHOLDER.match(pat.strip()) or " found" in pat or "none for" in pat:
        return False, "placeholder, not a pattern"
    if ev.get("kind") in ("acceptance_stat", "response_field") and not fe._DOTTED.match(pat):
        return False, "prose or regex in a dotted-path kind"
    if ev.get("kind") in ("log_regex", "metrics"):
        try:
            re.compile(pat)
        except re.error as exc:
            return False, f"bad regex: {exc}"
    return True, ""


def build(sources: dict):
    """sources: {file name: parsed JSON}. Returns (table, report lines)."""
    report, found, meta = [], {}, {}
    for name in SOURCE_FILES:
        for key, value in sources[name].items():
            if key == "_meta":
                meta[name] = value
                continue
            engine = SOURCE_KEY.get(key, key)
            if engine in found:
                report.append(f"DUPLICATE engine {engine} in {name}: kept first")
                continue
            found[engine] = (name, value)
    out = {}
    for engine in sorted(ENGINE_MODE):
        if engine not in found:
            report.append(f"{engine}: no research entry (fail closed, empty modes)")
            out[engine] = {"modes": {}, "log_source": "", "notes": "no research entry",
                           "confidence": "source-only"}
            continue
        name, entry = found[engine]
        mode = ENGINE_MODE[engine]
        modes = {}
        if mode is not None:
            evidence, fallback = [], []
            for ev in entry.get("evidence", []):
                pat = str(ev.get("pattern_or_path"))
                if mode == "dflash" and engine.startswith("omlx") and MTP_ONLY.search(pat):
                    report.append(f"{engine}: MTP evidence dropped from the dflash mode")
                    continue
                deny = next((why for rx, why in DENY.get(engine, ()) if rx.search(pat)), None)
                if deny:
                    report.append(f"{engine}: dropped {pat[:60]!r}: {deny}")
                    continue
                ok, why = _evaluable(ev)
                if not ok:
                    if FAILURE_MARK.search(pat) and ev.get("kind") == "log_regex":
                        fallback.append(pat)
                    report.append(f"{engine}: dropped {ev.get('kind')} {pat[:60]!r}: {why}")
                    continue
                item = {k: ev.get(k, "") for k in fe.REQUIRED}
                item["per_request"] = bool(ev.get("per_request", False))
                item["startup_sufficient"] = bool(
                    ev.get("startup_sufficient", engine == "mlx-lm-server" and ev.get("kind") == "log_regex")
                )
                evidence.append(item)
            if evidence:
                modes[mode] = {"evidence": evidence, "fallback": fallback}
            else:
                report.append(f"{engine}: no evaluable evidence for {mode}; mode entry omitted (fail closed)")
        confidence = entry.get("confidence", "")
        if engine in SOURCE_ONLY:  # no real spec-run log on disk for this engine
            confidence = "source-only: " + str(confidence)[:200]
        out[engine] = {
            "modes": modes,
            "log_source": entry.get("log_source", ""),
            "notes": f"from {name}; configured mode {mode}; " + str(entry.get("notes", ""))[:400],
            "confidence": confidence,
        }
    out["yunshu"] = YUNSHU_ENTRY
    table = {"_meta": {"built_by": "fullcmp_engaged_merge.py", "sources": meta}, "engines": out}
    problems = fe.validate_table(table)
    report.extend(f"SCHEMA: {p}" for p in problems)
    return table, report


def main(argv):
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else fe.DEFAULT_TABLE
    data = {name: json.loads((src / name).read_text()) for name in SOURCE_FILES}
    table, report = build(data)
    print("\n".join(report))
    if any(line.startswith("SCHEMA") for line in report):
        return 1
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
