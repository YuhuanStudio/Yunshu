"""Merge the research evidence files into scripts/research/fullcmp_engaged_evidence.json.

Usage: python3 scripts/research/fullcmp_engaged_merge.py <dir with engaged_evidence_*.json> [out]

Decisions (fail closed, each printed in the report):
- the expected mode per engine is the one its launch spec configures (ENGINE_MODE); an engine with
  no configured mode gets empty "modes" and can never be verified;
- evidence that is not evaluable (prose, placeholders, dotted-path kinds with a regex, a fallback
  marker) is dropped, or moved to "fallback" when it names a failure;
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
# expected mode per engine, from the launch spec in bench_engines (no configured mode: None)
ENGINE_MODE = {
    "tf-1.2.1": "dflash",
    "tf-1.1.8": "dflash",
    "splash-1.5.0": "dflash",
    "mlx-vlm-server": "dflash",
    "omlx-0.7.3": "dflash",
    "omlx-0.7.3rc1": "dflash",
    "mlx-lm-server-0.32.2": "ar",  # stock mlx-lm: no draft model in argv
    "llama-cpp": None,  # launch spec sets no --spec-type
    "mlx-serve-26.11.1": None,  # launch spec sets no speculative flag: unverified expected mode
    "kapitsa-0.2.26": None,  # launch spec spec_mode 'unverified'
}
FAILURE_MARK = re.compile(r"could not|falls back|incompatible", re.I)
PLACEHOLDER = re.compile(r"^(none|no |_load\(\)|n/a)", re.I)
MTP_ONLY = re.compile(r"MTP|mtp", re.I)


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
    report, engines = [], {}
    meta = {}
    for name in SOURCE_FILES:
        data = sources[name]
        for key, value in data.items():
            if key == "_meta":
                meta[name] = value
                continue
            if key in engines:
                report.append(f"DUPLICATE engine {key} in {name}: kept first")
                continue
            engines[key] = (name, value)
    out = {}
    for engine, (name, entry) in sorted(engines.items()):
        if engine not in ENGINE_MODE:
            report.append(f"{engine}: not in ENGINE_MODE, dropped")
            continue
        mode = ENGINE_MODE[engine]
        modes = {}
        if mode is not None:
            evidence, fallback = [], []
            for ev in entry.get("evidence", []):
                if mode == "dflash" and engine.startswith("omlx") and MTP_ONLY.search(str(ev.get("pattern_or_path"))):
                    report.append(f"{engine}: MTP evidence dropped from the dflash mode")
                    continue
                ok, why = _evaluable(ev)
                if not ok:
                    if FAILURE_MARK.search(str(ev.get("pattern_or_path"))) and ev.get("kind") == "log_regex":
                        fallback.append(str(ev["pattern_or_path"]))
                    report.append(f"{engine}: dropped {ev.get('kind')} {str(ev.get('pattern_or_path'))[:60]!r}: {why}")
                    continue
                item = {k: ev.get(k, "") for k in fe.REQUIRED}
                item["per_request"] = bool(ev.get("per_request", False))
                item["startup_sufficient"] = bool(
                    ev.get("startup_sufficient", engine == "mlx-lm-server-0.32.2" and ev.get("kind") == "log_regex")
                )
                evidence.append(item)
            if evidence:
                modes[mode] = {"evidence": evidence, "fallback": fallback}
            else:
                report.append(f"{engine}: no evaluable evidence for {mode}; mode entry omitted (fail closed)")
        out[engine] = {
            "modes": modes,
            "log_source": entry.get("log_source", ""),
            "notes": f"from {name}; configured mode {mode}; " + str(entry.get("notes", ""))[:400],
            "confidence": entry.get("confidence", ""),
        }
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
