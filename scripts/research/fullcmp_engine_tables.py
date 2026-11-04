#!/usr/bin/env python3
"""Per-engine table registry for the full comparison.

Every table the harness looks an engine up in is listed in TABLES with the accessor the harness
calls at run time and the key form it expects:

  manifest id  fullcmp_manifest.ENGINES key, e.g. "yunshu", "tf-1.2.1", "mlx-serve-26.11.1"
  bench id     bench_engines.ENGINES key, e.g. "yunshu-main" (the pinned Yunshu row), "tf-1.2.1"
               (reached from a manifest id through bench_engines.tfbench_engine)
  census key   census.json entry, e.g. "tensorfold", "mlx-serve", "kapitsa" (tfbench.FULLCMP_CENSUS_KEY)

check_all(engine_ids) resolves EVERY table for EVERY engine through the same accessor and returns
ALL misses at once. A miss is a lookup the harness would fail on, or a value that is missing or
'unverified' where the audit needs a real one. An open decision with a recorded reason (for example
a weights group the census cannot resolve) is a note: printed by run_checks, never a failure.
The runner (fullcmp_run.main) refuses before any job is written when check_all returns misses.

Tables the harness reads per engine but that have no run-time failure mode are listed in
UNCHECKED with the reason; they are not resolved here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bench_engines as be  # noqa: E402
import fullcmp_attribution as fa  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402
import fullcmp_engaged as fe  # noqa: E402
import fullcmp_engaged_merge as fem  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
import tfbench as tb  # noqa: E402

# Engines that never run: bench_engines.UNSUPPORTED_ENGINES (ollama, lmstudio) and the placeholder row.
EXCLUDED = frozenset(be.UNSUPPORTED_ENGINES) | {fm.PLACEHOLDER}
CHECKED_IDS = [e for e in fm.ENGINES if e not in EXCLUDED]
EVIDENCE_NAME = "fullcmp_engaged_evidence.json"


@dataclass(frozen=True)
class Table:
    name: str
    source: str  # module:symbol the harness reads
    key: str  # key form the accessor expects


@dataclass(frozen=True)
class Miss:
    engine: str
    table: str
    key: str  # the key the accessor tried
    error: str

    def line(self) -> str:
        return f"{self.engine}: [{self.table}] key {self.key!r}: {self.error}"


@dataclass(frozen=True)
class Note:
    engine: str
    table: str
    text: str


class TableMiss(Exception):
    def __init__(self, key, message: str):
        super().__init__(message)
        self.key = key


class Unresolved(Exception):
    """A documented open item: reported as a note, never a miss."""


class Ctx:
    """Inputs the accessors read: census.json, yunshu_pin.json, the engaged evidence table."""

    def __init__(self, census=None, pin=None, evidence=None, load_errors=()):
        self.census = census
        self.pin = pin
        self.evidence = evidence
        self.load_errors = list(load_errors)
        self.pins, self.pins_error = None, None
        if census is not None and pin is not None:
            try:
                self.pins = fam.build_pins(census, pin)
            except Exception as exc:  # noqa: BLE001 - reported as a miss for every engine
                self.pins_error = f"{type(exc).__name__}: {exc}"


def load_context(census_path=None, pin_path=None, evidence_path=None) -> Ctx:
    import fullcmp_run as frun  # lazy: fullcmp_run imports this module at run time

    census_path = Path(census_path or frun.RUN_DIR / "census.json")
    pin_path = Path(pin_path or frun.PIN_PATH)
    errors = []
    census = pin = None
    try:
        census = json.loads(census_path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(("census.json", f"{census_path}: {exc}"))
    try:
        pin = json.loads(pin_path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(("yunshu_pin.json", f"{pin_path}: {exc}"))
    evidence = fe.load_table(evidence_path or fe.DEFAULT_TABLE)
    if evidence is None:
        errors.append((EVIDENCE_NAME, "missing or malformed engaged evidence table"))
    return Ctx(census, pin, evidence, errors)


# ---------------------------------------------------------------------------
# Accessors. Each takes (manifest id, ctx) and returns the value the harness uses, or raises.
# ---------------------------------------------------------------------------


def _hid(engine: str) -> str:
    try:
        return be.tfbench_engine(engine)
    except KeyError as exc:
        raise TableMiss(engine, f"bench_engines.MANIFEST_ENGINES has no harness id: {exc}") from exc


def _is_yunshu(hid: str) -> bool:
    return tb.Srv._is_yunshu(hid)


def a_manifest(e, c):
    row = fm.ENGINES[e]
    for field in ("label", "version"):
        if not row.get(field):
            raise TableMiss(e, f"ENGINES[{e!r}] has no {field!r}")
    return row


def a_harness_id(e, c):
    return _hid(e)


def a_bench_engine(e, c):
    hid = _hid(e)
    if hid not in be.ENGINES:
        raise TableMiss(hid, "no bench_engines.ENGINES entry (expected_mode, kind)")
    return be.ENGINES[hid]


def a_launch_recipe(e, c):
    hid = _hid(e)
    return be.build_launch(hid, 1, "/nonexistent-fullcmp-home", {})  # pure: no filesystem


def a_required_paths(e, c):
    return be.required_paths(_hid(e))


def a_static_version(e, c):
    hid = _hid(e)
    if _is_yunshu(hid):
        return ("pin", None)  # the Yunshu version comes from yunshu_pin.json
    if hid not in be.STATIC_VERSIONS:
        raise TableMiss(hid, "bench_engines.STATIC_VERSIONS has no entry (version reported as unknown)")
    version = be.STATIC_VERSIONS[hid][0]
    if not version or version == "unknown":
        raise TableMiss(hid, f"STATIC_VERSIONS version is {version!r}")
    return be.STATIC_VERSIONS[hid]


def _version(e, c) -> str:
    hid = _hid(e)
    if _is_yunshu(hid):
        return str(c.pin["version"])
    return a_static_version(e, c)[0]


def a_census_key(e, c):
    hid = _hid(e)
    if _is_yunshu(hid):
        return "yunshu"
    if hid not in tb.FULLCMP_CENSUS_KEY:
        raise TableMiss(hid, "tfbench.FULLCMP_CENSUS_KEY has no entry (no release date)")
    return tb.FULLCMP_CENSUS_KEY[hid]


def a_census_entry(e, c):
    key = a_census_key(e, c)
    if c.census is None or not isinstance(c.census.get(key), dict):
        raise TableMiss(key, "census.json has no entry for this key")
    return c.census[key]


def a_release_date(e, c):
    key = a_census_key(e, c)
    a_census_entry(e, c)
    # the runtime path: fullcmp_provenance.release_date with the runner's environment (no override)
    return fp.release_date(c.census, key, _version(e, c), {})


def a_audit_pin(e, c):
    if c.pins is None:
        raise TableMiss("census.json", f"build_pins failed: {c.pins_error or 'no census or pin'}")
    if e not in c.pins:
        raise TableMiss(e, "fullcmp_audit_manifest.build_pins has no entry")
    pin = c.pins[e]
    missing = [k for k in ("version", "release_date", "binary_sha256", "install_path") if not pin.get(k)]
    if missing:
        raise TableMiss(e, f"pin lacks {missing} (the launched file is not on disk or not recorded)")
    return pin


def a_audit_pin_version(e, c):
    pin = a_audit_pin(e, c)
    if _is_yunshu(_hid(e)):
        return pin["version"]
    want = a_static_version(e, c)[0]
    if pin["version"] != want:
        raise TableMiss(e, f"audit pin version {pin['version']!r} != runtime version {want!r} (header would fail)")
    return pin["version"]


def a_weights_group(e, c):
    if e not in fam.WEIGHT_GROUP:
        raise TableMiss(e, "fullcmp_audit_manifest.WEIGHT_GROUP has no entry")
    group = fam.WEIGHT_GROUP[e]
    if group is None:
        reason = fam.WEIGHT_UNRESOLVED.get(e)
        if not reason:
            raise TableMiss(e, "WEIGHT_GROUP is None with no WEIGHT_UNRESOLVED reason")
        raise Unresolved(f"weights unresolved: {reason}")
    if not fam.census_weights(c.census).get(group):
        raise TableMiss(group, "census _meta.models has no sha256 for this weights group")
    return group


def a_expected_mode(e, c):
    if fe.is_yunshu(e):
        mode = fam.YUNSHU_EXPECTED_MODE
    elif e not in fam.EXPECTED_MODE_PROBE:
        raise TableMiss(e, "fullcmp_audit_manifest.EXPECTED_MODE_PROBE has no entry (the audit defaults to 'unverified')")
    else:
        mode = fam.EXPECTED_MODE_PROBE[e]["expected_mode"]
    if mode == "unverified":
        raise TableMiss(e, "expected spec mode is 'unverified': the engaged-mode guard fails every cell")
    return mode


def a_expected_mode_matches_bench(e, c):
    mode = a_expected_mode(e, c)
    bench_mode = be.ENGINES[_hid(e)].expected_mode
    if mode != bench_mode:
        raise TableMiss(_hid(e), f"audit probe mode {mode!r} != bench_engines expected_mode {bench_mode!r}")
    return mode


def a_engaged_regex(e, c):
    mode = a_expected_mode(e, c)
    if _is_yunshu(_hid(e)):
        return fam.YUNSHU_ENGAGED_REGEX
    # confidence is metadata: the header regex is the first non-refusal log pattern of any confidence
    if e not in fam.EXPECTED_MODE_PROBE:
        raise TableMiss(e, "no EXPECTED_MODE_PROBE entry")
    patterns = [
        p["regex"]
        for p in fam.EXPECTED_MODE_PROBE[e].get("log_patterns", [])
        if "regex" in p and "refus" not in str(p.get("meaning", "")) + str(p.get("note", "")).lower()
    ]
    if not patterns:
        raise TableMiss(e, f"no engaged log pattern for mode {mode!r} (EXPECTED_MODE_PROBE)")
    return patterns[0]


def a_engaged_evidence(e, c):
    mode = a_expected_mode(e, c)
    if c.evidence is None:
        raise TableMiss(EVIDENCE_NAME, "missing or malformed engaged evidence table")
    entry = fe.mode_entry(c.evidence, e, mode)
    if entry is None:
        raise TableMiss(f"{e}/{mode}", "no engaged evidence entry for the expected mode")
    if _is_yunshu(_hid(e)):  # Yunshu is judged by fullcmp_engaged.yunshu_runner_evidence (its runner line)
        return entry
    # the guard evaluates per measured row: a per-request item, or a startup-sufficient item (stock AR)
    if not any(ev.get("per_request") or ev.get("startup_sufficient") for ev in entry.get("evidence", [])):
        raise TableMiss(f"{e}/{mode}", "no per-request or startup-sufficient evidence item")
    return entry


def a_merge_mode(e, c):
    expected = a_expected_mode(e, c)
    if fe.is_yunshu(e):  # the merge writes Yunshu's entry explicitly (fem.YUNSHU_ENTRY), not from ENGINE_MODE
        if expected not in fem.YUNSHU_ENTRY["modes"]:
            raise TableMiss(e, f"fullcmp_engaged_merge.YUNSHU_ENTRY has no {expected!r} mode")
        return expected
    if e not in fem.ENGINE_MODE:
        raise TableMiss(e, "fullcmp_engaged_merge.ENGINE_MODE has no entry (a rebuild drops this engine's evidence)")
    if fem.ENGINE_MODE[e] != expected:
        raise TableMiss(e, f"fullcmp_engaged_merge.ENGINE_MODE {fem.ENGINE_MODE[e]!r} != probe {expected!r}")
    return fem.ENGINE_MODE[e]


def a_attribution(e, c):
    return fa.engine_attr(e)


def a_attribution_layers(e, c):
    fa.required_layer1(e)
    return fa.required_layer2(e)


def a_attribution_from_harness(e, c):
    """tfbench.attribution_fields looks the schema up by bench_engines.manifest_id(harness id)."""
    hid = _hid(e)
    got = be.manifest_id(hid)
    if got != e:
        raise TableMiss(hid, f"manifest_id({hid!r}) is {got!r}: the attribution lookup would use the wrong schema")
    return fa.engine_attr(got)


def a_perf_curve(e, c):
    return (fm.prefill_tps(e, 1024), fm.decode_tps(e, 1024), fm.slowdown(e, 1024, 1))


def a_disk_cache(e, c):
    caps = fm.ENGINES[e].get("caps", {})
    if "disk_cache" not in caps:
        raise Unresolved("caps.disk_cache not declared: restart cells run (silent default)")
    return caps["disk_cache"]


TABLES: list[tuple[Table, object]] = [
    (Table("manifest entry", "fullcmp_manifest.ENGINES", "manifest id"), a_manifest),
    (Table("harness id", "bench_engines.MANIFEST_ENGINES", "manifest id"), a_harness_id),
    (Table("bench engine (expected_mode, kind)", "bench_engines.ENGINES", "bench id"), a_bench_engine),
    (Table("launch recipe", "bench_engines.build_launch", "bench id"), a_launch_recipe),
    (Table("required paths", "bench_engines.required_paths", "bench id"), a_required_paths),
    (Table("static version", "bench_engines.STATIC_VERSIONS", "bench id"), a_static_version),
    (Table("census key (release date)", "tfbench.FULLCMP_CENSUS_KEY", "bench id"), a_census_key),
    (Table("census entry", "census.json[key]", "census key"), a_census_entry),
    (Table("release date (runtime)", "fullcmp_provenance.release_date", "census key + version"), a_release_date),
    (Table("audit pin (version, date, sha, install)", "fullcmp_audit_manifest.build_pins", "manifest id"), a_audit_pin),
    (Table("audit pin version = runtime version", "fullcmp_audit_manifest.build_pins", "manifest id"), a_audit_pin_version),
    (Table("weights group (+ census sha)", "fullcmp_audit_manifest.WEIGHT_GROUP", "manifest id"), a_weights_group),
    (Table("expected spec mode", "fullcmp_audit_manifest.EXPECTED_MODE_PROBE", "manifest id"), a_expected_mode),
    (Table("expected mode = bench expected_mode", "fullcmp_audit_manifest.EXPECTED_MODE_PROBE", "manifest id"), a_expected_mode_matches_bench),
    (Table("engaged regex (header)", "fullcmp_audit_manifest._expected", "manifest id"), a_engaged_regex),
    (Table("engaged evidence (mode entry)", "fullcmp_engaged_evidence.json", "manifest id + mode"), a_engaged_evidence),
    (Table("evidence builder mode", "fullcmp_engaged_merge.ENGINE_MODE", "manifest id"), a_merge_mode),
    (Table("attribution schema (layer 1+2)", "fullcmp_attribution.ENGINE_OVERRIDES", "manifest id"), a_attribution),
    (Table("attribution layers", "fullcmp_attribution.required_layer1/2", "manifest id"), a_attribution_layers),
    (Table("attribution lookup from tfbench", "bench_engines.manifest_id", "bench id"), a_attribution_from_harness),
    (Table("perf curve (planning)", "fullcmp_manifest.ENGINE_PERF", "manifest id"), a_perf_curve),
    (Table("disk-tier flag (restart cells)", "fullcmp_manifest.ENGINES[caps].disk_cache", "manifest id"), a_disk_cache),
]

# Per-engine tables the harness reads that are not resolved here, with the reason.
UNCHECKED = {
    "warm_lazy_engine list": "inline tuple in tfbench.warm_lazy_engine; needs a server to call",
    "port assignment": "free_port() at run time; no table",
    "readiness probe": "ready_request(url) + optional extra['startup_ready_regex'] (absent = no wait)",
    "log path": "server-<tag>.log per run; no table",
    "usage shape": "tests/unit/fullcmp_usage_shapes.json, stub-only; real engines are parsed as sent (default 'details')",
    "profile env": "fullcmp_profile.profile_env: competitors have none by design (default [])",
    "wall prior": "fullcmp_run.prior_wall_min: no history means 0 (timeout falls back to the estimate)",
    "behaviour runners": "fullcmp_behaviour.RUNNERS keyed by check name, not by engine",
}


def run_checks(engine_ids=None, ctx: Ctx | None = None):
    """(misses, notes) for the engines. Excluded ids are skipped; an unknown id is a miss."""
    ids = list(CHECKED_IDS if engine_ids is None else engine_ids)
    ctx = ctx or load_context()
    misses: list[Miss] = []
    notes: list[Note] = []
    for e in ids:
        if e in EXCLUDED:
            continue
        if e not in fm.ENGINES:
            misses.append(Miss(e, "manifest entry", e, "not a manifest engine id"))
            continue
        for path, error in ctx.load_errors:
            misses.append(Miss(e, "inputs", path, error))
        for table, fn in TABLES:
            try:
                fn(e, ctx)
            except Unresolved as exc:
                notes.append(Note(e, table.name, str(exc)))
            except TableMiss as exc:
                misses.append(Miss(e, table.name, str(exc.key), str(exc)))
            except Exception as exc:  # noqa: BLE001 - every failure is a miss, with its key
                misses.append(Miss(e, table.name, e, f"{type(exc).__name__}: {exc}"))
    return misses, notes


def check_all(engine_ids=None, ctx: Ctx | None = None) -> list[Miss]:
    """Every miss for the engines, all at once. Empty means the GPU will not hit a table miss."""
    return run_checks(engine_ids, ctx)[0]


def format_misses(misses) -> str:
    return "\n".join(["table misses (the runner refuses; nothing is submitted):"] + [f"  {m.line()}" for m in misses])


def main(argv=None) -> int:
    misses, notes = run_checks(argv or None)
    for n in notes:
        print(f"note {n.engine} [{n.table}]: {n.text}")
    if misses:
        print(format_misses(misses))
        return 1
    print(f"ok: {len(CHECKED_IDS)} engines, {len(TABLES)} tables, no misses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
