"""Path-parity guard: no optimization is wired into only some serving paths by accident.

python/yunshu_engine/path_matrix.json lists every optimization / capability and, for
each serving path, whether it is engaged (with a code needle that must be present),
not applicable (with a reason) or missing / partial (with an owner, an impact and,
where useful, a needle that must stay absent). Adding an optimization module or a
performance setting without a row here fails this test; wiring a documented gap makes
the "absent" needle appear and fails it too, so the matrix is updated in the same
commit as the wiring (flip the cell to engaged and add its evidence).

Static inspection only: no MLX import, no GPU.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python"
MATRIX = json.loads((PKG / "yunshu_engine" / "path_matrix.json").read_text())
STATUSES = {"engaged", "na", "missing", "partial"}
IMPACTS = {"high", "medium", "low"}
OWNERS = {
    "decode16",
    "flashnext80",
    "apcomni",
    "memidle",
    "unassigned",
}
# Module-name shapes that carry an optimization (a new one needs a decision).
OPT_PATTERNS = re.compile(
    r"^(draft_vocab|copy_.*|dflash_.*|spec_.*|adaptive_spec|apc_.*|mtp_.*|tree_.*|"
    r".*ngram.*|suffix_.*|cache_(?:decode|prefill|restore)|keyed_sampling|"
    r"constrained_spec|grammar_.*|kv_(?:quantization|optimizations)|turbo_quant|"
    r"n_confirmed_patch|mlxvlm_mtp|tool_.*grammar|tool_call_grammar)$"
)


def _read(rel: str) -> str:
    path = ROOT / rel
    assert path.is_file(), f"{rel} does not exist"
    return path.read_text()


def _modules() -> dict[str, Path]:
    mods = {}
    for p in PKG.rglob("*.py"):
        if "_vendor" in p.parts or "__pycache__" in p.parts:
            continue
        mods[
            ".".join(p.relative_to(PKG).with_suffix("").parts).removesuffix(".__init__")
        ] = p
    return mods


def _importers() -> dict[str, set[str]]:
    mods = _modules()
    users: dict[str, set[str]] = {m: set() for m in mods}
    for name, p in mods.items():
        pkg = name.rsplit(".", 1)[0] if p.name != "__init__.py" else name
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    parts = parts[: len(parts) - (node.level - 1)]
                    base = ".".join(parts + ([base] if base else []))
                targets.append(base)
                targets += [f"{base}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                targets += [a.name for a in node.names]
            for t in targets:
                if t in users and t != name:
                    users[t].add(name)
    return users


def _cells():
    for fid, feat in MATRIX["features"].items():
        for path, cell in feat["cells"].items():
            yield fid, path, cell


def test_every_feature_has_every_path():
    paths = set(MATRIX["paths"])
    for fid, feat in MATRIX["features"].items():
        assert set(feat["cells"]) == paths, (
            f"{fid}: cells must cover exactly {sorted(paths)}"
        )
        assert (ROOT / feat["module"]).exists(), (
            f"{fid}: module {feat['module']} missing"
        )


def test_cells_are_decided():
    for fid, path, cell in _cells():
        where = f"{fid}/{path}"
        assert cell["s"] in STATUSES, where
        if cell["s"] == "engaged":
            assert cell.get("ev"), f"{where}: engaged needs evidence"
        else:
            assert cell.get("why"), f"{where}: {cell['s']} needs a reason"
        if cell["s"] in ("missing", "partial"):
            assert cell.get("owner") in OWNERS, (
                f"{where}: owner must be one of {sorted(OWNERS)}"
            )
            assert cell.get("impact") in IMPACTS, where


def test_engaged_evidence_is_in_the_code():
    for fid, path, cell in _cells():
        for rel, needle in cell.get("ev", []):
            assert needle in _read(rel), (
                f"{fid}/{path}: '{needle}' no longer in {rel}; the wiring moved or was "
                "removed, update path_matrix.json"
            )


def test_documented_gaps_are_still_gaps():
    """Wiring a gap must flip its cell to engaged in the same commit."""
    for fid, path, cell in _cells():
        for rel, needle in cell.get("absent", []):
            assert needle not in _read(rel), (
                f"{fid}/{path}: '{needle}' now appears in {rel}: the gap looks wired. "
                "Flip this cell to engaged in path_matrix.json (and add evidence)."
            )


def test_exclusions_and_duplicates_are_grounded():
    for ex in MATRIX["exclusions"]:
        assert ex["kind"] in ("fundamental", "unimplemented"), ex["id"]
        assert ex["why"], ex["id"]
        for rel, needle in ex["ev"]:
            assert needle in _read(rel), (
                f"exclusion {ex['id']}: '{needle}' gone from {rel}"
            )
    for dup in MATRIX["duplicates"]:
        assert len(dup["impls"]) >= 2 and dup["unify"], dup["id"]
        for rel in dup["impls"]:
            assert (ROOT / rel).exists(), f"duplicate {dup['id']}: {rel} missing"


def test_new_optimization_modules_need_a_decision():
    known = {f["module"] for f in MATRIX["features"].values()}
    known |= set(MATRIX["modules_other"]) | set(MATRIX["orphans"])
    undecided = []
    for name, p in _modules().items():
        rel = str(p.relative_to(ROOT))
        parts = name.split(".")
        if parts[0] != "yunshu_engine" or p.name == "__init__.py":
            continue
        in_kernels = len(parts) > 1 and parts[1] == "kernels"
        in_driver = len(parts) > 1 and parts[1] == "round_driver"
        if not (in_kernels or in_driver or OPT_PATTERNS.match(parts[-1])):
            continue
        if rel in known or any(rel.startswith(k.rstrip("/") + "/") for k in known):
            continue
        undecided.append(rel)
    assert not undecided, (
        "optimization modules with no row in path_matrix.json (add a feature, or an entry "
        f"in modules_other with a reason): {undecided}"
    )


def test_orphan_modules_are_exactly_the_declared_ones():
    users = _importers()
    pattern_modules = {
        str(p.relative_to(ROOT)): name
        for name, p in _modules().items()
        if name.startswith("yunshu_engine.")
        and p.name != "__init__.py"
        and (
            name.split(".")[1] in ("kernels", "round_driver")
            or OPT_PATTERNS.match(name.split(".")[-1])
        )
    }
    orphans = {rel for rel, name in pattern_modules.items() if not users[name]}
    declared = set(MATRIX["orphans"])
    assert orphans == declared, (
        f"optimization modules nobody imports: undeclared {sorted(orphans - declared)}, "
        f"declared but now imported (remove from orphans) {sorted(declared - orphans)}"
    )
    for rel, why in MATRIX["orphans"].items():
        assert why, rel


def test_performance_settings_have_a_decision():
    from yunshu_engine import settings

    cats = set(MATRIX["settings_categories"])
    mapped = MATRIX["settings"]
    features = set(MATRIX["features"]) | {"loop"}
    perf = {n for n, s in settings.REGISTRY.items() if s.category in cats}
    assert perf - set(mapped) == set(), (
        f"settings with no path decision (add to path_matrix.json settings): {sorted(perf - set(mapped))}"
    )
    assert set(mapped) - set(settings.REGISTRY) == set(), (
        "settings map lists unknown names"
    )
    for name, target in mapped.items():
        assert target in features or target.startswith("text-only"), f"{name}: {target}"


def test_gap_budget_only_shrinks():
    """missing + partial cells may not grow without an explicit budget bump here."""
    gaps = sum(1 for _, _, c in _cells() if c["s"] in ("missing", "partial"))
    assert gaps == MATRIX["gap_budget"], (
        f"{gaps} gap cells but gap_budget is {MATRIX['gap_budget']}: when a gap is closed lower "
        "gap_budget; a new gap needs a reviewer's decision, not a silent bump"
    )


@pytest.mark.parametrize("fid", sorted(MATRIX["features"]))
def test_feature_ids_are_slugs(fid):
    assert re.fullmatch(r"[a-z0-9_]+", fid)
