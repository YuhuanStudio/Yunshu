"""API-layer parity guard: no request/response feature is wired into only some routes by accident.

python/yunshu_gateway/api_matrix.json lists every request / response feature of the OpenAI and
Anthropic APIs the gateway serves and, for each route x stream mode (plus Responses background),
whether it is engaged (with code evidence and a probe or named test that exercises exactly that
cell), not applicable (with the spec's reason) or partial / missing (with a kind, an impact and an
owner). It mirrors python/yunshu_engine/path_matrix.json and tests/unit/test_path_parity.py.

What fails here:
- a cell without a status, or engaged without evidence plus a probe / test that runs it;
- evidence (a code needle, a test function) that no longer exists;
- a new request field on any of the four generation routes, a new server-tool kind or a new
  server_tools module with no row;
- the number of partial / missing cells differing from ``gap_budget`` (it may only go down, and
  closing a gap lowers it in the same commit);
- a duplicate-implementation entry pointing at code that is gone.

tests/unit/test_api_parity_wire.py runs the probes (scripted engine through the real routers);
static inspection only in this file, no MLX import, no GPU.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from .test_api_parity_wire import CHECKS

ROOT = Path(__file__).resolve().parents[2]
MATRIX = json.loads((ROOT / "python/yunshu_gateway/api_matrix.json").read_text())
STATUSES = {"engaged", "na", "missing", "partial"}
IMPACTS = {"high", "medium", "low"}
KINDS = {"unimplemented", "divergent", "untested"}
COLUMNS = set(MATRIX["columns"])
FEATURES = MATRIX["features"]


def _read(rel: str) -> str:
    path = ROOT / rel
    assert path.is_file(), f"{rel} does not exist"
    return path.read_text()


def _cells():
    for fid, feat in FEATURES.items():
        for key, cell in feat["cells"].items():
            yield fid, key, cell


def test_every_feature_has_every_column():
    for fid, feat in FEATURES.items():
        assert set(feat["cells"]) == COLUMNS, (
            f"{fid}: cells must cover exactly {sorted(COLUMNS)}"
        )
        assert feat["what"], fid
        assert re.fullmatch(r"[a-z0-9_]+", fid), fid


def test_cells_are_decided():
    for fid, key, cell in _cells():
        where = f"{fid}/{key}"
        assert cell["s"] in STATUSES, where
        if cell["s"] == "engaged":
            assert cell.get("ev"), f"{where}: engaged needs code evidence"
            assert cell.get("probe") or cell.get("tests"), (
                f"{where}: engaged needs a probe or a named test that exercises this cell"
            )
        else:
            assert cell.get("why"), f"{where}: {cell['s']} needs a reason"
        if cell["s"] in ("missing", "partial"):
            assert cell.get("kind") in KINDS, (
                f"{where}: kind must be one of {sorted(KINDS)}"
            )
            assert cell.get("impact") in IMPACTS, where
            assert cell.get("owner") in MATRIX["owners"], where


def test_probes_exist_and_cover_their_cells():
    for fid, key, cell in _cells():
        probe = cell.get("probe")
        if not probe:
            continue
        assert probe in FEATURES[fid]["probes"], (
            f"{fid}/{key}: probe {probe} not listed on the row"
        )
        assert key in CHECKS.get(probe, {}), (
            f"{fid}/{key}: probe {probe} has no check for this cell in test_api_parity_wire.py"
        )


def test_every_probe_is_used_by_a_row():
    used = {p for f in FEATURES.values() for p in f["probes"]}
    unused = sorted(set(CHECKS) - used)
    assert not unused, f"probes no row cites (dead checks): {unused}"
    missing = sorted(used - set(CHECKS))
    assert not missing, f"rows cite probes that do not exist: {missing}"


def test_evidence_is_in_the_code_and_the_tests():
    for fid, key, cell in _cells():
        for rel, needle in cell.get("ev", []):
            assert needle in _read(rel), (
                f"{fid}/{key}: '{needle}' no longer in {rel}; the wiring moved, update api_matrix.json"
            )
        for rel, name in cell.get("tests", []):
            src = _read(rel)
            leaf = name.split("::")[-1]
            assert re.search(rf"def {re.escape(leaf)}\b", src), (
                f"{fid}/{key}: test {leaf} not found in {rel}"
            )


def _model_fields(rel: str, cls: str) -> set[str]:
    tree = ast.parse(_read(rel))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
    )
    return {b.target.id for b in node.body if isinstance(b, ast.AnnAssign)}


def test_new_request_fields_need_a_row():
    code = {"chat": "c", "responses": "r", "messages": "m", "completions": "p"}
    for route, (rel, cls) in MATRIX["request_models"].items():
        fields = _model_fields(rel, cls)
        listed = {
            f for feat in FEATURES.values() for f in feat["fields"].get(code[route], [])
        }
        assert fields - listed == set(), (
            f"{route}: request fields with no row in api_matrix.json (add a feature, or list the "
            f"field on one): {sorted(fields - listed)}"
        )
        assert listed - fields == set(), (
            f"{route}: rows list fields the request model no longer has: {sorted(listed - fields)}"
        )


def test_server_tool_kinds_need_a_row():
    src = _read("python/yunshu_gateway/server_tools/runtime.py")
    m = re.search(r"kind: str\s+#\s*(.+)", src)
    assert m, "ToolDef.kind comment moved; update this test"
    kinds = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert kinds, m.group(1)
    rows = set()
    for f in FEATURES.values():
        st = f.get("server_tool") or []
        rows |= {st} if isinstance(st, str) else set(st)
    assert kinds - rows == set(), (
        f"server tool kinds with no api_matrix.json row (set server_tool on a feature): {sorted(kinds - rows)}"
    )
    assert rows - kinds == set(), (
        f"rows for server tools that no longer exist: {sorted(rows - kinds)}"
    )


def test_new_server_tool_modules_need_a_decision():
    known = {f["module"] for f in FEATURES.values() if f.get("module")}
    known |= set(MATRIX["modules_other"])
    undecided = []
    base = ROOT / "python/yunshu_gateway/server_tools"
    for p in sorted(base.iterdir()):
        if p.name == "__pycache__":
            continue
        rel = str(p.relative_to(ROOT))
        if rel in known or any(rel.startswith(k.rstrip("/") + "/") for k in known):
            continue
        undecided.append(rel)
    assert not undecided, (
        f"server_tools modules with no row in api_matrix.json (module on a feature, or modules_other): {undecided}"
    )
    for rel in known:
        assert (ROOT / rel).exists(), f"{rel} is listed but gone"


def test_duplicates_are_grounded():
    ids = set()
    for dup in MATRIX["duplicates"]:
        assert dup["id"] not in ids
        ids.add(dup["id"])
        assert len(dup["impls"]) >= 2 and dup["unify"] and dup["note"], dup["id"]
        for rel, needle in dup["impls"]:
            assert needle in _read(rel), (
                f"duplicate {dup['id']}: '{needle}' gone from {rel}"
            )


def test_gap_budget_only_shrinks():
    """partial + missing cells may not grow; closing one lowers gap_budget in the same commit."""
    gaps = sum(1 for _, _, c in _cells() if c["s"] in ("missing", "partial"))
    assert gaps == MATRIX["gap_budget"], (
        f"{gaps} gap cells but gap_budget is {MATRIX['gap_budget']}: when a gap is closed lower "
        "gap_budget; a new gap needs a reviewer's decision, not a silent bump"
    )


def test_summary_counts_add_up():
    total = sum(1 for _ in _cells())
    assert total == len(FEATURES) * len(COLUMNS)


@pytest.mark.parametrize("fid", sorted(FEATURES))
def test_feature_ids_are_slugs(fid):
    assert re.fullmatch(r"[a-z0-9_]+", fid)
