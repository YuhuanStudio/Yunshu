"""The EngineCore continuous-batching loop (YUNSHU_ENGINE_LOOP) is gone for good (T0).

Nothing may re-introduce the loop modules, the setting, its path-matrix column or
imports of the deleted modules. A deliberate revival must delete this test first.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python" / "yunshu_engine"

DELETED_MODULES = {
    "engine_core",
    "scheduler",
    "scheduler_mixins",
    "paged_scheduler",
    "token_scheduler",
    "adaptive_batch",
    "two_batch_overlap",
    "cpu_gpu_overlap",
    "memory_aware_scheduler",
    "batch_sampler",
    "forward_batch",
    "inference_budget",
    "kv_lifecycle",
    "output_collector",
    "priority_queue",
    "request_lifecycle",
}
DELETED_SETTINGS = {
    "YUNSHU_ENGINE_LOOP",
    "YUNSHU_OVERLAP",
    "YUNSHU_COMPLETION_BATCH_SIZE",
    "YUNSHU_MAX_CONCURRENT",
}


def test_loop_modules_are_deleted():
    present = sorted(m for m in DELETED_MODULES if (PKG / f"{m}.py").exists())
    assert present == [], f"loop-only modules came back: {present}"


def _deleted_refs(path: Path, tree: ast.AST) -> list[str]:
    """Import statements in ``path`` that reach a deleted loop module."""
    hits = []
    in_package = path.parent == PKG
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level and in_package:
                first = [mod.split(".")[0]] if mod else []
                aliases = [a.name for a in node.names]
            elif mod.startswith("yunshu_engine") or mod.startswith(
                "python.yunshu_engine"
            ):
                parts = mod.split(".")
                first = (
                    [parts[parts.index("yunshu_engine") + 1]]
                    if len(parts) > parts.index("yunshu_engine") + 1
                    else []
                )
                aliases = (
                    [a.name for a in node.names]
                    if mod == "yunshu_engine" or mod.endswith("yunshu_engine")
                    else []
                )
            else:
                continue
            for name in first + [a for a in aliases if a in DELETED_MODULES]:
                if name in DELETED_MODULES:
                    hits.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} from {mod} import {name}"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if "yunshu_engine" in parts and parts.index("yunshu_engine") + 1 < len(
                    parts
                ):
                    if parts[parts.index("yunshu_engine") + 1] in DELETED_MODULES:
                        hits.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} import {alias.name}"
                        )
    return hits


def test_no_module_imports_a_deleted_module():
    offenders: list[str] = []
    for base in (ROOT / "python", ROOT / "tests", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            if "_vendor" in path.parts:
                continue
            offenders += _deleted_refs(path, ast.parse(path.read_text()))
    assert offenders == [], "imports of deleted loop modules:\n" + "\n".join(offenders)


def test_setting_is_unregistered():
    from yunshu_engine import settings

    assert DELETED_SETTINGS.isdisjoint(settings.REGISTRY), (
        "loop-only settings are back in the registry"
    )


def test_path_matrix_has_no_loop_path():
    matrix = json.loads((PKG / "path_matrix.json").read_text())
    assert "loop" not in matrix["paths"]
    for fid, feature in matrix["features"].items():
        assert "loop" not in feature["cells"], fid
    for name in DELETED_SETTINGS:
        assert name not in matrix["settings"], name


def test_loop_entry_points_are_gone():
    from yunshu_engine.batched_engine import BatchedEngine

    for attr in (
        "_ensure_engine_core",
        "_should_use_engine_loop",
        "_wire_spec_decoders_to_scheduler",
    ):
        assert not hasattr(BatchedEngine, attr), attr
