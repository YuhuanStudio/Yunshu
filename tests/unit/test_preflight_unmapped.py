"""yv preflight: a changed python/ module with no related unit test fails, unless allowlisted with a reason."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from verify import core  # noqa: E402


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "python" / "pkg").mkdir(parents=True)
    (tmp_path / "python" / "pkg" / "mapped.py").write_text("X = 1\n")
    (tmp_path / "python" / "pkg" / "orphan.py").write_text("Y = 2\n")
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_mapped.py").write_text(
        "from pkg.mapped import X\n"
    )
    return tmp_path


def test_changed_module_without_test_is_reported(tmp_path):
    tree = _tree(tmp_path)
    assert core.unmapped_python_files(["python/pkg/orphan.py"], tree) == [
        "python/pkg/orphan.py"
    ]


def test_module_with_a_related_test_passes(tmp_path):
    tree = _tree(tmp_path)
    assert core.unmapped_python_files(["python/pkg/mapped.py"], tree) == []


def test_non_python_and_deleted_files_are_ignored(tmp_path):
    tree = _tree(tmp_path)
    files = ["docs/x.md", "python/pkg/gone.py", "scripts/tool.py"]
    assert core.unmapped_python_files(files, tree) == []


def test_allowlisted_module_passes_with_its_reason(tmp_path):
    tree = _tree(tmp_path)
    allow = {"python/pkg/orphan.py": "covered by the end-to-end smoke only"}
    assert core.unmapped_python_files(["python/pkg/orphan.py"], tree, allow) == []


def test_allowlist_file_parses_path_and_reason(tmp_path):
    tree = _tree(tmp_path)
    (tree / core.PREFLIGHT_ALLOWLIST).parent.mkdir(parents=True, exist_ok=True)
    (tree / core.PREFLIGHT_ALLOWLIST).write_text(
        "# header comment\n\npython/pkg/orphan.py  # covered by smoke\n"
    )
    assert core.load_preflight_allowlist(tree) == {
        "python/pkg/orphan.py": "covered by smoke"
    }


def test_allowlist_entry_without_reason_is_an_error(tmp_path):
    tree = _tree(tmp_path)
    (tree / core.PREFLIGHT_ALLOWLIST).parent.mkdir(parents=True, exist_ok=True)
    (tree / core.PREFLIGHT_ALLOWLIST).write_text("python/pkg/orphan.py\n")
    with pytest.raises(ValueError, match="needs a path and a"):
        core.load_preflight_allowlist(tree)


def test_missing_allowlist_is_empty(tmp_path):
    assert core.load_preflight_allowlist(_tree(tmp_path)) == {}


def test_package_marker_without_logic_is_not_reported(tmp_path):
    tree = _tree(tmp_path)
    (tree / "python" / "pkg" / "__init__.py").write_text("")
    assert core.unmapped_python_files(["python/pkg/__init__.py"], tree) == []
