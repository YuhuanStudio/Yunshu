"""A dependency-upgrade A/B serves each arm from its own venv, only when marked."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tfbench = _load("tfbench_ov", "scripts/research/tfbench.py")
memory_ab = _load("memory_ab_ov", "scripts/research/memory_ab.py")


def _tree(tmp_path, marked):
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/yunshu").write_text("")
    (tmp_path / ".venv/bin/python").write_text("")
    (tmp_path / "python").mkdir()
    if marked:
        (tmp_path / ".yv-own-venv").write_text("")
    return tmp_path


def test_marked_tree_uses_its_own_venv(tmp_path):
    t = _tree(tmp_path, True)
    assert tfbench.own_venv_bin(str(t / "python"), "yunshu") == str(
        t / ".venv/bin/yunshu"
    )
    assert memory_ab._tree_python(str(t)) == str(t / ".venv/bin/python")


def test_unmarked_tree_uses_the_shared_venv(tmp_path):
    t = _tree(tmp_path, False)
    assert tfbench.own_venv_bin(str(t / "python"), "yunshu") is None
    assert memory_ab._tree_python(str(t)) != str(t / ".venv/bin/python")
    assert tfbench.own_venv_bin("", "yunshu") is None
