"""soak_mmlu_pro reads the dataset the gate passes (a pinned gate tree has no reference/)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/soak_mmlu_pro.py"


def _load():
    spec = importlib.util.spec_from_file_location("soak_mmlu_pro_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_dataset_overrides_the_tree_copy(monkeypatch, tmp_path):
    data = tmp_path / "mmlu_pro_test.jsonl"
    monkeypatch.setenv("MMLU_DATA", str(data))
    assert _load().DATASET == data


def test_default_is_the_tree_reference_copy(monkeypatch):
    monkeypatch.delenv("MMLU_DATA", raising=False)
    assert _load().DATASET == SCRIPT.parents[2] / (
        "reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl"
    )
