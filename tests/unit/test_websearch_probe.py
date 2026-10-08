"""CPU preflight for the GPU web tool probe and its yv validator."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "websearch_probe", ROOT / "scripts/research/websearch_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_and_fail_closed_evidence():
    module = load_probe()
    args = module.parser().parse_args(
        ["--model", "/model", "--src", "/tree/python", "--out", "/output.jsonl"]
    )
    assert not args.baseline
    assert module.validate([])[0] is False
    assert module.validate([{"complete": True}])[0] is False
    assert (
        module.validate([{"check": "search", "pass": False}, {"complete": True}])[0]
        is False
    )
    assert (
        module.validate(
            [{"check": "search", "pass": True}, {"complete": True, "pass": True}]
        )[0]
        is True
    )


def test_yv_stage_fake_executor_handles_dict_results(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    root = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(root))
    from verify import stages
    from verify.execute import CellResult

    class Executor:
        jobs = []

        def run_cells(self, cells):
            assert len(cells) == 2
            assert all(cell.device == "m5" and not cell.quiet for cell in cells)
            return {
                cell.key: CellResult(cell.key, True, evidence=None) for cell in cells
            }

    records = []
    ctx = SimpleNamespace(
        cand=SimpleNamespace(path=ROOT),
        model="tiny",
        py=sys.executable,
        exe=Executor(),
        run=SimpleNamespace(append=lambda stage, record: records.append(record)),
        tree=lambda arm: SimpleNamespace(path=ROOT),
        arm_env=lambda arm: {},
    )
    verdict = stages.stage_websearch(ctx)
    assert verdict.passed and verdict.numbers == {"base": [], "cand": []}
    assert records[-1]["ev"] == "stage_complete"


def test_probe_dependency_import_order_in_fresh_cpu_process():
    import subprocess
    import sys

    script = ROOT / "scripts/research/websearch_probe.py"
    code = "import sys; sys.path.insert(0, sys.argv[1]); import websearch_probe; deps = websearch_probe.load_dependencies(); assert len(deps) == 5"
    result = subprocess.run(
        [sys.executable, "-c", code, str(script.parent)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
