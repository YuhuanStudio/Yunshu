import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def probe():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_capabilities.py"
    spec = importlib.util.spec_from_file_location("priorfix_capabilities", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [None, "rc", "missing", "incomplete", "device"])
def test_seven_children_record_failure_and_continue(tmp_path, failure):
    module = probe()
    args = module.parser().parse_args(
        ["--out", str(tmp_path / "result.jsonl"), "--rerank-reference", "reference.py"]
    )
    calls = []

    def child(argv, check):
        assert not check
        calls.append(argv)
        out = Path(argv[-1])
        row = {"passed": True, "complete": True, "device": "M5"}
        if len(calls) == 2:
            if failure == "missing":
                return SimpleNamespace(returncode=0)
            if failure == "incomplete":
                row["complete"] = False
            if failure == "device":
                row["device"] = "M3"
        out.write_text(json.dumps(row) + "\n")
        return SimpleNamespace(
            returncode=1 if failure == "rc" and len(calls) == 2 else 0
        )

    arms = module.plan(args)
    # Stale successful evidence must be removed before the deliberately missing arm.
    Path(arms[1][1][-1]).parent.mkdir(parents=True)
    Path(arms[1][1][-1]).write_text(
        json.dumps({"passed": True, "complete": True, "device": "M5"})
    )
    result = module.run(args, child)
    assert len(calls) == len(result["arms"]) == 7
    assert result["passed"] is (failure is None)
    assert result["arms"]["omni"]["passed"]
    assert all(Path(argv[1]).exists() for argv in calls)
    assert "--rerank-reference" in calls[3]


def test_evidence_requires_object_and_boolean_complete(tmp_path):
    module = probe()
    path = tmp_path / "result"
    path.write_text(json.dumps({"passed": True, "complete": 1, "device": "M5"}))
    assert not module.evidence(path, 0)["passed"]
    path.write_text("not json")
    assert not module.evidence(path, 0)["passed"]
    path.write_text("[]")
    assert not module.evidence(path, 0)["passed"]


def test_cpu_admission_rejects_missing_reference_and_model(tmp_path):
    module = probe()
    ref = tmp_path / "reference.json"
    runtime = tmp_path / "reference.py"
    args = module.parser().parse_args(
        [
            "--out",
            str(tmp_path / "out"),
            "--reference",
            str(ref),
            "--rerank-reference",
            str(runtime),
            "--model-root",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="missing reference"):
        module.admission(args)
    ref.write_text("{}")
    runtime.write_text("pass\n")
    with pytest.raises(ValueError, match="missing model"):
        module.admission(args)
    for name in (
        "embeddinggemma-2-bf16",
        "embeddinggemma-2-4bit",
        "embeddinggemma-2-bf16-multishard",
    ):
        (tmp_path / name).mkdir()
    module.admission(args)


def test_yv_routes_capabilities_to_one_bounded_nonquiet_cell(monkeypatch, tmp_path):
    root = Path(__file__).parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from verify import stages

    (tmp_path / "aperepel-cfe20b0-server.py").write_text("pass\n")
    out = tmp_path / "evidence.jsonl"
    out.write_text(
        json.dumps({"passed": True, "complete": True, "device": "M5"}) + "\n"
    )
    cells = []

    def submit(value):
        cells.extend(value)
        return {"capabilities": SimpleNamespace(evidence=out)}

    monkeypatch.setattr(stages, "_failed_cells", lambda results: [])
    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    ctx = SimpleNamespace(
        env={
            "PRIORART_KINDS": "capabilities",
            "EMBEDDING_MODEL_ROOT": "model-root",
            "EMBEDDING_REFERENCE": "oracle.json",
        },
        cand=SimpleNamespace(path=root),
        run=SimpleNamespace(path=tmp_path),
        py="python",
        exe=SimpleNamespace(run_cells=submit),
    )
    result = stages.stage_priorart(ctx)
    assert len(cells) == 1
    cell = cells[0]
    assert str(root / "scripts/research/priorfix_capabilities.py") in cell.argv
    assert "--kind" not in cell.argv
    assert "model-root" in cell.argv and "oracle.json" in cell.argv
    assert cell.mem_gb == 64 and cell.timeout_min == 10 and cell.priority == -1
    assert not cell.quiet and cell.device == "m5"
    assert result.numbers["capabilities"]["passed"]
