"""CPU checks of the memory A/B harness (no server, no GPU)."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load():
    path = ROOT / "scripts" / "research" / "memory_ab.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("memory_ab_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_code_doc_is_deterministic_and_sized():
    m = _load()
    a, b = m.code_doc(7, 2000), m.code_doc(7, 2000)
    assert a == b and a != m.code_doc(8, 2000)
    assert len(a) > 2000


def test_arm_env_is_per_arm(monkeypatch, tmp_path):
    m = _load()
    seen = {}

    def fake_run(name, tree, model, port, rep, emit, extra_env=None):
        seen[name] = dict(extra_env or {})

    monkeypatch.setattr(m, "run_arm", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "memory_ab.py",
            "--arm",
            "a=/x",
            "--arm",
            "b=/y",
            "--arm-env",
            "b:YUNSHU_VLM_APC_MEMORY_GB=0",
            "--model",
            "/m",
            "--reps",
            "1",
            "--out",
            str(tmp_path / "o.jsonl"),
        ],
    )
    m.main()
    assert seen == {"a": {}, "b": {"YUNSHU_VLM_APC_MEMORY_GB": "0"}}
    assert '"complete": true' in (tmp_path / "o.jsonl").read_text()


def test_trim_ttft_summary_is_a_per_phase_median():
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    import trim_ttft

    rows = [
        {"phase": "hit_now", "secs": 0.9},
        {"phase": "hit_now", "secs": 1.1},
        {"phase": "hit_now", "secs": 1.0},
        {"phase": "hit_after_idle", "secs": 1.3},
    ]
    assert trim_ttft.summarize(rows) == {"hit_now": 1.0, "hit_after_idle": 1.3}


def test_apc_clone_probe_parses_arguments():
    import pytest

    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    import apc_clone_probe

    with pytest.raises(SystemExit) as exc:
        apc_clone_probe.main(["--help"])
    assert exc.value.code == 0


def test_apc_restore_probe_parses_arguments():
    import pytest

    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    import apc_restore_probe

    with pytest.raises(SystemExit) as exc:
        apc_restore_probe.main(["--help"])
    assert exc.value.code == 0


def test_apc_restore_time_parses_arguments():
    import pytest

    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    import apc_restore_time

    with pytest.raises(SystemExit) as exc:
        apc_restore_time.main(["--help"])
    assert exc.value.code == 0
