"""yv runs each arm's own scripts/research/tfbench.py (its commit), not the main checkout's
working copy, and the harness hash is the arm's (2026-11-02: main's uncommitted tfbench edits
changed in-flight runs, and the verdict could not say which harness ran)."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify import stages  # noqa: E402
from verify.core import Arm, RunDir  # noqa: E402


def _arm(name, path):
    (path / "python").mkdir(parents=True, exist_ok=True)
    (path / "scripts/research").mkdir(parents=True, exist_ok=True)
    (path / "scripts/research/tfbench.py").write_text(f"# {name}\n")
    return Arm(name, str(path), name * 40, path, "")


def _ctx(tmp_path, base, cand):
    return stages.Ctx(
        run=RunDir(tmp_path / "run"),
        exe=None,
        base=base,
        cand=cand,
        env={},
        cand_env={},
        base_env={},
        model="/models/m",
        model_name="m",
        suite={},
        mem_gb=10,
    )


def test_each_arm_runs_its_own_tfbench(tmp_path):
    base = _arm("a", tmp_path / "base")
    cand = _arm("c", tmp_path / "cand")
    ctx = _ctx(tmp_path, base, cand)
    argv = stages._tfbench_argv(ctx, "cand", "smoke", "cand", [])
    script = argv[argv.index(str(cand.path / "scripts/research/tfbench.py"))]
    assert script == str(cand.path / "scripts/research/tfbench.py")
    # the interpreter may be the main venv; the benchmark script must be the arm's own
    assert str(stages.REPO / "scripts/research/tfbench.py") not in " ".join(argv)
    base_argv = stages._tfbench_argv(ctx, "base", "smoke", "base", [])
    assert str(base.path / "scripts/research/tfbench.py") in base_argv


def test_harness_hash_is_the_arm_tfbench_blob(tmp_path):
    base = _arm("a", tmp_path / "base")
    cand = _arm("c", tmp_path / "cand")
    ctx = _ctx(tmp_path, base, cand)
    assert stages._harness_hash(ctx.tree("cand").path) != stages._harness_hash(
        base.path
    )
    assert stages._harness_hash(cand.path) == stages._file_hash(
        cand.path / "scripts/research/tfbench.py"
    )
