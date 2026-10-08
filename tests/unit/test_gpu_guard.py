import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gpu_guard", Path(__file__).resolve().parents[2] / "scripts/dev/gpu_guard.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_direct_gpu_work_is_blocked():
    for cmd in (
        "uv run --no-sync python - <<EOF\nimport mlx.core as mx\nEOF",
        'python3 -c "import mlx.core as mx; print(1)"',
        ".venv/bin/python -m yunshu_cli serve --model m --port 18990",
        "uv run python scripts/research/memory_ab.py --arm a=b",
        "/x/rapid-mlx/.venv/bin/python -m rapid_mlx.cli serve m",
        "git status; uv run python scripts/research/memory_ab.py --arm a=b",
        "cd x && python3 - <<'EOF'\nimport mlx.core as mx\nEOF",
    ):
        assert guard.verdict(cmd), cmd


def test_gpuq_tests_and_reads_pass():
    for cmd in (
        "scripts/dev/gpuq submit --label x -- .venv/bin/python scripts/research/memory_ab.py",
        "uv run pytest tests/unit/test_memory_census.py -q",
        "uv run python -m py_compile scripts/research/memory_ab.py",
        'grep -rn "import mlx" python/',
        "sed -n 1,40p scripts/research/process_memory.py",
        "git log --oneline -3",
        "/usr/bin/python3 tools/watchdog.py check; git diff main -- scripts/research/tfbench.py",
    ):
        assert guard.verdict(cmd) is None, cmd


def test_pattern_kills_are_blocked():
    # A pattern kill reaches processes this session did not start; only PIDs may be killed.
    blocked = (
        "pk" + "ill -f x",
        "cd x && kill" + "all python3",
        "nice -n 5 pk" + "ill y",
    )
    for cmd in blocked:
        assert guard.verdict(cmd), cmd
    assert guard.verdict("kill 12345") is None
    stash = "git st" + "ash"
    for cmd in (
        stash,
        stash + " -q",
        "git -C ../wt st" + "ash push",
        "cd x && " + stash + " pop",
    ):
        assert guard.verdict(cmd), cmd
    assert guard.verdict(stash + " list") is None
    assert guard.verdict("grep pk" + "ill notes.md") is None
