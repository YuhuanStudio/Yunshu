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
    ):
        assert guard.verdict(cmd) is None, cmd
