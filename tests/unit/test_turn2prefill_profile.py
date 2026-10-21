import importlib.util
import sys
from pathlib import Path


def test_only_server_launch_is_wrapped():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/turn2prefill_profile.py"
    )
    spec = importlib.util.spec_from_file_location("turn2profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = ["/venv/bin/yunshu", "serve", "--port", "18996"]
    assert module.server_command(command) == [
        sys.executable,
        str(path),
        "--serve",
        *command[1:],
    ]
    assert module.server_command(["lsof", "-t"]) == ["lsof", "-t"]
    assert module.server_command([]) == []


def test_padding_preserves_aligned_lengths_and_covers_tail():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/turn2prefill_matmul.py"
    )
    spec = importlib.util.spec_from_file_location("turn2matmul", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for rows, padded in ((512, 512), (513, 640), (535, 640), (2047, 2048)):
        assert module.padded_rows(rows) == padded
