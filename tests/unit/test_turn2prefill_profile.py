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


def test_layer_spans_only_evaluate_active_prompt_components():
    from types import SimpleNamespace

    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/turn2prefill_profile.py"
    )
    spec = importlib.util.spec_from_file_location("turn2profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    records = []
    active = [False]
    mx = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        eval=lambda x: events.append(("eval", x)),
    )
    result = object()

    def original(self, x, **kw):
        return result

    wrapped = module.measured_call(
        original, "mlp", mx, lambda **r: records.append(r), lambda: active[0]
    )
    owner = object()
    x = SimpleNamespace(shape=(1, 532, 5120))
    assert wrapped(owner, x, cache=None) is result
    assert not events and not records
    active[0] = True
    assert wrapped(owner, x) is result
    assert events == ["sync", ("eval", result), "sync"]
    assert records[0]["rows"] == 532 and records[0]["span"] == "mlp"
    events.clear()
    assert wrapped(owner, SimpleNamespace(shape=(1, 1, 5120))) is result
    assert not events
