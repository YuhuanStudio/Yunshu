"""The native accuracy gate measures a restored singleton, not a cold no-op."""

import importlib.util
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_vlm")


def test_native_paired_gate_primes_both_arms_and_reports_hit_engagement(
    monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts/research"))
    import native_model_cache
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

    from yunshu_engine import vlm_engine

    spec = importlib.util.spec_from_file_location(
        "paired_probe", root / "scripts/research/probe_prefill_paired_quality.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    def original(*args, **kwargs):
        pass

    monkeypatch.setattr(Qwen3_5Model, "__call__", original)

    def wrap(fn):
        def native(*args, **kwargs):
            native.native_calls += 1

        native.native_calls = 0
        return native

    monkeypatch.setattr(native_model_cache, "wrap", wrap)

    class Executor:
        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

    class Runner:
        def __init__(self):
            self.cached = 0
            self.apc_manager = SimpleNamespace(clear=lambda: setattr(self, "cached", 0))
            self.processor = SimpleNamespace(
                tokenizer=SimpleNamespace(
                    decode=lambda tokens, **kwargs: str(tokens[0])
                )
            )

        def iter_tokens(self, ids, *, stats, max_tokens, **kwargs):
            stats.cached_tokens = self.cached
            if self.cached and hasattr(Qwen3_5Model.__call__, "native_calls"):
                Qwen3_5Model.__call__()
            stats.last_logprob = {"logprob": -0.25}
            yield 90
            stats.finish_reason = "stop"
            self.cached = len(ids) - 1

    class Engine:
        def __init__(self, model):
            self._executor, self._batch_runner = Executor(), Runner()

        async def start(self):
            pass

        async def stop(self):
            pass

        def _runner_input(self, *args):
            return [1, 2, 3, 4, 5], {}, 123

    monkeypatch.setattr(vlm_engine, "VLMEngine", Engine)
    out = tmp_path / "paired.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paired",
            "--out",
            str(out),
            "--items",
            "1",
            "--variant",
            "native",
            "--model",
            "tiny",
        ],
    )
    probe.main()
    import json

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[-1]["success"] and rows[-1]["correct"] == {"baseline": 1, "native": 1}
    assert rows[0]["arms"]["baseline"]["cached"] == 4
    assert rows[0]["arms"]["native"]["cached"] == 4
    assert rows[0]["arms"]["native"]["native_calls"] > 0


@pytest.mark.parametrize("variant", ["async", "spans"])
def test_async_paired_gate_uses_real_canonical_chat_and_requires_restore(
    monkeypatch, tmp_path, variant
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts/research"))
    import async_restore
    import span_forward
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

    from yunshu_engine import vlm_engine

    counter = "async_merges" if variant == "async" else "forwards"
    counts = {"enabled": True, counter: 0}

    def installed(*args, **kwargs):
        pass

    def install_span():
        monkeypatch.setattr(Qwen3_5Model, "__call__", installed)
        return counts, lambda: None

    monkeypatch.setattr(async_restore, "install", lambda: (counts, lambda: None))
    monkeypatch.setattr(span_forward, "install", install_span)
    spec = importlib.util.spec_from_file_location(
        "async_quality_probe", root / "scripts/research/probe_prefill_paired_quality.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    priming = list(range(100))
    chat = priming[:-1] + list(range(1000, 1200))
    requests = []

    class Executor:
        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

    class Runner:
        def __init__(self):
            self.cached = 0
            self.apc_manager = SimpleNamespace(
                prefill_stride=2048, clear=lambda: setattr(self, "cached", 0)
            )
            self.processor = SimpleNamespace(
                tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs: "90")
            )

        def iter_tokens(self, ids, *, stats, max_tokens, **kwargs):
            requests.append((list(ids), max_tokens, counts["enabled"]))
            stats.cached_tokens = self.cached
            if self.cached and counts["enabled"]:
                if variant == "spans":
                    assert Qwen3_5Model.__call__ is installed
                counts[counter] += 1
            stats.last_logprob = {"logprob": -0.25}
            yield 90
            stats.finish_reason = "stop"
            self.cached = len(ids) - 1

    class Engine:
        def __init__(self, model):
            self._executor, self._batch_runner = Executor(), Runner()

        async def start(self):
            pass

        async def stop(self):
            pass

        def _runner_input(self, messages, *args):
            return (priming if len(messages) == 1 else chat), {}, None

    monkeypatch.setattr(vlm_engine, "VLMEngine", Engine)
    out = tmp_path / "async-quality.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paired",
            "--out",
            str(out),
            "--items",
            "1",
            "--variant",
            variant,
            "--model",
            "tiny",
        ],
    )
    probe.main()
    import json

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[-1]["success"] and rows[-1]["different"] == 0
    assert requests == [
        (priming, 1, False),
        (chat, 16, False),
        (priming, 1, True),
        (chat, 16, True),
    ]
    assert rows[0]["arms"][variant]["cached"] == 99
    assert counts[counter] == 1
