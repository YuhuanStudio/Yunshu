import importlib.util
from pathlib import Path

import pytest


def probe():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_embedding_parity.py"
    spec = importlib.util.spec_from_file_location("priorfix_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_probe_fails_closed():
    p = probe()
    assert p.compare([[1, 0]], [[1, 0]], 0.999999)["passed"]
    assert not p.compare([[1, 0]], [[0, 1]], 0.999999)["passed"]
    with pytest.raises(ValueError):
        p.compare([[1, 0]], [[float("nan"), 0]], 0.99)
    args = p.parser().parse_args(
        [
            "--model",
            "fixture",
            "--reference",
            "fixture.json",
            "--out",
            "out",
            "--dry-run",
        ]
    )
    assert args.dry_run


def test_runtime_probe_cli_and_exact_rule():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_runtime_parity.py"
    spec = importlib.util.spec_from_file_location("priorfix_runtime_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.exact([[1, 2]], [[1, 2]])
    assert not module.exact([[1, 2]], [1, 2])
    assert not module.exact([1, 2], [1, 3])
    for kind in ["omni", "retrieval", "classifier", "diffusion"]:
        assert (
            module.parser()
            .parse_args(["--kind", kind, "--out", "out", "--dry-run"])
            .dry_run
        )


def test_embedding_probe_normalizes_every_real_fixture_before_loading(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "scripts/research"))
    from egemma2_cases import cases, make_media

    p = probe()
    assert p.processor_payload("a cat")["text"] == ["a cat"]
    fixtures = cases(make_media(str(tmp_path)))
    assert len(fixtures) == 13
    for item in fixtures.values():
        payload = p.processor_payload(item)
        assert payload["return_tensors"] == "np"
        assert len(payload["text"]) == 1


def test_qwen_official_ids_match_published_role_template():
    from jinja2 import Template

    from yunshu_engine.scoring_engine import DEFAULT_INSTRUCTION, qwen_input_ids

    template = Template(
        (
            Path(__file__).parents[1] / "fixtures/priorfix/qwen3_reranker.jinja"
        ).read_text()
    )

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text)[: kwargs["max_length"]]}

    messages = [
        {"role": "system", "content": DEFAULT_INSTRUCTION},
        {"role": "query", "content": "capital of France"},
        {"role": "document", "content": "Paris is the capital."},
    ]
    rendered = template.render(messages=messages)
    assert qwen_input_ids(
        Tokenizer(), messages[1]["content"], messages[2]["content"]
    ) == Tokenizer().encode(rendered)
    wrong = template.render(
        messages=[messages[0], {"role": "user", "content": messages[2]["content"]}]
    )
    assert messages[2]["content"] not in wrong


def test_fixed_input_reference_rejects_empty_and_keeps_ids_immutable():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_runtime_parity.py"
    spec = importlib.util.spec_from_file_location("runtime_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        module.FixedInputTokenizer([])
    ids = [1, 2, 3]
    t = module.FixedInputTokenizer(ids)
    ids[0] = 9
    got = t.encode("ignored")
    got[0] = 8
    assert t.encode("another") == [1, 2, 3]


def test_runtime_exact_supports_mlx_bfloat16_and_packed_uint32():
    import mlx.core as mx

    path = Path(__file__).parents[2] / "scripts/research/priorfix_runtime_parity.py"
    spec = importlib.util.spec_from_file_location("runtime_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.exact(mx.array([1.25, 2.5], dtype=mx.bfloat16), [1.25, 2.5])
    assert not module.exact(mx.array([1.25, 2.5], dtype=mx.bfloat16), [1.25, 3])
    assert module.exact(mx.array([4294967295], dtype=mx.uint32), [4294967295])


def test_diffusion_timing_requires_three_interleaved_same_device_pairs():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_diffusion_timing.py"
    spec = importlib.util.spec_from_file_location("timing_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"rep": rep, "arm": arm, "seconds": 2 if arm == "ours" else 1, "device": "M5"}
        for rep, arm in module.plan()
    ]
    assert module.summarize(rows)["mflux_over_ours"] == 0.5
    with pytest.raises(ValueError):
        module.summarize(rows[:-1])
    rows[0]["device"] = "M3"
    with pytest.raises(ValueError):
        module.summarize(rows)
    rows[0]["device"] = "M5"
    rows[0]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        module.summarize(rows)
