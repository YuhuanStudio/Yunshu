"""Fail-closed golden validation without MLX or a checkpoint."""

import copy

import pytest
from scripts.research.round_driver_golden import (
    GLOBAL_STATES,
    compare,
    digest,
    validate,
)


def receipt():
    payload = dict(
        layer="generic",
        device="cpu",
        tokens=4,
        globals=dict.fromkeys(GLOBAL_STATES, False),
        capabilities=dict(
            spec=False,
            apc=False,
            replay=False,
            int8=False,
            apc_unavailable_reason="fixture",
        ),
        int8_reference=[],
        spec_cells=[],
        cache_cells=[],
        cells=[
            dict(
                mode=m,
                schedule=s,
                rows=[
                    [dict(token=3, logprob=(-0.25).hex()) for _ in range(4)]
                    for _ in range(2 if s == "cancel" else 3)
                ],
            )
            for m in ("greedy", "seeded")
            for s in ("solo", "burst", "stagger", "cancel", "media", "repeat")
        ],
    )
    return dict(complete=True, payload=payload, sha256=digest(payload))


def test_exact_receipt_comparison():
    compare(receipt(), receipt())


@pytest.mark.parametrize(
    "damage", ["complete", "checksum", "matrix", "globals", "truncated", "nan"]
)
def test_missing_or_corrupted_evidence_fails_closed(damage):
    r = receipt()
    if damage == "complete":
        r["complete"] = False
    elif damage == "checksum":
        r["sha256"] = "bad"
    elif damage == "matrix":
        r["payload"]["cells"].pop()
    elif damage == "globals":
        r["payload"]["globals"].pop("ragged_kv._STATE")
    elif damage == "truncated":
        r["payload"]["cells"][0]["rows"][0].pop()
    else:
        r["payload"]["cells"][0]["rows"][0][0]["logprob"] = "nan"
    if damage != "checksum":
        r["sha256"] = digest(r["payload"])
    with pytest.raises(ValueError):
        validate(r)


def test_one_logprob_bit_changes_the_golden():
    a = receipt()
    b = copy.deepcopy(a)
    b["payload"]["cells"][0]["rows"][0][0]["logprob"] = (-0.25000000000000006).hex()
    b["sha256"] = digest(b["payload"])
    with pytest.raises(ValueError, match="golden mismatch"):
        compare(a, b)


def test_module_scope_leak_is_reported(monkeypatch):
    from scripts.research import round_driver_golden as golden

    expected = dict.fromkeys(GLOBAL_STATES, False)
    actual = dict(expected)
    actual["gdn_prefill._TAIL"] = True
    monkeypatch.setattr(golden, "global_state", lambda: actual)
    with pytest.raises(AssertionError, match="module state leaked"):
        golden.assert_globals(expected)


def test_yv_missing_cell_cannot_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.verify import stages

    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    ctx = SimpleNamespace(
        env={"M0_LAYER": "generic"},
        run=SimpleNamespace(path=tmp_path),
        cand=SimpleNamespace(path=tmp_path),
        py="python",
        model="tiny",
        exe=SimpleNamespace(run_cells=lambda cells: {}),
    )
    result = stages.stage_round_golden(ctx)
    assert not result.passed


def test_cli_writes_a_complete_json_line(tmp_path, monkeypatch):
    import json
    import sys

    from scripts.research import round_driver_golden as golden

    out = tmp_path / "receipt.jsonl"
    monkeypatch.setattr(golden, "capture", lambda *args: receipt())
    monkeypatch.setattr(
        sys,
        "argv",
        ["golden", "--layer", "generic", "--device", "cpu", "--out", str(out)],
    )
    golden.main()
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 1 and records[-1]["complete"] is True


def test_apc_probe_uses_a_prompt_above_the_cache_floor():
    from scripts.research.round_driver_golden import cache_prompt_index

    prompts = [list(range(18)), list(range(11)), list(range(23))]
    assert len(prompts[cache_prompt_index(prompts)]) > 16
    with pytest.raises(ValueError, match="16-token floor"):
        cache_prompt_index([[1], [2, 3]])


def test_gpu_capture_requires_queue_before_importing_mlx(monkeypatch):
    from scripts.research.round_driver_golden import capture

    monkeypatch.delenv("GPUQ_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="gpuq admission"):
        capture("qwen3_5")


@pytest.mark.parametrize("capability", ["spec", "apc", "int8"])
def test_capability_without_evidence_fails(capability):
    r = receipt()
    r["payload"]["capabilities"][capability] = True
    r["sha256"] = digest(r["payload"])
    with pytest.raises(ValueError):
        validate(r)


def test_receipt_survives_gpuq_recursive_device_annotations(tmp_path, monkeypatch):
    import json
    import sys

    from scripts.research import round_driver_golden as golden
    from scripts.verify.stages import _round_golden_valid

    out = tmp_path / "receipt.jsonl"
    monkeypatch.setattr(golden, "capture", lambda *args: receipt())
    monkeypatch.setattr(
        sys,
        "argv",
        ["golden", "--layer", "generic", "--device", "cpu", "--out", str(out)],
    )
    golden.main()
    record = json.loads(out.read_text())

    def annotate(value):
        if isinstance(value, dict):
            for child in list(value.values()):
                annotate(child)
            value.update(device="m5", execution_device="m5", remote_host=None)
        elif isinstance(value, list):
            for child in value:
                annotate(child)

    annotate(record)
    out.write_text(json.dumps(record) + "\n")
    assert _round_golden_valid(out) == (True, "")


def test_adaptive_step_counts_are_evidence_not_output_identity():
    a, b = receipt(), receipt()
    a["payload"]["execution"] = [{"steps": 8, "drafted": 7}]
    b["payload"]["execution"] = [{"steps": 9, "drafted": 6}]
    for record in (a, b):
        record["sha256"] = digest(record["payload"])
    compare(a, b)


def test_spec_token_rows_without_engaged_drafter_evidence_fail():
    r = receipt()
    r["payload"]["capabilities"]["spec"] = True
    r["payload"]["spec_cells"] = [
        copy.deepcopy(c)
        for c in r["payload"]["cells"]
        if c["schedule"] in ("solo", "burst", "stagger")
    ]
    r["sha256"] = digest(r["payload"])
    with pytest.raises(ValueError, match="speculative execution evidence"):
        validate(r)


def test_yv_golden_scratch_stays_on_run_disk(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.verify import stages

    cells = []
    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    ctx = SimpleNamespace(
        env={"M0_LAYER": "generic"},
        run=SimpleNamespace(path=tmp_path),
        cand=SimpleNamespace(path=tmp_path),
        py="python",
        model="tiny",
        exe=SimpleNamespace(run_cells=lambda batch: cells.extend(batch) or {}),
    )
    stages.stage_round_golden(ctx)
    scratch = tmp_path / "round-golden-home"
    assert f"TMPDIR={scratch / 'tmp'}" in cells[0].argv
    assert (scratch / "tmp").is_dir()


def test_yv_golden_checkpoint_memory_is_overridable(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.verify import stages

    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    seen = {}

    def run_cells(batch):
        seen[batch[0].key] = batch[0].mem_gb
        return {batch[0].key: SimpleNamespace(ok=True, evidence=None)}

    for env, want in (({}, 80.0), ({"M0_CHECKPOINT_MEM_GB": "30"}, 30.0)):
        seen.clear()
        ctx = SimpleNamespace(
            env={"M0_LAYER": "qwen3_5_moe", "M0_TINY_ONLY": "0", **env},
            run=SimpleNamespace(path=tmp_path),
            cand=SimpleNamespace(path=tmp_path),
            py="python",
            model="/models/moe",
            exe=SimpleNamespace(run_cells=run_cells),
        )
        stages.stage_round_golden(ctx)
        assert seen.get("checkpoint") == want


def test_real_checkpoint_apc_probe_crosses_a_checkpoint():
    from scripts.research.round_driver_golden import (
        APC_PROBE_MIN_TOKENS,
        apc_probe_prompt,
        cache_prompt_index,
    )

    # Regression: the real-checkpoint prompts were <= 33 tokens, so no checkpoint
    # was stored and the warm APC hit could never happen.
    with pytest.raises(ValueError, match="too few tokens"):
        apc_probe_prompt(lambda text: [1] * 5)
    prompt = apc_probe_prompt(lambda text: list(range(len(text))))
    assert len(prompt) >= APC_PROBE_MIN_TOKENS > 512
    prompts = [list(range(17)), list(range(33)), prompt]
    assert cache_prompt_index(prompts) == 2


def test_real_prompts_keep_short_cells_and_add_a_probe(monkeypatch):
    import sys
    import types

    from scripts.research import round_driver_golden as golden

    class Tok:
        def encode(self, text):
            return list(range(len(text)))

    fake = types.ModuleType("mlx_lm.utils")
    fake.load_tokenizer = lambda path: Tok()
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake)
    ids = list(range(200))
    prompts = golden.real_prompts(ids, "/models/x")
    # Cells use indices 0-2: they must keep the short prompts that were validated.
    assert [len(p) for p in prompts[:3]] == [17, 33, 25]
    # The APC probe is the longest prompt and crosses a checkpoint interval.
    assert golden.cache_prompt_index(prompts) == 3
    assert len(prompts[3]) >= golden.APC_PROBE_MIN_TOKENS > 512


def test_checkpoint_without_mtp_tensors_runs_without_a_head():
    from scripts.research.round_driver_golden import optional_mtp_head

    def no_mtp(model):
        raise ValueError(f"No MTP tensors found in {model}")

    assert optional_mtp_head(no_mtp, "/models/moe") is None
    assert optional_mtp_head(lambda model: "head", "/models/x") == "head"
    with pytest.raises(ValueError, match="corrupt"):
        optional_mtp_head(lambda m: (_ for _ in ()).throw(ValueError("corrupt")), "/m")


def test_apc_probe_sizing_keeps_tiny_and_grows_for_checkpoints():
    from scripts.research.round_driver_golden import apc_probe_sizing

    # The tiny digests were captured with 64 blocks / 0.1 GB: they must not change.
    assert apc_probe_sizing(None) == (64, 0.1)
    # A real 1024-token prompt fills a 64-block pool exactly and exceeds 0.1 GB.
    blocks, gb = apc_probe_sizing("/models/27b")
    assert blocks * 16 > 1024 and gb > 0.1


def checkpoint_receipt(kernels=None, long_burst=True, layer="qwen3_5"):
    """A real-checkpoint shaped receipt built from the tiny fixture (no MLX, no checkpoint)."""
    r = receipt()
    p = r["payload"]
    p["layer"] = layer
    p["checkpoint"] = {"config_sha256": "fixture", "weight_sizes": {}}
    p["serving_kernels"] = (
        {"split_prefill": "split64-v1"} if kernels is None else kernels
    )
    if long_burst:
        p["long_burst"] = [
            dict(
                mode=mode,
                schedule="burst",
                rows=[
                    [dict(token=3, logprob=(-0.25).hex()) for _ in range(4)]
                    for _ in range(3)
                ],
            )
            for mode in ("greedy", "seeded")
        ]
    r["sha256"] = digest(p)
    return r


def test_served_qwen3_5_checkpoint_receipt_is_valid():
    validate(checkpoint_receipt())


def test_qwen3_5_checkpoint_without_split_prefill_fails_closed():
    with pytest.raises(ValueError, match="split-invariant prefill"):
        validate(checkpoint_receipt(kernels={}))


def test_checkpoint_without_long_row_burst_fails_closed():
    with pytest.raises(ValueError, match="long-row burst"):
        validate(checkpoint_receipt(long_burst=False))


def test_checkpoint_build_runs_the_served_kernel_install(tmp_path, monkeypatch):
    """build() of a real checkpoint installs the same kernel stack VLMEngine serves."""
    import json
    from types import SimpleNamespace

    from scripts.research import round_driver_golden as golden

    from yunshu_engine import serving_kernels
    from yunshu_engine.vlm_engine import VLMEngine

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    wrapper = SimpleNamespace(language_model=SimpleNamespace())
    monkeypatch.setattr(VLMEngine, "_load_vision_model", lambda self, path: wrapper)
    monkeypatch.setattr(
        "mlx_lm.utils.load_tokenizer",
        lambda path: SimpleNamespace(encode=lambda text, **kw: list(range(len(text)))),
    )
    import yunshu_engine.mlxvlm_mtp as mtp

    monkeypatch.setattr(mtp, "_load_drafter_in_memory", lambda path: None)
    seen = {}

    def fake_install(model, config, **kwargs):
        seen.update(model=model, config=config, **kwargs)
        return {"split_prefill": "split64-v1"}, None

    monkeypatch.setattr(serving_kernels, "install", fake_install)
    built = golden.build("qwen3_5", str(tmp_path))
    assert built[0] is wrapper and built[4] == {"split_prefill": "split64-v1"}
    assert seen["model"] is wrapper and seen["draft_kind"] == "mtp"
    assert seen["config"]["model_type"] == "qwen3_5"
