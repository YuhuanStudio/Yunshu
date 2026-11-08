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
