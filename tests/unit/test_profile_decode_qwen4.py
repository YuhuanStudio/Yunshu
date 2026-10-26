"""Qwen4 profiling follows fused forward calls instead of dense-only leaves."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_qwen4_wraps_actual_router_and_expert_kernels(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "profile_decode_qwen4",
        Path(__file__).resolve().parents[2] / "scripts/research/profile_decode_ops.py",
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpBatchInvariantForward as Forward,
    )

    from yunshu_engine.kernels.tf_flash_next import experts, hc

    for owner, names in (
        (
            Forward,
            (
                "_gated_delta",
                "_qsa_attention",
                "_ple",
                "_switch_glu",
                "_hyper_connection",
            ),
        ),
        (experts, ("router", "expert_gateup", "expert_down_y")),
        (hc, ("hc_norm", "hc_project")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, lambda *args, **kw: "result")
    events = []
    monkeypatch.setattr(probe.B, "enter", lambda label, x: events.append((label, x)))
    monkeypatch.setattr(probe.B, "leave", lambda label, x: None)
    assert probe.wrap_model(SimpleNamespace(model_type="qwen4_exp")) == []
    assert experts.router("input", "weights") == "result"
    assert experts.expert_gateup("rows", "logits") == "result"
    assert Forward()._switch_glu("mlp", "hidden") == "result"
    assert events == [
        ("moe.router", "input"),
        ("moe.expert_gateup", "rows"),
        ("moe.gather_qmm", "hidden"),
    ]
