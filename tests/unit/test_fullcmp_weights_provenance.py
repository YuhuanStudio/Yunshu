"""Weights provenance per engine (scripts/research/fullcmp_audit.py check_weight_identity; the launch
spec is bench_engines, the one source). Synthetic headers only: no GPU, no server, no gpuq.

Failing before the fix: llama-cpp's header recorded the Jundot MLX safetensors although the launch
loads the UD-Q4_K_M GGUF, and nothing compared the header with the launch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_audit as fa  # noqa: E402

LLAMA = "llama-cpp-b12061"
SHA = "a" * 64


def _run(header, engine=LLAMA):
    cell = {"cell": "c1", "engine": engine, "weight_group": "g"}
    cf = SimpleNamespace(
        cell=cell, rep=0, path=Path("/x/c1-r0.jsonl"), header=header, rows=[]
    )
    return SimpleNamespace(files=[cf])


def _mlx_weights():
    return [
        {
            "file": "model-00001-of-00004.safetensors",
            "sha256": SHA,
            "bits": 4,
            "group_size": 64,
        },
    ]


def _gguf_weights():
    return be.gguf_weights_record(LLAMA)


def test_llama_header_with_mlx_safetensors_fails():
    header = {"weights": _mlx_weights(), "weights_note": be.weights_note(LLAMA)}
    res = fa.check_weight_identity(_run(header))
    assert res.violations, "the MLX shards are not what llama-server loads"
    assert any("not the files the launch loads" in v for v in res.violations)


def test_llama_header_with_gguf_weights_and_note_passes():
    header = {"weights": _gguf_weights(), "weights_note": be.weights_note(LLAMA)}
    res = fa.check_weight_identity(_run(header))
    assert res.violations == []
    assert header["weights_note"] == "different weights (GGUF UD-Q4_K_M + MTP Q4_0)"


def test_llama_gguf_header_without_the_note_fails():
    res = fa.check_weight_identity(_run({"weights": _gguf_weights()}))
    assert any("weights_note" in v for v in res.violations)


def test_gguf_record_is_the_loaded_files_with_their_quants():
    rec = be.gguf_weights_record(LLAMA)
    assert [e["file"] for e in rec] == [
        "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "mtp-Qwen3.8-27B-Q4_0.gguf",
    ]
    assert [e["quant"] for e in rec] == ["UD-Q4_K_M", "MTP Q4_0"]
    for e in rec:
        assert len(e["sha256"]) == 64 and e["sha256"] != SHA


def test_no_engine_other_than_yunshu_is_labelled_same_weights_by_mistake():
    # the note says 'same weights' only for the Jundot checkpoint the Yunshu launch serves
    assert be.weights_note("yunshu-main").startswith("same weights as Yunshu")
    assert be.weights_note("mlx-serve-26.11.1").startswith("same weights as Yunshu")
    assert be.weights_note("kapitsa-0.2.26").startswith("different weights")
    assert be.weights_note("tf-1.2.1").startswith("different weights")
    assert be.weights_note(LLAMA).startswith("different weights")


def test_served_dirs_follow_the_launch_spec():
    assert be.served_model_dir("kapitsa-0.2.26", "J") == be.KAPITSA_MODEL_DIR
    assert be.served_model_dir("tf-1.2.1", "J") == be.TF_NATIVE_MODEL
    assert be.served_model_dir("yunshu-main", "J") == "J"
