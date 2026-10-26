"""An agent conversation with a late system note after every tool result, rendered by the REAL
VLMEngine (qwen35 chat template, real Qwen tokenizer, family message adapter), run through
VLMBatchRunner + YunshuAPCManager on a random-weight hybrid ``qwen3_5`` in fp32.

Each turn's prompt starts with the previous turn's prompt (omlx 5a0a15a7: the note stays in
place), so with a split-invariant 64-aligned manager (what the Metal kernels give on a Mac) every
turn restores the previous turn's prompt-end checkpoint, and the tokens equal a cold run's."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("transformers")
pytest.importorskip("gguf")

from tests.unit.cpu_models import build_qwen35_dir, qwen_vocab_gguf, session_dir
from tests.unit.test_apc_manager_e2e import _processor, _run
from yunshu_engine.apc_manager import YunshuAPCManager
from yunshu_engine.vlm_batch_runner import VLMBatchRunner
from yunshu_engine.vlm_engine import VLMEngine

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        qwen_vocab_gguf() is None,
        reason="Qwen vocab GGUF absent: run `python -m tests.unit.cpu_models`",
    ),
]


def _conversation(turns: int) -> list[dict]:
    msgs = [
        {"role": "system", "content": "You are an agent. " * 8},
        {"role": "user", "content": "Fix the bug in module foo. " * 4},
    ]
    for i in range(turns):
        msgs += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "Read", "arguments": json.dumps({"p": i})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": "lorem ipsum " * 20},
            {"role": "system", "content": f"hook note {i}"},
        ]
    return msgs


@pytest.fixture(scope="module")
def engine():
    path = build_qwen35_dir(session_dir("qwen35-runner"))
    eng = VLMEngine(str(path))
    eng.load()  # on this thread: the runner below is driven from it too
    return eng


def _ids(eng, turns):
    prompt = eng._format_prompt(_conversation(turns))
    return eng._tokenizer.encode(prompt, add_special_tokens=False)


def test_each_turn_restores_the_previous_prompt_and_equals_a_cold_run(engine):
    tok = engine._tokenizer
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    user = tok.encode("user", add_special_tokens=False)[0]
    manager = YunshuAPCManager(
        num_blocks=256,
        block_size=16,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
        head_marker=(im_start, user),
        prefill_alignment=64,
    )
    runner = VLMBatchRunner(
        engine._model,
        processor=_processor(),
        apc_manager=manager,
        apc_semantic_hash=0,
    )
    cold = VLMBatchRunner(engine._model, processor=_processor())
    turns = [_ids(engine, n) for n in (0, 1, 2, 3)]
    gen = len(tok.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False))
    for a, b in zip(turns, turns[1:], strict=False):
        assert b[: len(a) - gen] == a[: len(a) - gen], (
            "the next prompt extends this one"
        )
    cached = []
    for ids in turns:
        out, stats = _run(runner, ids)
        cached.append(stats.cached_tokens)
        assert out == _run(cold, ids)[0], f"turn of {len(ids)} tokens: APC != cold"
    # turn 0 is cold; every later turn restores the previous prompt-end checkpoint
    # (the previous prompt minus its generation prompt, rounded down to 64)
    assert cached[0] == 0
    for prev, got in zip(turns[:-1], cached[1:], strict=True):
        assert got == (len(prev) - 1) // 64 * 64
        assert got > 0
