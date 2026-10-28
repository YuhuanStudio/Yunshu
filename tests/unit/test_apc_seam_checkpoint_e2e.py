"""Seam checkpoint through the real BatchGenerator on a tiny hybrid (GDN) model:
a follow-up with a new question restores at the document and equals a cold run."""

import random

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_apc_manager_e2e import (  # noqa: E402
    IM_START,
    USER,
    VLMBatchRunner,
    YunshuAPCManager,
    _processor,
    _run,
    model,  # noqa: F401  (fixture)
)


def _manager(seam):
    return YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
        head_marker=(IM_START, USER),
        prefill_alignment=64,
        seam_checkpoint=seam,
        seam_min_message_tokens=256,
        seam_question_tokens=128,
    )


def test_follow_up_with_a_new_question_resumes_at_the_seam_and_equals_cold(model):  # noqa: F811
    if model.language_model.model.embed_tokens.weight.dtype != mx.float32:
        # off vs cold already differs in bf16 on this CPU stack (stock prefill is span-dependent);
        # split-invariant bf16 needs the Metal kernels. fp32 is exact.
        pytest.skip(
            "bf16 prefill is span-dependent without the split-invariant kernels"
        )
    rnd = random.Random(11)

    def toks(n):
        return [rnd.randrange(3, 800) for _ in range(n)]

    head, doc = toks(100), toks(420)
    s1 = [7, *head, IM_START, USER, *doc, *toks(100)]
    s2 = [7, *head, IM_START, USER, *doc, *toks(110)]
    results = {}
    for seam in (False, True):
        runner = VLMBatchRunner(
            model,
            processor=_processor(),
            apc_manager=_manager(seam),
            apc_semantic_hash=0,
        )
        _, st1 = _run(runner, s1)
        out2, st2 = _run(runner, s2)
        results[seam] = (out2, st1.cached_tokens, st2.cached_tokens)
    cold, _ = _run(VLMBatchRunner(model, processor=_processor()), s2)
    out_off, _, cached_off = results[False]
    out_on, first_cached, cached_on = results[True]
    assert first_cached == 0
    seam_at = (len(s1) - 1 - 128) // 64 * 64
    assert cached_off == 64  # the system turn only
    assert cached_on == seam_at > cached_off  # the document
    assert out_off == cold, "off vs cold"
    assert out_on == cold, "on vs cold"
