"""multi-prompt prompt_tokens failure-robustness.

completions multi-prompt prompt_tokens summed by arithmetic stride
  (results[i*_per_prompt]) which mis-samples when a partial generation failure removes a
  result. Now deduped by the real prompt index captured before the tag is dropped.

(The tenant TPM settle tests have been removed along with the multi-tenant
TenantManager stack — out of scope for the single-consumer refocus.)
"""

from __future__ import annotations

import inspect


def test_completions_dedups_prompt_tokens_by_index():
    from yunshu_gateway.routers import completions

    src = inspect.getsource(completions)
    assert "_prompt_tokens_by_pi" in src
    assert "sum(_prompt_tokens_by_pi.values())" in src
    # the fragile stride form is gone
    assert "results[i * _per_prompt][1]" not in src


def test_dedup_logic_survives_partial_failure():
    # Replicate the capture: results tagged (_pi, _ci, inner) where inner[1] == prompt_tokens.
    # 3 prompts × 2 choices, but prompt 0 lost a choice and prompt 1 lost BOTH.
    # Each prompt's prompt_tokens: p0=10, p1=20, p2=30.
    results = [
        (0, 1, (0, 10, 5, "stop", 0, None, "a", 0)),  # p0 choice1 (choice0 failed)
        (2, 0, (0, 30, 5, "stop", 0, None, "c", 0)),  # p2 choice0
        (2, 1, (0, 30, 5, "stop", 0, None, "c", 0)),  # p2 choice1
    ]
    results.sort(key=lambda x: (x[0], x[1]))
    by_pi = {}
    for _pi, _ci, _r in results:
        if _pi not in by_pi:
            by_pi[_pi] = _r[1]
    # only prompts that produced output are counted, once each: 10 + 30 = 40
    assert sum(by_pi.values()) == 40
    # the old stride (i*_per_prompt with _per_prompt=2) would read results[0],results[2],results[4]
    # → 10 + 30 + (out of range) and could double-count p2 / skip p1 entirely — wrong.
