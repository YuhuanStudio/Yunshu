"""(MED): /v1/responses n>1 had NO per-choice failure isolation.

chat.py and completions.py both wrap each choice in try/except so a transient mid-generation
MemoryError/TimeoutError/RuntimeError on one choice returns the OTHER (already-generated,
billed) choices instead of 500/507-ing the entire request. responses.py was the lone outlier
(the previously-deferred "reindent risk" item) — choice 0 could succeed and be discarded when
choice 1 threw. Now each engine call is wrapped (continue on failure, append to _choice_errors),
and only an all-choices-failed request re-raises so the outer 507/500 handler responds.

This endpoint is a deep async handler needing a live engine/model-manager, so the isolation
structure is source-guarded (matching test_responses_per_choice_seed_w797.py's approach).
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    responses as R,  # noqa: N812  # intentional short module alias
)


def test_n_choice_engine_calls_are_failure_isolated():
    src = inspect.getsource(R.create_response)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # an accumulator for per-choice failures
    assert "_choice_errors" in code
    # BOTH engine calls (batched chat + non-batched generate) skip on failure
    assert code.count("_choice_errors.append(_choice_exc)") >= 2
    assert code.count("continue") >= 2
    # client-disconnect (HTTPException/499) must still propagate, not be swallowed as a choice failure
    assert "except HTTPException:" in code


def test_all_choices_failed_reraises_not_empty_200():
    src = inspect.getsource(R.create_response)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # if every choice failed → re-raise the last error (outer 507/500), NOT a 200 with no output
    assert "if not all_output_items and _choice_errors:" in code
    assert "raise _choice_errors[-1]" in code


def test_engine_call_wrapped_in_try():
    # the wrap is around the engine.chat / engine.generate awaits specifically
    src = inspect.getsource(R.create_response)
    assert "await run_with_disconnect_guard(request, engine.chat(" in src
    assert "await run_with_disconnect_guard(request, engine.generate(" in src
    # both are now inside a try (the except clause references the isolation var)
    assert "except Exception as _choice_exc:" in src
