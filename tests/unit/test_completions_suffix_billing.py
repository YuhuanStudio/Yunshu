"""(MED): completions stop-correction stripped req.suffix from the
ACTUAL completion → completion_tokens (ct) undercount (billing bug).

Since the OpenAI `suffix` (FIM context) is NEVER appended to `text` (this
engine has no FIM template). The stop-sequence overcount-correction block, however,
still chopped `len(req.suffix)` chars off `_raw` BEFORE searching for the stop
string — truncating real generated content, so the recomputed `ct` was wrong.
Output text was unaffected (the return uses `text`, not `_raw`). Fix: remove the
suffix-truncation lines so the correction operates on the true completion.

Source-level guard test (the bug lived in a deep per-request closure that's hard
to exercise without a live engine).
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import completions


def test_stop_correction_does_not_truncate_by_suffix():
    src = inspect.getsource(completions)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the stale suffix-truncation of _raw is gone
    assert "_raw[:-len(req.suffix)]" not in code
    # the correction still runs on the real completion: stop search over _raw
    assert "_raw.find(_seq)" in code
