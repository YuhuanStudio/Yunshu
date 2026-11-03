"""Mixed-load variant of the session stub (test-only, no GPU, no tokenizer).

Same server as fullcmp_stub_session (same routes, same disk tier under HOME, same usage shapes).
The one difference: a prompt's reported token count is its character count over 4, so a 128K-class
prompt reports a 128K-class count without a real tokenizer. The mixed-load e2e test selects this
module through shim_dir(module="fullcmp_stub_mixed").
"""

from __future__ import annotations

import sys
from pathlib import Path

UNIT_DIR = Path(__file__).resolve().parent
if str(UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_DIR))

import fullcmp_stub_session as session  # noqa: E402

CHARS_PER_TOKEN = 4


def count_long_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt) // CHARS_PER_TOKEN)


def main(argv=None) -> int:
    # The handler looks the counter up at call time, so the override applies to every request.
    session.count_prompt_tokens = count_long_prompt_tokens
    return session.main(argv)


if __name__ == "__main__":
    sys.exit(main())
