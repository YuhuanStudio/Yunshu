"""EngineCore loop: message text with a special-token string stays text (its own template renderer
called the tokenizer directly and skipped the neutralisation the other text routes apply)."""

from __future__ import annotations

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tests.unit.test_control_token_injection import EVIL, MSGS, _ids, _tok
from yunshu_engine.engine_core import EngineCore


def test_messages_to_text_does_not_turn_user_text_into_control_tokens():
    tok = _tok()
    start, end = _ids(tok)
    core = EngineCore.__new__(EngineCore)
    core._tokenizer = tok
    text = core._messages_to_text(MSGS)
    ids = tok.encode(text, add_special_tokens=False)
    assert ids.count(start) == len(MSGS) + 1, "user text opened a turn"
    assert ids.count(end) == len(MSGS), "user text closed a turn"
    assert EVIL not in text
