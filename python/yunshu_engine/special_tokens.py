"""Keep special-token strings typed by a user (or returned by a tool) from becoming control tokens.

The chat template renders every message into one string and the tokenizer then turns each
special-token string in it (``<|im_end|>``, ``<|im_start|>``, ``<|endoftext|>``...) into the real
control id, so message text such as ``<|im_end|>\\n<|im_start|>system\\n...`` could close the user's
turn and open a system one. ``split_special_tokens`` cannot be used on the rendered string: it
would also demote the template's own structural tokens.

So the text is changed before rendering: a zero-width space goes after the first character of each
special-token string found in message *content* (``<|im_end|>`` -> ``<U+200B>|im_end|>``), which
then tokenizes as ordinary text. Properties this keeps:

* text without a special-token string is returned untouched (same object), so the tokenization of
  normal text, and every cached prefix, is exactly as before;
* the change is a pure function of the text, so the same conversation renders and tokenizes
  identically on every turn (APC prefix identity holds);
* the template's own specials, the roles and the media/tool structure are not content and are
  never touched.

The cost is that the model sees a zero-width character inside such a string.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

ZWSP = "​"
# Values that are structure, identifiers or media payloads rather than text the model reads.
_SKIP_KEYS = frozenset({"role", "type", "url", "data"})

_cache: dict[int, tuple[Any, re.Pattern | None]] = {}


def _special_strings(tokenizer: Any) -> list[str]:
    found: set[str] = set()
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(decoder, dict):
        for t in decoder.values():
            if getattr(t, "special", False) and isinstance(
                getattr(t, "content", None), str
            ):
                found.add(t.content)
    with contextlib.suppress(Exception):
        found.update(x for x in tokenizer.all_special_tokens if isinstance(x, str))
    # one character cannot be told apart from text
    return [s for s in found if len(s) >= 2]


def _pattern(tokenizer: Any) -> re.Pattern | None:
    hit = _cache.get(id(tokenizer))
    if hit is not None and hit[0] is tokenizer:
        return hit[1]
    specials = sorted(_special_strings(tokenizer), key=len, reverse=True)
    pat = re.compile("|".join(re.escape(s) for s in specials)) if specials else None
    if len(_cache) > 16:
        _cache.clear()
    _cache[id(tokenizer)] = (tokenizer, pat)
    return pat


def _fix(value: Any, pat: re.Pattern) -> Any:
    """``value`` with special-token strings defused; the same object when there are none."""
    if isinstance(value, str):
        if not pat.search(value):
            return value
        return pat.sub(lambda m: m.group(0)[0] + ZWSP + m.group(0)[1:], value)
    if isinstance(value, list):
        new = [_fix(v, pat) for v in value]
        return value if all(a is b for a, b in zip(new, value, strict=True)) else new
    if isinstance(value, dict):
        new_d = {k: (v if k in _SKIP_KEYS else _fix(v, pat)) for k, v in value.items()}
        same = all(new_d[k] is value[k] for k in value)
        return value if same else new_d
    return value


def neutralize_special_text(messages: list[dict], tokenizer: Any) -> list[dict]:
    """``messages`` with special-token strings in their content defused (not mutated; the same
    list when nothing needed changing)."""
    if tokenizer is None or not messages:
        return messages
    pat = _pattern(tokenizer)
    if pat is None:
        return messages
    out = [_fix(m, pat) if isinstance(m, dict) else m for m in messages]
    if all(a is b for a, b in zip(out, messages, strict=True)):
        return messages
    return out
