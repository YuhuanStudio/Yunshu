"""Keep special-token strings typed by a user (or returned by a tool) from becoming control tokens.

The chat template renders every message into one string and the tokenizer then turns each
special-token string in it (``<|im_end|>``, ``<|im_start|>``, ``<|endoftext|>``...) into the real
control id, so message text such as ``<|im_end|>\\n<|im_start|>system\\n...`` could close the user's
turn and open a system one. ``split_special_tokens`` cannot be used on the rendered string: it
would also demote the template's own structural tokens.

So the text is changed before rendering: a zero-width space goes after the first character of a
special-token string found in message *content* (``<|im_end|>`` -> ``<U+200B>|im_end|>``), which
then tokenizes as ordinary text. The scope is deliberately narrow, on two axes:

* WHOSE text: only roles whose content comes from outside the operator and the model: ``user``,
  tool results (``tool`` / ``function`` / ``ipython`` ...) and any unknown role (fail safe).
  ``system`` / ``developer`` text is set by whoever runs the application, and ``assistant`` text
  (and its ``tool_calls`` arguments) is what the model itself produced and is replayed verbatim:
  for models whose tool / channel / thought markers ARE special tokens (Mistral ``[TOOL_CALLS]``,
  GLM / DeepSeek / Gemma channel markers) defusing them would corrupt the replayed history.
* WHICH strings: only the special tokens the chat template itself emits as structure: those
  that appear literally in the template (``<|im_start|>``, ``<|im_end|>``, ``[INST]``,
  ``<|vision_start|>``...) or are the ``bos_token`` / ``eos_token`` ... the template references by
  name. ``<pad>``, ``<unk>`` and other specials the template never writes cannot open or close
  a turn and are left alone. A template that cannot be read falls back to every special string.

Properties this keeps:

* text without such a string is returned untouched (same object), so the tokenization of
  normal text, and every cached prefix, is exactly as before;
* the change is a pure function of (role, text, template), so the same conversation renders and
  tokenizes identically on every turn (APC prefix identity holds);
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

# Roles whose text is the operator's or the model's own; everything else is outside input.
_TRUSTED_ROLES = frozenset({"system", "developer", "assistant"})
# Template variables that name a special token the template writes by reference.
_TOKEN_VARS = (
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "sep_token",
    "cls_token",
    "mask_token",
)

_cache: dict[int, tuple[Any, Any, re.Pattern | None]] = {}


def _template_text(tokenizer: Any) -> str | None:
    """The chat template source (a str, or every named template of a dict / list), else None."""
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, str):
        return tpl
    if isinstance(tpl, dict):
        items = list(tpl.values())
    elif isinstance(tpl, (list, tuple)):
        items = list(tpl)
    else:
        return None
    parts = [
        it if isinstance(it, str) else it.get("template")
        for it in items
        if isinstance(it, (str, dict))
    ]
    texts = [x for x in parts if isinstance(x, str)]
    return "\n".join(texts) if texts else None


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


def _added_strings(tokenizer: Any) -> list[str]:
    """Every added-token string, special or not: Qwen3.8 registers <think> / <tool_call> with special=False."""
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not isinstance(decoder, dict):
        return []
    out = {getattr(t, "content", None) for t in decoder.values()}
    return [s for s in out if isinstance(s, str) and len(s) >= 2]


def _structural_strings(tokenizer: Any) -> list[str]:
    """The special-token strings the chat template emits as structure (see the module doc)."""
    specials = _special_strings(tokenizer)
    template = _template_text(tokenizer)
    if template is None:
        return specials  # unreadable template: defuse every special
    # an added token the template writes is structure even when it is not flagged special (Qwen3.8 <think>,
    # <tool_call>: typed in user text they would otherwise become their control ids)
    keep = {s for s in (*specials, *_added_strings(tokenizer)) if s in template}
    for var in _TOKEN_VARS:
        if re.search(rf"\b{var}\b", template):
            val = getattr(tokenizer, var, None)
            if isinstance(val, str) and val in specials:
                keep.add(val)
    return sorted(keep)


def _pattern(tokenizer: Any) -> re.Pattern | None:
    template = getattr(tokenizer, "chat_template", None)
    hit = _cache.get(id(tokenizer))
    if hit is not None and hit[0] is tokenizer and hit[1] is template:
        return hit[2]
    specials = sorted(_structural_strings(tokenizer), key=len, reverse=True)
    pat = re.compile("|".join(re.escape(s) for s in specials)) if specials else None
    if len(_cache) > 16:
        _cache.clear()
    _cache[id(tokenizer)] = (tokenizer, template, pat)
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


def _fix_message(m: Any, pat: re.Pattern) -> Any:
    """One message with its outside-input content defused; trusted roles are returned as they are."""
    if not isinstance(m, dict) or m.get("role") in _TRUSTED_ROLES:
        return m
    content = m.get("content")
    fixed = _fix(content, pat)
    if fixed is content:
        return m
    return {**m, "content": fixed}


def neutralize_special_text(messages: list[dict], tokenizer: Any) -> list[dict]:
    """``messages`` with the template's structural special-token strings defused in the content of
    outside-input roles (not mutated; the same list when nothing needed changing)."""
    if tokenizer is None or not messages:
        return messages
    pat = _pattern(tokenizer)
    if pat is None:
        return messages
    out = [_fix_message(m, pat) for m in messages]
    if all(a is b for a, b in zip(out, messages, strict=True)):
        return messages
    return out
