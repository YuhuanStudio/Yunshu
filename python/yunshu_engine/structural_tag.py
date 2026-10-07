"""Bind llguidance structural-tag byte grammars to actual special token IDs."""

from __future__ import annotations

import json
import re
from typing import Any


def bind_structural_tag(source: str, tokenizer: Any, llt: Any) -> str:
    """Reasoning special tokens must remain free outside the constrained tags.

    llguidance text regexes cannot consume HF special tokens. Match these
    explicitly, and use token productions for special trigger/end markers.
    """
    from .tool_call_grammar import _eos_ids, _hf_tokenizer, _token_id

    hf = _hf_tokenizer(tokenizer)
    candidates = set(getattr(hf, "all_special_tokens", []))
    candidates.update(hf.get_added_vocab())
    specials = [
        t
        for t in candidates
        if (i := _token_id(tokenizer, t)) is not None and llt.is_special_token(i)
    ]
    eos = set(_eos_ids(tokenizer))
    trigger_ids = set()
    for match in list(
        re.finditer(r'(tag_\d+_trig)\[lazy\]: TAG_TEXT ("(?:[^"\\]|\\.)*")', source)
    ):
        marker = json.loads(match[2])
        if marker not in specials:
            continue
        tid = _token_id(tokenizer, marker)
        if tid is not None:
            trigger_ids.add(tid)
            source = source.replace(match[0], f"{match[1]}: TAG_TEXT <[{tid}]>")
    # Protect %json schemas: a const string equal to a special marker is
    # still JSON data, never a Lark token production.
    protected = []
    for match in re.finditer(r"%json\s+", source):
        _, size = json.JSONDecoder().raw_decode(source[match.end() :])
        protected.append((match.end(), match.end() + size))
    ids_by_text = {t: _token_id(tokenizer, t) for t in specials}

    def replace_literal(match):
        if any(a <= match.start() < b for a, b in protected):
            return match[0]
        token = json.loads(match[0])
        tid = ids_by_text.get(token)
        return f"<[{tid}]>" if tid is not None else match[0]

    source = re.sub(r'"(?:[^"\\]|\\.)*"', replace_literal, source)
    ids = sorted(
        {i for t in specials if (i := _token_id(tokenizer, t)) is not None}
        - eos
        - trigger_ids
    )
    if ids:
        source = source.replace(")* tag_end", " | free_special)* tag_end", 1)
        source += "\nfree_special: TAG_TEXT <[" + ",".join(map(str, ids)) + "]>\n"
    return source
