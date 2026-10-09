"""Bind llguidance structural-tag byte grammars to actual special token IDs."""

from __future__ import annotations

import json
import re
from typing import Any


def is_lazy_constraint(spec: Any) -> bool:
    return (
        isinstance(spec, dict)
        and spec.get("type") == "cfg"
        and str(spec.get("grammar", "")).startswith("// yunshu structural_tag")
    )


def constrains_initial_output(spec: Any) -> bool:
    return spec is not None and not is_lazy_constraint(spec)


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
    literal_pattern = r'"(?:[^"\\]|\\.)*"'
    trigger_pattern = re.compile(
        r"(tag_\d+)_trig\[lazy\]: TAG_TEXT (" + literal_pattern + ")"
    )
    triggers = []
    additions = []
    for match in list(trigger_pattern.finditer(source)):
        name, trigger = match[1], json.loads(match[2])
        triggers.append(trigger)
        body_match = re.search(rf"^{name}: {name}_trig (.*)$", source, re.M)
        if body_match is None:
            raise ValueError("invalid structural_tag grammar")
        body = body_match[1]
        remainder = re.match(literal_pattern, body)
        suffix = json.loads(remainder[0]) if remainder else ""
        tail = body[remainder.end() :] if remainder else body
        begin = trigger + suffix
        matching = [
            t for t in specials if t.startswith(trigger) and begin.startswith(t)
        ]
        if not matching and any(t in trigger for t in specials):
            raise ValueError(
                "structural_tag trigger mixes special tokens and text; use a text-only or whole special-token trigger"
            )
        for i, token in enumerate(sorted(matching)):
            tid = _token_id(tokenizer, token)
            if tid in eos:
                raise ValueError("structural_tag trigger cannot end generation")
            branch = f"{name}_special_{i}"
            rest = begin[len(token) :]
            additions.append(
                (
                    branch,
                    f"{branch}: TAG_TEXT <[{tid}]> {json.dumps(rest) if rest else ''} {tail}",
                )
            )
    if additions:
        source = source.replace(
            ")* tag_end", " | " + " | ".join(n for n, _ in additions) + ")* tag_end", 1
        )
        source += "\n" + "\n".join(rule for _, rule in additions) + "\n"
    # Keep byte-trigger alternatives as well as actual special-token branches.
    protected = [(m.start(2), m.end(2)) for m in trigger_pattern.finditer(source)]
    for match in re.finditer(r"%json\s+", source):
        _, size = json.JSONDecoder().raw_decode(source[match.end() :])
        protected.append((match.end(), match.end() + size))
    ids_by_text = {t: i for t in specials if (i := _token_id(tokenizer, t)) is not None}
    split_pattern = (
        "("
        + "|".join(re.escape(t) for t in sorted(specials, key=len, reverse=True))
        + ")"
        if specials
        else ""
    )

    def replace_literal(match):
        if any(a <= match.start() < b for a, b in protected) or not split_pattern:
            return match[0]
        token = json.loads(match[0])
        pieces = re.split(split_pattern, token)
        return (
            " ".join(
                f"({json.dumps(p)} | <[{ids_by_text[p]}]>)"
                if p in ids_by_text
                else json.dumps(p)
                for p in pieces
                if p
            )
            or match[0]
        )

    source = re.sub(literal_pattern, replace_literal, source)
    excluded = {
        ids_by_text[t]
        for t in specials
        if any(t.startswith(trigger) for trigger in triggers)
    }
    ids = sorted(set(ids_by_text.values()) - eos - excluded)
    if ids:
        source = source.replace(")* tag_end", " | free_special)* tag_end", 1)
        source += "\nfree_special: TAG_TEXT <[" + ",".join(map(str, ids)) + "]>\n"
    return source
