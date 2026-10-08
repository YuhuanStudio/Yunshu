"""Template expansion and deterministic, model-free lexical similarity metrics.

Cosine uses token frequencies; METEOR uses exact-token alignment (no downloaded
synonym corpus). BLEU is sentence BLEU with effective order, no smoothing.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher

from .conversations_store import ConversationError

METRICS = {
    "cosine",
    "fuzzy_match",
    "bleu",
    "gleu",
    "meteor",
    "rouge_l",
    *(f"rouge_{i}" for i in range(1, 6)),
}


def lookup(path: str, context: dict):
    value = context
    try:
        for part in path.strip().split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    except (KeyError, IndexError, TypeError, ValueError):
        raise ConversationError(
            400, f"Unknown template reference: {path}", "invalid_value"
        ) from None


def render(value, context: dict):
    if isinstance(value, str):

        def replace(match):
            v = lookup(match.group(1), context)
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

        return re.sub(r"{{\s*([^{}]+?)\s*}}", replace, value)
    if isinstance(value, list):
        return [render(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render(v, context) for k, v in value.items()}
    return value


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def similarity(a: str, b: str, metric: str) -> float:
    x, y = a.split(), b.split()
    if metric == "fuzzy_match":
        return SequenceMatcher(None, a, b, autojunk=False).ratio()
    if not x or not y:
        return 0.0
    if metric == "cosine":
        u, v = Counter(x), Counter(y)
        return sum(c * v[t] for t, c in u.items()) / math.sqrt(
            sum(c * c for c in u.values()) * sum(c * c for c in v.values())
        )
    if metric in ("rouge_l", "meteor"):
        if metric == "rouge_l":
            row = [0] * (len(y) + 1)
            for t in x:
                old = row[:]
                for j, s in enumerate(y, 1):
                    row[j] = old[j - 1] + 1 if t == s else max(old[j], row[j - 1])
            matches = row[-1]
            return 2 * matches / (len(x) + len(y))
        # Greedy exact-token alignment, fragmented matches receive METEOR's penalty.
        unused = set(range(len(y)))
        alignment = []
        for i, t in enumerate(x):
            matched_j = next((j for j in sorted(unused) if y[j] == t), None)
            if matched_j is not None:
                unused.remove(matched_j)
                alignment.append((i, matched_j))
        m = len(alignment)
        if not m:
            return 0.0
        chunks = 1 + sum(
            (b[0] != a[0] + 1 or b[1] != a[1] + 1)
            for a, b in zip(alignment, alignment[1:], strict=False)
        )
        return (10 * m / (len(x) + 9 * len(y))) * (1 - 0.5 * (chunks / m) ** 3)
    if metric.startswith("rouge_"):
        n = int(metric.split("_")[1])
        u, v = ngrams(x, n), ngrams(y, n)
        denom = sum(u.values()) + sum(v.values())
        return 2 * sum((u & v).values()) / denom if denom else 0.0
    orders = range(1, min(4, len(x)) + 1)
    if metric == "gleu":
        u, v = Counter(), Counter()
        for n in range(1, 5):
            u.update(ngrams(x, n))
            v.update(ngrams(y, n))
        return sum((u & v).values()) / max(sum(u.values()), sum(v.values()))
    if metric == "bleu":
        precision = []
        for n in orders:
            u, v = ngrams(x, n), ngrams(y, n)
            p = sum((u & v).values()) / sum(u.values())
            if not p:
                return 0.0
            precision.append(math.log(p))
        return math.exp(min(0, 1 - len(y) / len(x)) + sum(precision) / len(precision))
    raise ValueError(f"Unknown metric {metric}")


def lexical(criterion: dict, context: dict) -> dict:
    g = render(criterion, context)
    a, b = g["input"], g["reference"]
    if g["type"] == "string_check":
        op = g["operation"]
        score = float(
            {
                "eq": a == b,
                "ne": a != b,
                "like": b in a,
                "ilike": b.casefold() in a.casefold(),
            }[op]
        )
    else:
        score = similarity(a, b, g["evaluation_metric"])
    return dict(
        name=g["name"],
        type=g["type"],
        score=score,
        passed=score >= g.get("pass_threshold", 1),
    )
