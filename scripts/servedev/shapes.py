"""Request SHAPES only: numbers and flags, never prompt or completion content (user rule, 2026-10-02).

Input is the numbers-only serve log (``yunshu_gateway.serve_log``). A record that carries any content-like key
is rejected, not filtered: a leak upstream must be loud.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

FORBIDDEN = {
    "prompt",
    "messages",
    "content",
    "text",
    "completion",
    "output",
    "input",
    "tool_calls",
    "token_ids",
    "ids",
}
CTX_EDGES = (1024, 4096, 16384, 32768, 65536, 131072)


@dataclass(frozen=True)
class Shape:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    tools: bool = False

    @property
    def bucket(self) -> int:
        return next(
            (e for e in CTX_EDGES if self.prompt_tokens <= e), CTX_EDGES[-1] * 2
        )


def from_event(ev: dict) -> Shape | None:
    """Shape of one finished request, None when incomplete (cancelled / missing counts)."""
    leaked = FORBIDDEN & set(ev)
    if leaked:
        raise ValueError(
            f"serve-log record carries content-like keys {sorted(leaked)}: refusing to build shapes"
        )
    if (
        ev.get("cancelled")
        or ev.get("prompt_tokens") is None
        or not ev.get("completion_tokens")
    ):
        return None
    return Shape(
        int(ev["prompt_tokens"]),
        int(ev["completion_tokens"]),
        int(ev.get("cached_tokens") or 0),
        bool(ev.get("tools")),
    )


def load(events: list[dict]) -> list[Shape]:
    return [s for s in map(from_event, events) if s]


def sample(shapes: list[Shape], n: int, seed: int = 0) -> list[Shape]:
    """Deterministic, stratified by context bucket (long requests are not drowned by short ones)."""
    if not shapes:
        raise ValueError("no shapes to sample")
    by: dict[int, list[Shape]] = {}
    for s in shapes:
        by.setdefault(s.bucket, []).append(s)
    rng = random.Random(seed)
    out: list[Shape] = []
    keys = sorted(by)
    i = 0
    while len(out) < n:
        out.append(rng.choice(by[keys[i % len(keys)]]))
        i += 1
    return out


_WORDS = (
    "alpha",
    "river",
    "stone",
    "window",
    "forest",
    "signal",
    "market",
    "planet",
    "bridge",
    "garden",
    "engine",
    "violet",
)


def synthetic_text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(
        rng.choice(_WORDS) + str(rng.randrange(100)) for _ in range(n_words)
    )


def synthetic_messages(
    shape: Shape, seed: int, tokens_per_word: float = 2.0
) -> list[dict]:
    """Same-shape stand-in: ~prompt_tokens of seeded filler. Cached share = shared prefix with seed 0 of the family.

    Synthetic text cannot establish real-content parity; replay therefore reports itself as shape-level only.
    """
    words = max(1, int(shape.prompt_tokens / tokens_per_word))
    shared = int(words * min(1.0, shape.cached_tokens / max(shape.prompt_tokens, 1)))
    text = synthetic_text(shared, 0) + " " + synthetic_text(words - shared, seed)
    return [
        {"role": "user", "content": text.strip() + "\nSummarize the above in one line."}
    ]
