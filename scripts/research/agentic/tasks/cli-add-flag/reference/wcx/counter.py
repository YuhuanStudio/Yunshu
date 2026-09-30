"""Counting helpers."""

import re
from collections import Counter

WORD = re.compile(r"[A-Za-z0-9']+")


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def count_text(text):
    """Return (lines, words, chars) for ``text``."""
    return text.count("\n"), len(text.split()), len(text)


def count_file(path):
    return count_text(read_text(path))


def top_words(texts, n, ignore_case=False):
    c = Counter()
    for t in texts:
        c.update(w.lower() if ignore_case else w for w in WORD.findall(t))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
