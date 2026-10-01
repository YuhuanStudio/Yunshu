"""Counting helpers."""


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def count_text(text):
    """Return (lines, words, chars) for ``text``."""
    return text.count("\n"), len(text.split()), len(text)


def count_file(path):
    return count_text(read_text(path))
