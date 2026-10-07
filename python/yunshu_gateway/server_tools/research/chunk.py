"""Heading-aware exact spans: offsets always refer to the extracted page text."""

import re

from ..search import Passage


def chunks(text: str, size: int = 1200, overlap: int = 240) -> list[Passage]:
    headings: list[str] = []
    sections: list[tuple[int, int, str]] = []
    start = 0
    path = ""
    for match in re.finditer(r"(?m)^(#{1,6})\s+(.+)$", text):
        if match.start() > start:
            sections.append((start, match.start(), path))
        level = len(match[1])
        headings = headings[: level - 1] + [match[2]]
        path = " / ".join(headings)
        start = match.start()
    sections.append((start, len(text), path))
    out = []
    for start, end, heading in sections:
        while start < end:
            stop = min(end, start + size)
            if stop < end:
                boundary = text.rfind(" ", start + size // 2, stop)
                if boundary > start:
                    stop = boundary
            left = start
            while left < stop and text[left].isspace():
                left += 1
            right = stop
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                out.append(
                    Passage(text[left:right], heading=heading, start=left, end=right)
                )
            if stop == end:
                break
            start = max(start + 1, stop - overlap)
    return out
