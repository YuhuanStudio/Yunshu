"""Greedy word wrapping."""


def wrap(text, width):
    if width < 1:
        raise ValueError("width must be positive")
    lines = []
    line = ""
    for word in text.split():
        if not line:
            line = word
        elif len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line += " " + word
    if line:
        lines.append(line)
    return lines
