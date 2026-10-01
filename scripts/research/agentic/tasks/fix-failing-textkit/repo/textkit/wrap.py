"""Greedy word wrapping."""


def wrap(text, width):
    """Split ``text`` into lines of at most ``width`` characters, breaking at whitespace.

    Runs of whitespace collapse to one space. A word longer than ``width`` gets a line of its own.
    """
    if width < 1:
        raise ValueError("width must be positive")
    lines = []
    line = ""
    for word in text.split():
        if not line:
            line = word
        elif len(line) + 1 + len(word) >= width:
            lines.append(line)
            line = word
        else:
            line += " " + word
    if line:
        lines.append(line)
    return lines
