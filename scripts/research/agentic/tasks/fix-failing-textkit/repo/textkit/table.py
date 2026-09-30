"""Plain-text tables."""


def render(rows, headers):
    """Render rows under headers; numbers are right-aligned, everything else left-aligned."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    out = []
    for n, row in enumerate(cells):
        parts = []
        for i, c in enumerate(row):
            numeric = n > 0 and isinstance(rows[n - 1][i], (int, float))
            parts.append(c.rjust(widths[i]) if numeric else c.ljust(widths[i]))
        out.append("  ".join(parts).rstrip())
        if n == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)
