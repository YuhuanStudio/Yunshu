"""Discount codes and price arithmetic (all money is integer cents)."""

CODES = {
    "SAVE10": ("percent", 10),
    "SAVE25": ("percent", 25),
    "FIVEOFF": ("fixed", 500),
}


def apply_discount(total_cents, code):
    """Return the total after applying ``code``; unknown or empty codes change nothing."""
    if not code or code not in CODES:
        return total_cents
    kind, value = CODES[code]
    if kind == "percent":
        return (total_cents * (100 - value) + 50) // 100
    return max(0, total_cents - value)
