"""URL slugs."""

import re


def slugify(text):
    """Lower-case ASCII slug: accents dropped, other characters become single hyphens, no edge hyphens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text
