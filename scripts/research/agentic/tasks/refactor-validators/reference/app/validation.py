"""Shared input validation."""

import re

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)


def is_valid_email(value):
    return bool(_EMAIL.match(value or ""))


def is_valid_phone(value):
    digits = re.sub(r"\D", "", value or "")
    return 10 <= len(digits) <= 15


def normalize_name(value):
    return " ".join(part.capitalize() for part in (value or "").split())
