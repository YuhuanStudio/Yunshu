"""User registry."""

import re

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)


def _valid_email(value):
    return bool(_EMAIL.match(value or ""))


def _valid_phone(value):
    digits = re.sub(r"\D", "", value or "")
    return 10 <= len(digits) <= 15


def _normalize_name(value):
    return " ".join(part.capitalize() for part in (value or "").split())


class UserStore:
    def __init__(self):
        self._users = {}

    def register(self, name, email, phone):
        if not _valid_email(email):
            raise ValueError(f"invalid email: {email!r}")
        if not _valid_phone(phone):
            raise ValueError(f"invalid phone: {phone!r}")
        name = _normalize_name(name)
        if not name:
            raise ValueError("name required")
        key = email.lower()
        if key in self._users:
            raise ValueError("already registered")
        self._users[key] = {
            "name": name,
            "email": key,
            "phone": re.sub(r"\D", "", phone),
        }
        return dict(self._users[key])

    def get(self, email):
        return self._users.get(email.lower())

    def __len__(self):
        return len(self._users)
