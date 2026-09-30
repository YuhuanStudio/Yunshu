"""User registry."""

import re

from .validation import is_valid_email, is_valid_phone, normalize_name


class UserStore:
    def __init__(self):
        self._users = {}

    def register(self, name, email, phone):
        if not is_valid_email(email):
            raise ValueError(f"invalid email: {email!r}")
        if not is_valid_phone(phone):
            raise ValueError(f"invalid phone: {phone!r}")
        name = normalize_name(name)
        if not name:
            raise ValueError("name required")
        key = email.lower()
        if key in self._users:
            raise ValueError("already registered")
        self._users[key] = {"name": name, "email": key, "phone": "".join(c for c in phone if c.isdigit())}
        return dict(self._users[key])

    def get(self, email):
        return self._users.get(email.lower())

    def __len__(self):
        return len(self._users)
