"""Order book."""

import itertools
import re

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)


def _valid_email(value):
    return bool(_EMAIL.match(value or ""))


def _valid_phone(value):
    digits = re.sub(r"\D", "", value or "")
    return 10 <= len(digits) <= 15


def _normalize_name(value):
    return " ".join(part.capitalize() for part in (value or "").split())


class OrderBook:
    def __init__(self):
        self._orders = []
        self._ids = itertools.count(1)

    def create(self, customer, email, phone, items):
        if not _valid_email(email):
            raise ValueError(f"invalid email: {email!r}")
        if not _valid_phone(phone):
            raise ValueError(f"invalid phone: {phone!r}")
        if not items:
            raise ValueError("an order needs items")
        order = {
            "id": next(self._ids),
            "customer": _normalize_name(customer),
            "email": email.lower(),
            "phone": re.sub(r"\D", "", phone),
            "items": list(items),
        }
        self._orders.append(order)
        return dict(order)

    def by_email(self, email):
        return [o for o in self._orders if o["email"] == email.lower()]
