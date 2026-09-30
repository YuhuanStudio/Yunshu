"""Order book."""

import itertools

from .validation import is_valid_email, is_valid_phone, normalize_name


class OrderBook:
    def __init__(self):
        self._orders = []
        self._ids = itertools.count(1)

    def create(self, customer, email, phone, items):
        if not is_valid_email(email):
            raise ValueError(f"invalid email: {email!r}")
        if not is_valid_phone(phone):
            raise ValueError(f"invalid phone: {phone!r}")
        if not items:
            raise ValueError("an order needs items")
        order = {
            "id": next(self._ids),
            "customer": normalize_name(customer),
            "email": email.lower(),
            "phone": "".join(c for c in phone if c.isdigit()),
            "items": list(items),
        }
        self._orders.append(order)
        return dict(order)

    def by_email(self, email):
        return [o for o in self._orders if o["email"] == email.lower()]
