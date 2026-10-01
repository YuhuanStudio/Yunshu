"""Bank-style accounts."""

import re

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)


def _valid_email(value):
    return bool(_EMAIL.match(value or ""))


def _valid_phone(value):
    digits = re.sub(r"\D", "", value or "")
    return 10 <= len(digits) <= 15


def _normalize_name(value):
    return " ".join(part.capitalize() for part in (value or "").split())


class AccountService:
    def __init__(self):
        self._accounts = {}

    def open(self, holder, email, phone, deposit=0):
        if not _valid_email(email):
            raise ValueError(f"invalid email: {email!r}")
        if not _valid_phone(phone):
            raise ValueError(f"invalid phone: {phone!r}")
        if deposit < 0:
            raise ValueError("deposit must not be negative")
        acct = {
            "holder": _normalize_name(holder),
            "email": email.lower(),
            "balance": deposit,
        }
        self._accounts[acct["email"]] = acct
        return dict(acct)

    def balance(self, email):
        return self._accounts[email.lower()]["balance"]

    def deposit(self, email, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._accounts[email.lower()]["balance"] += amount
        return self.balance(email)
