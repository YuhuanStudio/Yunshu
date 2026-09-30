"""Bank-style accounts."""

from .validation import is_valid_email, is_valid_phone, normalize_name


class AccountService:
    def __init__(self):
        self._accounts = {}

    def open(self, holder, email, phone, deposit=0):
        if not is_valid_email(email):
            raise ValueError(f"invalid email: {email!r}")
        if not is_valid_phone(phone):
            raise ValueError(f"invalid phone: {phone!r}")
        if deposit < 0:
            raise ValueError("deposit must not be negative")
        acct = {"holder": normalize_name(holder), "email": email.lower(), "balance": deposit}
        self._accounts[acct["email"]] = acct
        return dict(acct)

    def balance(self, email):
        return self._accounts[email.lower()]["balance"]

    def deposit(self, email, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._accounts[email.lower()]["balance"] += amount
        return self.balance(email)
