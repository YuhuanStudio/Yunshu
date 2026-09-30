import pytest

from app.accounts import AccountService
from app.orders import OrderBook
from app.users import UserStore


def test_user_register_normalizes():
    u = UserStore().register(
        "  ada   LOVELACE ", "Ada@Example.com", "+1 (555) 123-4567"
    )
    assert u == {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "15551234567",
    }


def test_order_create():
    o = OrderBook().create("bob smith", "bob@x.io", "5551234567", ["pen"])
    assert o["customer"] == "Bob Smith" and o["id"] == 1


def test_account_open_and_deposit():
    s = AccountService()
    s.open("cy", "cy@x.io", "555-123-4567", deposit=10)
    assert s.deposit("cy@x.io", 5) == 15


def test_bad_email():
    with pytest.raises(ValueError):
        UserStore().register("a", "nope", "5551234567")
