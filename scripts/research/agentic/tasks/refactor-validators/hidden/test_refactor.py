import ast
import os
from pathlib import Path

import pytest

from app.accounts import AccountService
from app.orders import OrderBook
from app.users import UserStore

TASK = Path(os.environ["AGENTIC_TASK_DIR"])
MODULES = ["users", "orders", "accounts"]


def test_validation_module_api():
    from app import validation as v

    assert v.is_valid_email("a@b.co") and v.is_valid_email("A.B+c@sub.example.ORG")
    assert not v.is_valid_email("a@b") and not v.is_valid_email("a b@c.io")
    assert not v.is_valid_email(None) and not v.is_valid_email("")
    assert v.is_valid_phone("+1 (555) 123-4567") and v.is_valid_phone("5551234567")
    assert not v.is_valid_phone("555-1234") and not v.is_valid_phone(None)
    assert not v.is_valid_phone("1" * 16)
    assert v.normalize_name("  ada   LOVELACE ") == "Ada Lovelace"
    assert v.normalize_name(None) == ""


@pytest.mark.parametrize("mod", MODULES)
def test_private_copies_removed(mod):
    src = (TASK / "app" / f"{mod}.py").read_text()
    tree = ast.parse(src)
    defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defs & {"_valid_email", "_valid_phone", "_normalize_name"}
    assert "_EMAIL" not in src and "re.compile" not in src


@pytest.mark.parametrize("mod", MODULES)
def test_module_uses_validation(mod):
    tree = ast.parse((TASK / "app" / f"{mod}.py").read_text())
    used = False
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("validation"):
            used = True
        if (
            isinstance(n, ast.ImportFrom)
            and n.module in (None, "", "app")
            and any(a.name == "validation" for a in n.names)
        ):
            used = True
        if isinstance(n, ast.Import) and any(
            a.name.endswith("validation") for a in n.names
        ):
            used = True
    assert used


def test_users_behaviour():
    s = UserStore()
    u = s.register("  ada   LOVELACE ", "Ada@Example.com", "+1 (555) 123-4567")
    assert u == {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "15551234567",
    }
    assert s.get("ADA@example.com")["name"] == "Ada Lovelace" and len(s) == 1
    with pytest.raises(ValueError, match="already registered"):
        s.register("x", "ada@example.com", "5551234567")
    with pytest.raises(ValueError, match="invalid email"):
        s.register("x", "nope", "5551234567")
    with pytest.raises(ValueError, match="invalid phone"):
        s.register("x", "x@y.io", "123")
    with pytest.raises(ValueError, match="name required"):
        s.register("   ", "x@y.io", "5551234567")


def test_orders_behaviour():
    b = OrderBook()
    o = b.create("bob smith", "Bob@x.io", "5551234567", ["pen"])
    assert o["customer"] == "Bob Smith" and o["id"] == 1 and o["email"] == "bob@x.io"
    assert b.create("a", "a@b.co", "5551234567", ["x"])["id"] == 2
    assert len(b.by_email("BOB@x.io")) == 1
    with pytest.raises(ValueError, match="needs items"):
        b.create("a", "a@b.co", "5551234567", [])
    with pytest.raises(ValueError, match="invalid phone"):
        b.create("a", "a@b.co", "12", ["x"])


def test_accounts_behaviour():
    s = AccountService()
    a = s.open("cy  dee", "Cy@x.io", "555-123-4567", deposit=10)
    assert a["holder"] == "Cy Dee" and a["balance"] == 10
    assert s.deposit("cy@x.io", 5) == 15
    with pytest.raises(ValueError, match="invalid email"):
        s.open("a", "bad", "5551234567")
    with pytest.raises(ValueError, match="negative"):
        s.open("a", "a@b.co", "5551234567", deposit=-1)
    with pytest.raises(ValueError, match="positive"):
        s.deposit("cy@x.io", 0)
