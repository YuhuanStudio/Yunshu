import os
from pathlib import Path

from fleet.billing.invoice import price_invoice
from fleet.carriers import carrier_03, carrier_07
from fleet.pricing import fuel, fuel_v2, handling
from fleet.pricing.weight import freight

TASK = Path(os.environ["AGENTIC_TASK_DIR"])


def test_answer_file():
    text = (TASK / "ANSWER.txt").read_text().strip()
    assert text.replace("\\", "/").lstrip("./") == "fleet/pricing/fuel_v2.py"


def test_active_eu_west_changed():
    assert fuel_v2.FUEL_SURCHARGE["EU-WEST"] == 0.0825
    inv = price_invoice(10, "EU-WEST")
    assert inv["fuel"] == round(freight(10) * 0.0825, 2)


def test_eu_east_and_others_unchanged():
    assert fuel_v2.FUEL_SURCHARGE["EU-EAST"] == 0.075
    assert price_invoice(10, "EU-EAST")["fuel"] == round(freight(10) * 0.075, 2)
    assert len(fuel_v2.FUEL_SURCHARGE) == 24
    assert fuel_v2.fuel_surcharge("MARS") == 0.0


def test_unrelated_075_untouched():
    assert (
        fuel.FUEL_SURCHARGE["EU-WEST"] == 0.075
    )  # legacy table, not imported anywhere
    assert handling.HANDLING_RATE == 0.075
    assert carrier_03.INSURANCE_RATE == 0.075 and carrier_07.INSURANCE_RATE == 0.075


def test_no_other_source_changed():
    import hashlib

    files = sorted((TASK / "fleet").rglob("*.py"))
    assert len(files) >= 30
    changed = [p for p in files if "0.0825" in p.read_text()]
    assert [p.relative_to(TASK).as_posix() for p in changed] == [
        "fleet/pricing/fuel_v2.py"
    ]
    assert hashlib.sha1(b"x").hexdigest()  # keep hashlib import meaningful
