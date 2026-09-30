"""Deterministically generate the ~2,100-line `fleet` codebase for the long-context task.

    python gen.py OUTDIR

The active fuel-surcharge table is fleet/pricing/fuel_v2.py (fleet/billing/invoice.py imports it).
fleet/pricing/fuel.py is a legacy copy nothing imports, and several carrier modules carry other
rates that happen to be 0.075 -- naive search-and-replace breaks things.
"""

import random
import sys
from pathlib import Path

out = Path(sys.argv[1])
rnd = random.Random(20240607)
files: dict[str, str] = {}

REGIONS = [
    "EU-WEST",
    "EU-EAST",
    "EU-NORTH",
    "EU-SOUTH",
    "US-EAST",
    "US-WEST",
    "US-CENTRAL",
    "CA-EAST",
    "CA-WEST",
    "MX",
    "BR-SOUTH",
    "BR-NORTH",
    "AR",
    "UK",
    "IE",
    "JP",
    "KR",
    "CN-EAST",
    "CN-SOUTH",
    "IN-WEST",
    "IN-SOUTH",
    "AU-EAST",
    "AU-WEST",
    "NZ",
]

files["fleet/__init__.py"] = '"""fleet: shipment quoting and invoicing."""\n'

# ---- regions ---------------------------------------------------------------------------------
lines = [
    '"""Region catalogue: code, display name, tax rate, customs class."""',
    "",
    "REGIONS = {",
]
for r in REGIONS:
    lines.append(
        f'    "{r}": {{"name": "{r.title()}", "tax": {rnd.choice([0.0, 0.05, 0.07, 0.19, 0.2, 0.21]):.2f}, '
        f'"customs": "{rnd.choice("ABC")}"}},'
    )
lines += [
    "}",
    "",
    "",
    "def known(region):",
    "    return region in REGIONS",
    "",
    "",
    "def tax_rate(region):",
    '    return REGIONS[region]["tax"]',
    "",
    "",
    "def customs_class(region):",
    '    return REGIONS[region]["customs"]',
    "",
]
files["fleet/regions.py"] = "\n".join(lines)

# ---- pricing ---------------------------------------------------------------------------------
files["fleet/pricing/__init__.py"] = ""


def fuel_table(active: bool):
    rows = []
    for r in REGIONS:
        if r in ("EU-WEST", "EU-EAST"):
            v = 0.075
        else:
            v = round(rnd.choice([0.03, 0.04, 0.05, 0.06, 0.065, 0.07, 0.08, 0.09]), 3)
        rows.append(f'    "{r}": {v},')
    return "\n".join(rows)


files["fleet/pricing/fuel.py"] = (
    '"""LEGACY fuel surcharge table (kept for the 2019 reconciliation report; no longer imported)."""\n\n'
    "FUEL_SURCHARGE = {\n" + fuel_table(False) + "\n}\n\n\n"
    "def fuel_surcharge(region):\n"
    "    return FUEL_SURCHARGE.get(region, 0.0)\n"
)
files["fleet/pricing/fuel_v2.py"] = (
    '"""Fuel surcharge as a fraction of the base freight charge, by region (active table)."""\n\n'
    "FUEL_SURCHARGE = {\n" + fuel_table(True) + "\n}\n\n"
    "DEFAULT = 0.0\n\n\n"
    "def fuel_surcharge(region):\n"
    "    return FUEL_SURCHARGE.get(region, DEFAULT)\n"
)
files["fleet/pricing/weight.py"] = '''"""Weight bands and chargeable weight."""

import math

BANDS = [(1, 4.0), (5, 3.5), (20, 3.0), (100, 2.5), (1000, 2.0)]


def volumetric(l_cm, w_cm, h_cm):
    return math.ceil(l_cm * w_cm * h_cm / 5000)


def chargeable(weight_kg, l_cm=0, w_cm=0, h_cm=0):
    return max(math.ceil(weight_kg), volumetric(l_cm, w_cm, h_cm))


def per_kg(weight_kg):
    for limit, rate in BANDS:
        if weight_kg <= limit:
            return rate
    return BANDS[-1][1]


def freight(weight_kg):
    return round(chargeable(weight_kg) * per_kg(weight_kg), 2)
'''
files["fleet/pricing/handling.py"] = '''"""Handling fees per customs class."""

HANDLING = {"A": 2.0, "B": 3.5, "C": 6.0}
HANDLING_RATE = 0.075  # share of freight charged as handling for class C (unrelated to fuel)


def handling_fee(customs_class, freight_charge):
    fee = HANDLING[customs_class]
    if customs_class == "C":
        fee += freight_charge * HANDLING_RATE
    return round(fee, 2)
'''

# ---- carriers --------------------------------------------------------------------------------
files["fleet/carriers/__init__.py"] = (
    '"""Carrier plug-ins."""\n\nfrom . import '
    + ", ".join(f"carrier_{i:02d}" for i in range(12))
    + "\n\nALL = {\n"
    + "".join(f'    "{i:02d}": carrier_{i:02d}.Carrier,\n' for i in range(12))
    + "}\n"
)
for i in range(12):
    base = round(rnd.uniform(1.0, 9.0), 2)
    ins = (
        0.075 if i in (3, 7) else round(rnd.choice([0.01, 0.015, 0.02, 0.025, 0.03]), 3)
    )
    slots = rnd.randint(2, 6)
    max_kg = rnd.choice([30, 50, 70, 150, 500])
    zones = rnd.sample(REGIONS, 8)
    lines = [
        f'"""Carrier {i:02d}: service levels, surcharges and transit times."""',
        "",
        f"NAME = 'carrier-{i:02d}'",
        f"BASE_FEE = {base}",
        f"INSURANCE_RATE = {ins}  # fraction of declared value",
        f"MAX_KG = {max_kg}",
        f"PICKUP_SLOTS = {slots}",
        f"ZONES = {zones!r}",
        "TRANSIT_DAYS = {",
    ]
    for z in zones:
        lines.append(f'    "{z}": {rnd.randint(1, 9)},')
    lines += [
        "}",
        "",
        "",
        "class Carrier:",
        "    name = NAME",
        "",
        "    def serves(self, region):",
        "        return region in ZONES",
        "",
        "    def accepts(self, weight_kg):",
        "        return 0 < weight_kg <= MAX_KG",
        "",
        "    def transit_days(self, region):",
        "        if region not in TRANSIT_DAYS:",
        '            raise KeyError(f"{NAME} does not serve {region}")',
        "        return TRANSIT_DAYS[region]",
        "",
        "    def base_fee(self):",
        "        return BASE_FEE",
        "",
        "    def insurance(self, declared_value):",
        "        return round(declared_value * INSURANCE_RATE, 2)",
        "",
        "    def pickup_slot(self, n):",
        "        return n % PICKUP_SLOTS",
        "",
        "    def describe(self):",
        '        return f"{NAME}: base {BASE_FEE}, up to {MAX_KG} kg, {len(ZONES)} zones"',
        "",
    ]
    for k in range(rnd.randint(2, 4)):
        c = round(rnd.uniform(0.5, 3.0), 2)
        lines += [
            "",
            f"    def option_{k}(self, weight_kg):",
            f'        """Optional service #{k} priced per kg."""',
            f"        return round(weight_kg * {c}, 2)",
        ]
    files[f"fleet/carriers/carrier_{i:02d}.py"] = "\n".join(lines) + "\n"

# ---- utilities (filler with real, tested behaviour) -------------------------------------------
files["fleet/util/__init__.py"] = ""
files["fleet/util/dates.py"] = '''"""Business-day arithmetic."""

import datetime as dt

HOLIDAYS = {dt.date(2024, 12, 25), dt.date(2025, 1, 1)}


def is_business_day(d):
    return d.weekday() < 5 and d not in HOLIDAYS


def add_business_days(d, n):
    step = 1 if n >= 0 else -1
    left = abs(n)
    while left:
        d += dt.timedelta(days=step)
        if is_business_day(d):
            left -= 1
    return d


def business_days_between(a, b):
    if a > b:
        a, b = b, a
    n = 0
    d = a
    while d < b:
        d += dt.timedelta(days=1)
        if is_business_day(d):
            n += 1
    return n
'''
files["fleet/util/money.py"] = '''"""Money helpers (floats rounded to cents)."""


def cents(x):
    return int(round(x * 100))


def fmt(x):
    return f"{x:,.2f}"


def split(total, parts):
    base = cents(total) // parts
    rest = cents(total) - base * parts
    return [(base + (1 if i < rest else 0)) / 100 for i in range(parts)]
'''
files["fleet/util/text.py"] = '''"""Small text helpers used in labels and reports."""


def pad(s, n):
    return s + " " * max(0, n - len(s))


def title(s):
    return " ".join(w.capitalize() for w in s.replace("-", " ").split())


def truncate(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def table(rows, headers):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    out = ["  ".join(pad(str(h), w) for h, w in zip(headers, widths))]
    for r in rows:
        out.append("  ".join(pad(str(c), w) for c, w in zip(r, widths)))
    return "\\n".join(out)
'''

# ---- domain modules (each ~60 lines of small functions) --------------------------------------
for name, fields in [
    ("customers", ["name", "email", "tier"]),
    ("shipments", ["origin", "destination", "weight_kg"]),
    ("parcels", ["length_cm", "width_cm", "height_cm"]),
    ("tracking", ["status", "location", "note"]),
    ("returns", ["reason", "refund", "approved"]),
    ("warehouses", ["city", "capacity", "open"]),
    ("drivers", ["licence", "region", "hours"]),
    ("routes", ["stops", "km", "vehicle"]),
]:
    cls = name[:-1].capitalize() if name.endswith("s") else name.capitalize()
    lines = [
        f'"""{name.capitalize()} records and a tiny in-memory registry."""',
        "",
        "",
        f"class {cls}:",
    ]
    lines.append("    def __init__(self, ident, " + ", ".join(fields) + "):")
    lines.append("        self.ident = ident")
    for f in fields:
        lines.append(f"        self.{f} = {f}")
    lines += ["", "    def as_dict(self):", "        return {"]
    lines.append('            "ident": self.ident,')
    for f in fields:
        lines.append(f'            "{f}": self.{f},')
    lines += [
        "        }",
        "",
        "",
        "class Registry:",
        "    def __init__(self):",
        "        self._items = {}",
        "",
    ]
    lines += [
        "    def add(self, item):",
        "        if item.ident in self._items:",
        '            raise ValueError(f"duplicate {item.ident}")',
        "        self._items[item.ident] = item",
        "        return item",
        "",
        "    def get(self, ident):",
        "        return self._items[ident]",
        "",
        "    def remove(self, ident):",
        "        return self._items.pop(ident)",
        "",
        "    def __len__(self):",
        "        return len(self._items)",
        "",
        "    def all(self):",
        "        return [self._items[k] for k in sorted(self._items)]",
        "",
        "    def find(self, **kw):",
        "        return [i for i in self.all() if all(getattr(i, k) == v for k, v in kw.items())]",
        "",
    ]
    for k in range(4):
        lines += [
            "",
            f"def summary_{k}(registry):",
            f'    """Summary #{k}: count and a checksum of idents."""',
            "    items = registry.all()",
            f"    return len(items), sum(len(str(i.ident)) * {k + 1} for i in items)",
            "",
        ]
    files[f"fleet/{name}.py"] = "\n".join(lines) + "\n"

# ---- billing ---------------------------------------------------------------------------------
files["fleet/billing/__init__.py"] = ""
files[
    "fleet/billing/invoice.py"
] = '''"""Price an invoice for a shipment: freight + fuel surcharge + handling + tax."""

from ..pricing.fuel_v2 import fuel_surcharge
from ..pricing.handling import handling_fee
from ..pricing.weight import freight
from ..regions import customs_class, known, tax_rate


def price_invoice(weight_kg, region):
    if not known(region):
        raise ValueError(f"unknown region {region!r}")
    base = freight(weight_kg)
    fuel = round(base * fuel_surcharge(region), 2)
    handling = handling_fee(customs_class(region), base)
    net = round(base + fuel + handling, 2)
    tax = round(net * tax_rate(region), 2)
    return {
        "region": region,
        "freight": base,
        "fuel": fuel,
        "handling": handling,
        "net": net,
        "tax": tax,
        "total": round(net + tax, 2),
    }
'''
files["fleet/billing/reports.py"] = '''"""Reports over many invoices."""

from .invoice import price_invoice


def batch(shipments):
    return [price_invoice(w, r) for w, r in shipments]


def totals_by_region(shipments):
    out = {}
    for inv in batch(shipments):
        out[inv["region"]] = round(out.get(inv["region"], 0) + inv["total"], 2)
    return out


def grand_total(shipments):
    return round(sum(i["total"] for i in batch(shipments)), 2)
'''

# ---- visible tests ---------------------------------------------------------------------------
files["tests/test_fleet.py"] = """import datetime as dt

import pytest

from fleet.billing.invoice import price_invoice
from fleet.billing.reports import grand_total, totals_by_region
from fleet.carriers import ALL
from fleet.pricing.weight import chargeable, freight, per_kg
from fleet.util.dates import add_business_days, business_days_between
from fleet.util.money import cents, split


def test_freight_bands():
    assert per_kg(0.5) == 4.0 and per_kg(20) == 3.0 and per_kg(5000) == 2.0
    assert chargeable(2.2) == 3
    assert freight(10) == 30.0


def test_invoice_shape():
    inv = price_invoice(10, "US-EAST")
    assert set(inv) == {"region", "freight", "fuel", "handling", "net", "tax", "total"}
    assert inv["total"] == round(inv["net"] + inv["tax"], 2)


def test_unknown_region():
    with pytest.raises(ValueError):
        price_invoice(1, "MARS")


def test_reports():
    ship = [(10, "UK"), (3, "UK"), (7, "JP")]
    assert set(totals_by_region(ship)) == {"UK", "JP"}
    assert grand_total(ship) == round(sum(totals_by_region(ship).values()), 2)


def test_business_days():
    assert add_business_days(dt.date(2024, 6, 7), 1) == dt.date(2024, 6, 10)
    assert business_days_between(dt.date(2024, 6, 3), dt.date(2024, 6, 10)) == 5


def test_money():
    assert cents(1.005) in (100, 101)
    assert sum(split(10, 3)) == pytest.approx(10)


def test_carriers_load():
    assert len(ALL) == 12
    c = ALL["03"]()
    assert c.accepts(1) and not c.accepts(0)
"""

for rel, text in files.items():
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
