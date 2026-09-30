"""Generate messy sales exports.  usage: gen.py OUTDIR [SEED]  (writes OUTDIR/data/*).

Each file uses a different dialect: delimiter (comma, semicolon, tab, pipe), column order, header
capitalisation, a UTF-8 BOM, CRLF endings, decimal commas, currency prefixes and a few corrupt rows.
Also writes OUTDIR/expected.json (only when SEED is given explicitly, for the grader).
"""

import json
import random
import sys
from pathlib import Path

out = Path(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rnd = random.Random(seed)
data = out / "data"
data.mkdir(parents=True, exist_ok=True)

REGIONS = ["north", "south", "east", "west", "central"]
DIALECTS = [
    dict(
        name="a.csv",
        delim=",",
        cols=["region", "quantity", "unit_price", "sku"],
        crlf=False,
        bom=False,
        comma_dec=False,
    ),
    dict(
        name="b.csv",
        delim=";",
        cols=["sku", "Region", "Quantity", "Unit_Price"],
        crlf=True,
        bom=False,
        comma_dec=True,
    ),
    dict(
        name="c.txt",
        delim="\t",
        cols=["quantity", "unit_price", "region", "sku"],
        crlf=False,
        bom=True,
        comma_dec=False,
    ),
    dict(
        name="d.csv",
        delim="|",
        cols=["REGION", "SKU", "QUANTITY", "UNIT_PRICE"],
        crlf=False,
        bom=False,
        comma_dec=False,
    ),
    dict(
        name="e.txt",
        delim=",",
        cols=["sku", "region", "quantity", "unit_price"],
        crlf=True,
        bom=True,
        comma_dec=False,
    ),
]
expected = {"files": 0, "rows": 0, "bad_rows": 0, "revenue_by_region": {}}
for d in DIALECTS:
    lines = [d["delim"].join(d["cols"])]
    if rnd.random() < 0.5:
        lines.append("")
    for _ in range(rnd.randint(25, 40)):
        region = rnd.choice(REGIONS)
        qty = rnd.randint(1, 20)
        price = round(rnd.uniform(1, 90), 2)
        kind = rnd.random()
        rvals = {
            "region": " " + region.upper() + " "
            if rnd.random() < 0.3
            else region.capitalize(),
            "sku": f"SKU{rnd.randint(100, 999)}",
            "quantity": str(qty),
        }
        pstr = f"{price:.2f}"
        if d["comma_dec"]:
            pstr = pstr.replace(".", ",")
        if rnd.random() < 0.15:
            pstr = "$" + pstr
        rvals["unit_price"] = pstr
        good = True
        if kind < 0.08:
            rvals["quantity"] = rnd.choice(["", "n/a", "-3", "2.5"])
            good = False
        elif kind < 0.14:
            rvals["unit_price"] = rnd.choice(["", "free", "-1.00"])
            good = False
        elif kind < 0.18:
            rvals["region"] = ""
            good = False
        cells = [rvals[c.lower()] for c in d["cols"]]
        lines.append(d["delim"].join(cells))
        if good:
            expected["rows"] += 1
            reg = region.upper()
            expected["revenue_by_region"][reg] = (
                expected["revenue_by_region"].get(reg, 0) + qty * price
            )
        else:
            expected["bad_rows"] += 1
    text = ("\r\n" if d["crlf"] else "\n").join(lines) + ("\r\n" if d["crlf"] else "\n")
    (data / d["name"]).write_text(
        ("﻿" if d["bom"] else "") + text, encoding="utf-8", newline=""
    )
    expected["files"] += 1
expected["revenue_by_region"] = {
    k: round(v, 2) for k, v in sorted(expected["revenue_by_region"].items())
}
if len(sys.argv) > 2:
    (out / "expected.json").write_text(json.dumps(expected))
