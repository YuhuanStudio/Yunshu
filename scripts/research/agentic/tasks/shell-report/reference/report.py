import csv
import json
import sys
from pathlib import Path


def num(s):
    s = s.strip().lstrip("$").replace(",", ".")
    return float(s)


def main(data_dir, out):
    files = rows = bad = 0
    rev = {}
    for p in sorted(Path(data_dir).iterdir()):
        if p.suffix not in (".csv", ".txt"):
            continue
        files += 1
        text = p.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        first = lines[0]
        delim = max(",;\t|", key=first.count)
        rd = csv.reader(lines, delimiter=delim)
        head = [h.strip().lower() for h in next(rd)]
        for r in rd:
            d = dict(zip(head, r))
            try:
                q = d["quantity"].strip()
                if not q.isdigit() or int(q) <= 0:
                    raise ValueError
                price = num(d["unit_price"])
                if price <= 0:
                    raise ValueError
                region = d["region"].strip().upper()
                if not region:
                    raise ValueError
            except (ValueError, KeyError):
                bad += 1
                continue
            rows += 1
            rev[region] = rev.get(region, 0) + int(q) * price
    Path(out).write_text(
        json.dumps(
            {
                "files": files,
                "rows": rows,
                "bad_rows": bad,
                "revenue_by_region": {k: round(v, 2) for k, v in sorted(rev.items())},
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
