import sys
from pathlib import Path

w = Path(sys.argv[1])
p = w / "fleet/pricing/fuel_v2.py"
s = p.read_text().replace('"EU-WEST": 0.075,', '"EU-WEST": 0.0825,')
p.write_text(s)
(w / "ANSWER.txt").write_text("fleet/pricing/fuel_v2.py\n")
