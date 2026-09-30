"""Split a recorded requests.jsonl (one {path, body} per line) into NNNN-req.json body files.

split_jsonl.py requests.jsonl OUTDIR
"""

import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for i, line in enumerate(src.read_text().splitlines(), 1):
    rec = json.loads(line)
    body = rec.get("body")
    if isinstance(body, str):
        body = json.loads(body)
    (out / f"{i:04d}-req.json").write_text(json.dumps(body))
print(f"{i} bodies -> {out}")
