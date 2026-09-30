"""Print the engine messages and the rendered text (head / tail) for one captured body."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_bodies import Renderer, load_body  # noqa: E402

logging.disable(logging.CRITICAL)
model, body = sys.argv[1], sys.argv[2]
r = Renderer(model)
out = r.render(load_body(Path(body)))
for i, m in enumerate(out["messages"]):
    c = m.get("content")
    print(
        i,
        m.get("role"),
        repr(c if isinstance(c, str) else json.dumps(c))[:120],
        list(m),
    )
t = out["text"]
print("---- text len", len(t), "tokens", len(out["ids"]))
print(t[:300])
print("....")
print(t[-2500:])
