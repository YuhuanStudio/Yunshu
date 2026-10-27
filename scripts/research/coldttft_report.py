"""CPU-only: render analyzed cold-prefill span tables (reads analysis.json files)."""

import json
import sys
from pathlib import Path


def main(path, top=45):
    a = json.loads(Path(path).read_text())["profile"]
    print({k: round(v, 3) for k, v in a.items() if isinstance(v, float)})
    for s in a["spans"][:top]:
        print(
            f"{s['name'][:46]:46} {s['seconds']:7.3f}s n={s['count']:4d} "
            f"TF={s.get('tflops', 0):6.1f} pk={s.get('compute_peak_fraction', 0):.2f} "
            f"GB/s={s['minimum_tensor_gbps']:6.0f} floor={s.get('arithmetic_floor_s', 0):.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
