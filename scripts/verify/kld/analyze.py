"""Run the kld analysis under the engine interpreter (numpy): prints one JSON document.

python -m verify.kld.analyze --ref DIR --base DIR --cand DIR [--eps X] [--top1-noise Y]
"""

from __future__ import annotations

import argparse
import json
import sys

from . import corpus, stage


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--eps", type=float)
    p.add_argument("--top1-noise", type=float)
    a = p.parse_args(argv)
    manifest = corpus.load_manifest()
    from pathlib import Path

    arms = {
        n: stage.load_arm(Path(d), manifest)
        for n, d in (("ref", a.ref), ("base", a.base), ("cand", a.cand))
    }
    noise = stage.noise()
    result = stage.evaluate(
        arms["ref"],
        arms["base"],
        arms["cand"],
        a.eps if a.eps is not None else noise["eps"],
        a.top1_noise if a.top1_noise is not None else noise["top1_noise"],
    )
    result["noise_source"] = noise["source"]
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
