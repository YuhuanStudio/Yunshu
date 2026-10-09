"""Build the hard-linked external-PLE view of a Flash-Next pack (the source pack is never modified).

    flashnext80_prepare_view.py SOURCE_PACK TARGET_VIEW [--cache-rows N]

The view holds hard links to the source shards plus a manifest of the PLE byte ranges, so the n-gram
table (32 GB) is read row by row from the SSD instead of being loaded. No payload is copied.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--cache-rows", type=int, default=0)
    a = ap.parse_args(argv)
    from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model

    provenance = prepare_external_ple_model(a.source, a.target, cache_rows=a.cache_rows)
    print(json.dumps(provenance, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
