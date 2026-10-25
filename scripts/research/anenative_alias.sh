#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-anenative
export PYTHONPATH=$PWD/python TMPDIR=/Volumes/P5Plus/yunshu-build/codex/anenative-tmp
mkdir -p "$TMPDIR"
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_native_bench.py -q -p no:cacheprovider
exec /Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_alias_probe.py --cache /Volumes/P5Plus/yunshu-build/codex/anenative-alias-cache "$@"
