#!/bin/bash
set -euo pipefail
cd "${ANENATIVE_ROOT:-/Volumes/P5Plus/yunshu-build/wt-anenative2}"
export PYTHONPATH=$PWD/python TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1 TFB_PORT_LAST=18996
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$TMPDIR"
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_native_bench.py -q -p no:cacheprovider
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/anenative_warm_interference.py "$@"
