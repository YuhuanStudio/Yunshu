#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-anenative2
export PYTHONPATH=$PWD/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$TMPDIR"
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_native.py tests/unit/test_ane_convert.py -q -p no:cacheprovider
exec /Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_micro.py "$@"
