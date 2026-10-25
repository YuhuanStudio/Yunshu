#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-anenative2
export PYTHONPATH=$PWD/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_ffn_split.py tests/unit/test_ane_native.py -q -p no:cacheprovider
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/anenative_splash_kernels_smoke.py "$@"
