#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-anenative
export PYTHONPATH=$PWD/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/codex/anenative-tmp
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_ffn_split.py tests/unit/test_ane_native.py -q -p no:cacheprovider
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/anenative_splash_kernels_smoke.py "$@"
