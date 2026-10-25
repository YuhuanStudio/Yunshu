#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-anenative2
export PYTHONPATH=$PWD/python TMPDIR=/Volumes/P5Plus/yunshu-build/tmp/anenative2
mkdir -p "$TMPDIR"
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_native_bench.py -q -p no:cacheprovider
exec /Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_coreml_plan.py --out /Volumes/P5Plus/yunshu-build/codex/anenative-coreml-plan.json --packages \
 /Volumes/P5Plus/yunshu-build/codex/anenative-conc-pilot-cache/l40-c17408-r128-fp16.mlpackage \
 /Volumes/P5Plus/yunshu-build/codex/anenative-matrix-cache/l40-c17408-r128-int8-h6.mlpackage \
 /Volumes/P5Plus/yunshu-build/codex/anenative-matrix-cache/l40-c2176-r1024-fp16-h6.mlpackage \
 /Volumes/P5Plus/yunshu-build/codex/anenative-matrix-cache/l40-c4352-r1024-int8-h6.mlpackage
