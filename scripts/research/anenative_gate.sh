#!/bin/bash
# Correctness only, no --quiet needed. Run under gpuq, after phase-1 measurements.
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-anenative
cd "$ROOT"
export PYTHONPATH=$ROOT/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/codex/anenative-tmp
mkdir -p "$TMPDIR"
nice -n 15 /Volumes/P5Plus/yunshu-build/lead/precheck.sh "$ROOT" tests/unit/test_ane_native.py tests/unit/test_ane_convert.py
# just is not installed here; this is the exact vendor-check recipe, using the existing venv.
set +e
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/vendor/check_upstream.py --no-fetch > /Volumes/P5Plus/yunshu-build/codex/anenative-vendor-direct.log 2>&1
vendor_rc=$?
printf '%s\n' "$vendor_rc" > /Volumes/P5Plus/yunshu-build/codex/anenative-vendor-direct.rc
set -e
env ROOT="$ROOT" TAG=anenative- /Volumes/P5Plus/yunshu-build/lead/warmttft/id_any.sh 32768
printf '{"complete":true,"vendor_check_rc":%s}\n' "$vendor_rc"
