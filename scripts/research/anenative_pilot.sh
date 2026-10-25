#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-anenative
export PYTHONPATH=$PWD/python
export TMPDIR=/Volumes/P5Plus/yunshu-build/codex/anenative-tmp
mkdir -p "$TMPDIR"
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_native.py tests/unit/test_ane_convert.py tests/unit/test_vendor_manifest.py tests/unit/test_path_parity.py -q -p no:cacheprovider
/Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_probe.py --cache /Volumes/P5Plus/yunshu-build/codex/anenative-cache --mode fp16 "$@"
/Volumes/P5Plus/venvs/anesplit-mlx/bin/python scripts/research/anenative_probe.py --cache /Volumes/P5Plus/yunshu-build/codex/anenative-cache --mode int8 "$@"
nice -n 15 /Volumes/P5Plus/yunshu-build/lead/precheck.sh "$PWD" tests/unit/test_ane_native.py tests/unit/test_ane_convert.py > /Volumes/P5Plus/yunshu-build/codex/anenative-precheck.log 2>&1
cat /Volumes/P5Plus/yunshu-build/codex/anenative-precheck.log
set +e
UV_PROJECT_ENVIRONMENT=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv UV_NO_SYNC=1 nice -n 15 just vendor-check --no-fetch > /Volumes/P5Plus/yunshu-build/codex/anenative-vendor.log 2>&1
vendor_rc=$?
printf '%s\n' "$vendor_rc" > /Volumes/P5Plus/yunshu-build/codex/anenative-vendor.rc
set -e
printf '{"complete":true,"vendor_check_rc":%s}\n' "$vendor_rc"
