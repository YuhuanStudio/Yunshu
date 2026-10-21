#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-turn2prefill
OUT=/Volumes/P5Plus/yunshu-build/verify/runs/turn2prefill-tests-1021b
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
set +e
PYTHONPATH=/Volumes/P5Plus/yunshu-build/codex/turn2prefill-base-719ab7ac/python \
 nice -n 15 "$PY" -m pytest tests/unit/test_nax_prefill.py -k short_tail_native_route \
 -q -p no:cacheprovider > "$OUT/baseline-regression.log" 2>&1
baseline_rc=$?
set -e
cat "$OUT/baseline-regression.log"
[[ "$baseline_rc" == 1 ]] || exit 1
export PYTHONPATH=python
nice -n 15 "$PY" -m pytest tests/unit/test_nax_prefill.py tests/unit/test_turn2prefill_profile.py \
 -q -p no:cacheprovider
printf '{"complete":true,"baseline_regression_rc":%s}\n' "$baseline_rc" > "$OUT/complete.json"
