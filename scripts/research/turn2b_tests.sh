#!/bin/bash
set -euo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-turn2b
BASE=/Volumes/P5Plus/yunshu-build/codex/turn2b-base-17850ae1
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
cd "$ROOT"
mkdir -p "$ROOT/.scratch/turn2b"
export TMPDIR="$ROOT/.scratch/turn2b"
set +e
nice -n 15 env PYTHONPATH="$BASE/python" "$PY" -m pytest tests/unit/test_nax_prefill.py -k short_tail_native_route -q -p no:cacheprovider
rc=$?
set -e
[ "$rc" = 1 ] || exit 2
nice -n 15 env PYTHONPATH="$ROOT/python" "$PY" -m pytest tests/unit/test_nax_prefill.py tests/unit/test_turn2prefill_profile.py -q -p no:cacheprovider
printf '{"complete":true}\n'
