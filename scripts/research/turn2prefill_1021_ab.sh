#!/bin/bash
set -uo pipefail
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-turn2prefill
BASE=/Volumes/P5Plus/yunshu-build/codex/turn2prefill-base-719ab7ac
OUT=/Volumes/P5Plus/yunshu-build/verify/runs/turn2prefill-ab-1021c
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
mkdir -p "$OUT"
run() {
 local arm=$1 src=$2
 env TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1 \
 TFB_YUNSHU_SRC="$src/python" TFB_OUT="$OUT/tfb.$arm" TFB_PORT_LAST=18996 \
 "$PY" "$ROOT/scripts/research/tfbench.py" \
 --engine yunshu --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
 --out "$OUT/$arm.jsonl" --tag="-turn2prefill-1021b-$arm" \
 --env "YUNSHU_VLM_APC_DISK_DIR=/Volumes/Micron/yunshu-apc/turn2prefill-ab-1021c-$arm-$$" \
 --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2 \
 --part decode --rep 0 --only-ctx 32768 --only-kind code --only-kind prose \
 --decode-tokens 512 --long-ask --phases cold,warm,turn2
 local rc=$?
 echo "$rc" > "$OUT/$arm.rc"
 return "$rc"
}
failed=0
trap "rm -rf /Volumes/Micron/yunshu-apc/turn2prefill-ab-1021c-*-$$" EXIT
run base "$BASE" || failed=1
run cand "$ROOT" || failed=1
run cand-r1 "$ROOT" || failed=1
run base-r1 "$BASE" || failed=1
printf '{"complete":true,"failed":%s}\n' "$failed" > "$OUT/complete.json"
exit "$failed"
