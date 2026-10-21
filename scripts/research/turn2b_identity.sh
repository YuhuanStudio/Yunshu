#!/bin/bash
set -euo pipefail
test "$(cat /Volumes/P5Plus/yunshu-gpuq/logs/1022-030118-00-turn2b-tests-smoke-1022a.rc)" = 0 || exit 3
ROOT=/Volumes/P5Plus/yunshu-build/wt-codex-turn2b
OUT=/Volumes/P5Plus/yunshu-build/verify/runs/turn2b-identity-1022a
mkdir -p "$OUT"
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$OUT/tfb" TFB_PORT_LAST=18996
for kind in code prose; do
 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$ROOT/scripts/research/apc_restore_identity.py" \
 --ctx 32768 --kind "$kind" --turn2 --turn2-repeat 48 --compare-spec-off --out "$OUT/$kind.jsonl" \
 --env YUNSHU_VLM_APC_DISK=0 \
 --env YUNSHU_VLM_APC_DISK_DIR=/Volumes/Micron/yunshu-apc/turn2b-identity-1022a \
 --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2
done
printf '{"complete":true}\n' > "$OUT/complete.json"
