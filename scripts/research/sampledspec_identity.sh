#!/bin/bash
# Focused sampled APC/spec identity; run only through gpuq after synthetic unit PASS.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CTX=${1:-1024}
TEMP=${2:-0.7}
TOPP=${3:-1}
TOPK=${4:-0}
KIND=${5:-code}
LABEL=sampledspec-id-${CTX}-${TEMP}-${TOPP}-${TOPK}-${KIND}
OUT="$ROOT/docs/research/sampledspec/$LABEL"
APC="/Volumes/Micron/yunshu-apc/$LABEL"
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
[[ "$CTX" =~ ^(1024|32768|131072)$ && "$TEMP" =~ ^(0.7|1.0)$ && "$TOPP" =~ ^(1|0.95)$ && "$TOPK" =~ ^(0|20)$ && "$KIND" =~ ^(code|prose)$ ]]
[[ -f "$ROOT/docs/research/sampledspec/unit.complete.json" ]]
mkdir -p "$OUT"
trap 'rm -rf "$APC"' EXIT
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$OUT/tfb" TFB_PORT_LAST=18996
export PYTHONPATH="$ROOT/python" TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
"$PY" "$ROOT/scripts/research/apc_restore_identity.py" \
 --ctx "$CTX" --kind "$KIND" --turn2 --turn2-repeat 48 --compare-spec-off --tokens 64 --raw-ids \
 --temperature "$TEMP" --top-p "$TOPP" --top-k "$TOPK" --seed 17 --out "$OUT/result.jsonl" \
 --env YUNSHU_VLM_APC_DISK=0 --env YUNSHU_NEXT_TURN_PREFILL=off \
 --env YUNSHU_VLM_APC_DISK_DIR="$APC" \
 --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2
for arm in A B; do
 grep -q 'DFlash fast verification: keyed=True logprobs=True' "$OUT/tfb/out/server-apcid-$arm-$CTX.log"
done
printf '{"complete":true,"engaged":"fast-keyed-logprobs"}\n' > "$OUT/complete.json"
