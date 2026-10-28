#!/bin/bash
# gpuq job body: plain cold prefill TTFT, one arm per server. fnprefill_ttft.sh OUTDIR ARM(base|cand) SRC_ROOT [ctx...]
set -euo pipefail
OUT=$1; ARM=$2; SRC=$3; shift 3
CTX=${@:-8192 32768}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export TFB_WORK=/Volumes/P5Plus/yunshu-build/tfnew TFB_YUNSHU_SRC="$SRC/python" TFB_OUT="$OUT/tfb-$ARM" TFB_PORT_LAST=18996
export PYTHONPATH="$SRC/python" TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
ENVARG=()
[ "$ARM" != base ] && ENVARG=(--env YUNSHU_QSA_SPARSE_PREFILL=1)
[ "$ARM" = step4096 ] && export FNPREFILL_STEP=4096
cd "$ROOT/scripts/research"
"$ROOT/.venv/bin/python" fnprefill_profile.py --model /Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp --ctx $CTX --plain ${ENVARG[@]+"${ENVARG[@]}"} --out "$OUT/$ARM.json"
