#!/bin/bash
# gpuq job body: profiled then plain cold prefill, one server each. fnprefill_job.sh OUTDIR [ctx...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$1; shift
CTX=${@:-8192 32768}
export TFB_WORK=/Volumes/P5Plus/yunshu-build/tfnew TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$OUT/tfb" TFB_PORT_LAST=18996
export PYTHONPATH="$ROOT/python" TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
M=/Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp
cd "$ROOT/scripts/research"
"$ROOT/.venv/bin/python" fnprefill_profile.py --model $M --ctx $CTX --plain --out "$OUT/plain.json"
"$ROOT/.venv/bin/python" fnprefill_profile.py --model $M --ctx $CTX --out "$OUT/profiled.json"
