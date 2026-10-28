#!/bin/bash
# gpuq job body: thinking MMLU-Pro n=60, candidate only (broken-output sanity). sanity60.sh OUTDIR
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$1
export PAIRED_TREE="$ROOT" PAIRED_OUT="$OUT" PAIRED_PORT_LAST=18996 PAIRED_MAX_TOKENS=16384 PAIRED_THINKING=1 PAIRED_PY="$ROOT/.venv/bin/python"
mkdir -p "$OUT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/research/accuracy/paired_eval.py" run --bench mmlu_pro --mmlu-n 60 --arm cand \
  --model /Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp-ple-ssd --model-name Qwen3.8-Flash-Next \
  --budget-min 40 --concurrency 8 --port 18990 --env YUNSHU_QSA_SPARSE_PREFILL=1
