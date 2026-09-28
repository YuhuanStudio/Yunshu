#!/bin/zsh
# 27B validation of the DFlash2 path (python/yunshu_engine/dflash_context.py:
# per-row prefill with capture, shared verify head, drafter context window;
# batch-invariant verify kernels; block ceiling = the drafter's trained block).
#
#   zsh scripts/research/validate_dflash.sh                   # every phase
#   PHASE=inprocess|server|tensorfold zsh scripts/research/validate_dflash.sh
#
# inprocess:  sweep_mtp_depth.py --dflash, block ceilings 3/4/5/6/8 vs plain
#             (parity + decode tok/s) for code/prose/json_like at 1K and 32K;
#             131K for code+prose at ceilings 4 and 8.
# server:     Yunshu DFlash2 (default ceiling) and Yunshu MTP (default):
#             bench_context_batch on code_python and novel_en, and
#             bench_engine_matrix for DFlash2.
# tensorfold: TensorFold 0.3.6.1 DFlash2 bench_context_batch on novel_en (its
#             code_python run: docs/research/runs/2026-09-28-tensorfold).
set -u
cd "$(dirname "$0")/../.."
M=${M:-/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp}
D=${D:-/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2}
PORT=${PORT:-18764}
OUT=${OUT:-docs/research/runs/$(date +%Y-%m-%d)-dflash}
PHASE=${PHASE:-all}
URL=http://127.0.0.1:$PORT
PY=.venv/bin/python
TF=${TF:-/Volumes/P5Plus/yunshu-test-envs/tensorfold/bin/tensorfold}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*"; }
wait_ready(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health/ready 2>/dev/null | grep -q '"ready":true' && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
wait_models(){ for i in $(seq 1 300); do curl -s -m 2 $URL/v1/models 2>/dev/null | grep -q '"id"' && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
stop(){ kill -INT $1 2>/dev/null; for i in $(seq 1 30); do kill -0 $1 2>/dev/null || return; sleep 1; done; kill $1 2>/dev/null; }
KERNELS=(--yunshu-kernels=exact --invariant --invariant-packed --ragged-lane)

if [ $PHASE = all -o $PHASE = inprocess ]; then
  for ctx in 1024 32768; do
    log "in-process dflash ctx=$ctx"
    $PY scripts/research/sweep_mtp_depth.py $M 3 4 5 6 8 --dflash=$D $KERNELS --context=$ctx \
      >> $OUT/sweep-ctx$ctx.jsonl 2> $OUT/sweep-ctx$ctx.err || log "sweep ctx=$ctx FAILED"
  done
  log "in-process dflash ctx=131072"
  $PY scripts/research/sweep_mtp_depth.py $M 4 8 --dflash=$D $KERNELS --context=131072 \
    --tasks=code,prose >> $OUT/sweep-ctx131072.jsonl 2> $OUT/sweep-ctx131072.err \
    || log "sweep ctx=131072 FAILED"
  grep -h '"parity": false' $OUT/sweep-ctx*.jsonl && log "PARITY FAILURES above" || log "parity ok"
fi

if [ $PHASE = all -o $PHASE = server ]; then
  for draft in dflash mtp; do
    if [ $draft = dflash ]; then
      YUNSHU_VLM_DRAFT=$D YUNSHU_MODEL=$M YUNSHU_AUTH_DISABLED=1 \
        $PY -m uvicorn yunshu_gateway.main:app --host 127.0.0.1 --port $PORT > $OUT/server-$draft.log 2>&1 &
    else
      YUNSHU_MODEL=$M YUNSHU_AUTH_DISABLED=1 \
        $PY -m uvicorn yunshu_gateway.main:app --host 127.0.0.1 --port $PORT > $OUT/server-$draft.log 2>&1 &
    fi
    YP=$!
    if wait_ready $YP; then
      grep -o "draft=[a-z]* block=[0-9a-z]*" $OUT/server-$draft.log | head -1
      for corpus in code_python novel_en; do
        log "server $draft $corpus"
        $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M \
          --pid $YP --corpus $corpus --note "Yunshu $draft $corpus" \
          --output $OUT/speed-$draft.jsonl > /dev/null 2>&1 || log "speed $draft $corpus FAILED"
      done
      if [ $draft = dflash ]; then
        $PY scripts/research/bench_engine_matrix.py --url $URL --model Qwen3.8-27B \
          --engine yunshu-dflash2 --checkpoint $M --pid $YP --note "DFlash2 lossless path" \
          --output $OUT/matrix-dflash.jsonl > /dev/null 2>&1 || log "matrix FAILED"
      fi
    else log "$draft server failed"; fi
    stop $YP; sleep 5
  done
fi

if [ $PHASE = all -o $PHASE = tensorfold ]; then
  C=/Volumes/P5Plus/yunshu-test-cache/tensorfold-snap
  mkdir -p $C
  $TF serve $M --port $PORT --name qwen38 --parallel 8 --drafter $D --snapshot-dir $C \
    --no-update-check > $OUT/server-tensorfold.log 2>&1 &
  TP=$!
  if wait_models $TP; then
    log "tensorfold dflash2 novel_en"
    $PY scripts/research/bench_context_batch.py --url $URL --model qwen38 --tokenizer $M \
      --pid $TP --corpus novel_en --note "TensorFold DFlash2 novel_en" \
      --output $OUT/speed-tensorfold.jsonl > /dev/null 2>&1 || log "tensorfold speed FAILED"
  else log "tensorfold server failed"; fi
  stop $TP; rm -rf $C
fi
log "dflash validation done"
