#!/bin/zsh
# 27B validation of the round driver (YUNSHU_ROUND_DRIVER) on an idle GPU.
#
#   zsh scripts/research/validate_round_driver.sh            # every phase
#   PHASE=inprocess|server|mmlu zsh scripts/research/validate_round_driver.sh
#
# inprocess: sweep_round_driver.py parity (alone / batch / staggered, MTP on
#            and off must match token for token) and rows 1/2/4/8 throughput,
#            at 1K and 32K context.
# server:    off vs on, port 18764: probe_concurrency n=8, bench_engine_matrix,
#            bench_context_batch (1K/32K/131K, b2/b4/b8), bench_mixed_load
#            (4 streams + 16K / 32K prompt). The "on" server log must show
#            "Round driver: N lane projections".
# mmlu:      MMLU-Pro 300 b8 (16384, medium) with the driver on.
#
# M (the Qwen3.8-27B checkpoint dir) comes from the environment or
# scripts/research/local.env (gitignored).
set -u
cd "$(dirname "$0")/../.."
[ -f scripts/research/local.env ] && source scripts/research/local.env
M=${M:?set M to the Qwen3.8-27B checkpoint directory}
PORT=${PORT:-18764}
OUT=${OUT:-docs/research/runs/$(date +%Y-%m-%d)-round-driver}
PHASE=${PHASE:-all}
URL=http://127.0.0.1:$PORT
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*"; }
wait_ready(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health/ready 2>/dev/null | grep -q '"ready":true' && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
stop(){ kill -INT $1 2>/dev/null; for i in $(seq 1 30); do kill -0 $1 2>/dev/null || return; sleep 1; done; kill $1 2>/dev/null; }
serve(){  # $1 = 0|1; sets YP
  YUNSHU_ROUND_DRIVER=$1 YUNSHU_MODEL=$M YUNSHU_AUTH_DISABLED=1 \
    $PY -m uvicorn yunshu_gateway.main:app --host 127.0.0.1 --port $PORT > $OUT/server-rd$1.log 2>&1 &
  YP=$!
  wait_ready $YP
}

if [ $PHASE = all -o $PHASE = inprocess ]; then
  log "in-process sweep 1K"
  $PY scripts/research/sweep_round_driver.py $M --rows 1 2 4 8 --tokens 256 \
    --output $OUT/sweep.jsonl > $OUT/sweep-1k.log 2>&1 || log "sweep 1K FAILED"
  log "in-process sweep 32K"
  $PY scripts/research/sweep_round_driver.py $M --rows 1 4 --tokens 256 --context 32768 \
    --output $OUT/sweep.jsonl > $OUT/sweep-32k.log 2>&1 || log "sweep 32K FAILED"
  grep '"kind": "parity"' $OUT/sweep.jsonl
fi

if [ $PHASE = all -o $PHASE = server ]; then
  for rd in 0 1; do
    if serve $rd; then
      [ $rd = 0 ] || grep -q "Round driver: [0-9]* lane projections" $OUT/server-rd1.log \
        && log "rd=$rd up" || log "ROUND DRIVER NOT ENGAGED"
      $PY scripts/research/probe_concurrency.py --url $URL --model Qwen3.8-27B --n 8 \
        --note "round driver=$rd" --output $OUT/concurrency.jsonl > /dev/null 2>&1 || log "concurrency rd=$rd FAILED"
      $PY scripts/research/bench_engine_matrix.py --url $URL --model Qwen3.8-27B --engine yunshu-rd$rd \
        --checkpoint $M --pid $YP --note "round driver=$rd" --output $OUT/matrix.jsonl > /dev/null 2>&1 || log "matrix rd=$rd FAILED"
      $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
        --lengths 1024 32768 131072 --batches 2 4 8 --note "round driver=$rd" \
        --output $OUT/context-batch.jsonl > /dev/null 2>&1 || log "context rd=$rd FAILED"
      for pp in 16384 32768; do
        $PY scripts/research/bench_mixed_load.py --url $URL --model Qwen3.8-27B --tokenizer $M \
          --streams 4 --pp $pp --label rd$rd --output $OUT/mixed-load.jsonl > /dev/null 2>&1 || log "mixed rd=$rd pp=$pp FAILED"
      done
      log "rd=$rd server done"
    else log "rd=$rd server failed"; fi
    stop $YP; sleep 5
  done
fi

if [ $PHASE = all -o $PHASE = mmlu ]; then
  if serve 1; then
    log "mmlu rd=1"
    $PY scripts/research/soak_mmlu_pro.py --url $URL --model Qwen3.8-27B --pid $YP \
      --note "Yunshu round driver" --output $OUT/mmlu-soak.jsonl > $OUT/mmlu-soak.log 2>&1 || log "mmlu FAILED"
  else log "mmlu server failed"; fi
  stop $YP
fi
log "round driver validation done"
