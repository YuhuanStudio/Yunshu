#!/bin/zsh
# ANE on a prefill-shaped workload (Qwen3.8-27B block, T = 64..1024 tokens): alone, then next to a
# GPU decode loop. One locked session; run through gpu_run.sh from the repo root.
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
export PYTHONPATH=$PWD/python:$PWD/scripts/research/hw
H=scripts/research/hw
# packages already built by ane_probe.py --arch q27 (see its rows); best two by tokens/s
$PY $H/pick_best_ane.py 2 | while read pkg tokens; do
  $PY $H/bg_gpu_concurrency.py --kind ane --model $pkg --seconds 8 --tokens-per-call $tokens
done
