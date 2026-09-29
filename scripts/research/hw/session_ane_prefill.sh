#!/bin/zsh
# ANE on a prefill-shaped workload (Qwen3.8-27B block, T = 512 and 1024 tokens): alone, then next to a
# GPU decode loop. One locked session; run through gpu_run.sh from the repo root.
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
export PYTHONPATH=$PWD/python:$PWD/scripts/research/hw
H=scripts/research/hw
W=/Volumes/P5Plus/yunshu-test-cache/ane
for spec in M512_pal4:512 M1024_pal4:1024 M1024_fp16:1024; do
  $PY $H/bg_gpu_concurrency.py --kind ane --model $W/layer27_x1_${spec%%:*}.mlpackage --seconds 8 --tokens-per-call ${spec##*:}
done
