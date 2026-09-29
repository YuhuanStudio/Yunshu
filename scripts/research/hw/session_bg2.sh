#!/bin/zsh
# ANE / CPU interference with GPU decode, then the engine-path decode step (one locked session).
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
export PYTHONPATH=$PWD/python:$PWD/scripts/research/hw
H=scripts/research/hw
W=/Volumes/P5Plus/yunshu-test-cache/ane
for m in layer_x1_M8_fp16 layer_x1_M8_pal4 layer_x5_M8_int8 layer_x5_M8_pal4; do
  $PY $H/bg_gpu_concurrency.py --kind ane --model $W/$m.mlpackage --seconds 8
done
$PY $H/bg_gpu_concurrency.py --kind cpu_gemv --procs 1 --seconds 8
$PY $H/bg_gpu_concurrency.py --kind cpu_gemv --procs 4 --seconds 8
$PY $H/bg_gpu_concurrency.py --kind cpu_gemm --procs 2 --seconds 8
$PY $H/decode_step_engine.py /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp
