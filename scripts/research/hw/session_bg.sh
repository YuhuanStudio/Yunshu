#!/bin/zsh
# ANE / CPU probes and their interference with GPU decode, in one locked session.
[ -f scripts/research/local.env ] && source scripts/research/local.env
PY=${YUNSHU_PY:-$PWD/.venv/bin/python}
ANE=${YUNSHU_ANE_PY:?set YUNSHU_ANE_PY in scripts/research/local.env}
export PYTHONPATH=$PWD/python:$PWD/scripts/research/hw
H=scripts/research/hw
W=${YUNSHU_ANE_WORK:-$HOME/.cache/yunshu/ane}
$PY $H/cpu_probe.py
$ANE $H/ane_probe.py --cases linear,layer --units ne,cpu --iters 30 --ms 1,8,16 --precs fp16,int8,pal4
$ANE $H/ane_probe.py --cases layers5 --units ne --iters 20 --ms 8 --precs int8,pal4
for m in layer_x1_M8_fp16 layer_x1_M8_int8 layer_x1_M8_pal4 layer_x5_M8_int8 layer_x5_M8_pal4; do
  $PY $H/bg_gpu_concurrency.py --kind ane --model $W/$m.mlpackage --seconds 8
done
$PY $H/bg_gpu_concurrency.py --kind cpu_gemv --procs 1 --seconds 8
$PY $H/bg_gpu_concurrency.py --kind cpu_gemv --procs 4 --seconds 8
$PY $H/bg_gpu_concurrency.py --kind cpu_gemm --procs 2 --seconds 8
