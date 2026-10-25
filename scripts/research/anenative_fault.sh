#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-anenative
export PYTHONPATH=$PWD/python TMPDIR=/Volumes/P5Plus/yunshu-build/codex/anenative-tmp
mkdir -p "$TMPDIR"
PY=/Volumes/P5Plus/venvs/anesplit-mlx/bin/python
M=/Volumes/P5Plus/venvs/anesplit-mlx/lib/python3.13/site-packages/mlx
LIB=/Volumes/P5Plus/yunshu-build/codex/libanenative-fault.dylib
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_ane_native.py tests/unit/test_ane_native_bench.py -q -p no:cacheprovider
nice -n 15 clang++ -std=c++20 -fobjc-arc -fblocks -dynamiclib -O2 -framework Foundation -framework Metal -framework IOSurface \
 -DYS_ANE_MLX -DSPLASH_ANE_INSTRUMENTATION -I"$M/include" -I"$M/include/metal_cpp" -L"$M/lib" -Wl,-rpath,"$M/lib" -lmlx \
 python/yunshu_engine/ane/native/Program.mm python/yunshu_engine/ane/native/Handoff.mm python/yunshu_engine/ane/native/Bridge.mm -o "$LIB"
for fault in selector objc async stalled; do
 "$PY" scripts/research/anenative_fault_probe.py --library "$LIB" --cache /Volumes/P5Plus/yunshu-build/codex/anenative-fault-cache --fault "$fault"
done
printf '{"complete":true}\n'
