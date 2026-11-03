#!/bin/bash
export TMPDIR=/Volumes/Micron/yunshu-fullcmp/tmp
cd /Volumes/P5Plus/yunshu-build/wt-fullcmp-harness || exit 2
PYTHONPATH=scripts/research /Volumes/P5Plus/yunshu-build/wt-fullcmp-harness/.venv/bin/python scripts/research/fullcmp_xctrace.py --validate-only /Volumes/Micron/yunshu-fullcmp/xctrace/yunshu-main/code-1k-out128.trace --pid 50342
