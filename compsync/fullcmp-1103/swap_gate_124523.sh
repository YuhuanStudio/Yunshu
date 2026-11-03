#!/bin/bash
cd /Volumes/P5Plus/yunshu-build/wt-fullcmp-harness || exit 2
OUT=/Volumes/P5Plus/yunshu-build/wt-fullcmp-harness/compsync/fullcmp-1103/swap_preflight_124523.json
echo "== before"; sysctl vm.swapusage
memory_pressure > /Volumes/P5Plus/yunshu-build/wt-fullcmp-harness/compsync/fullcmp-1103/memory_pressure_before_124523.txt 2>&1
PYTHONPATH=scripts/research python3 scripts/research/fullcmp_preflight.py --window-s 300 --interval-s 10 --limit-mb 64 --out "$OUT"
RC=$?
echo "EXIT=$RC"
echo "== after"; sysctl vm.swapusage
memory_pressure > /Volumes/P5Plus/yunshu-build/wt-fullcmp-harness/compsync/fullcmp-1103/memory_pressure_after_124523.txt 2>&1
exit $RC
