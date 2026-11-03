#!/bin/bash
export TMPDIR=/Volumes/Micron/yunshu-fullcmp/tmp
T=/Volumes/Micron/yunshu-fullcmp/xctrace/yunshu-main/code-1k-out128.trace
XC=/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace
D=/Volumes/Micron/yunshu-fullcmp/xctrace/inspect
mkdir -p $D
$XC export --input $T --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > $D/gpu-intervals.xml 2>$D/err.txt
echo rc=$?
ls -la $D
head -c 3000 $D/gpu-intervals.xml
echo
grep -o '<row[^>]*>' $D/gpu-intervals.xml | head -2
grep -o '<schema[^>]*>' $D/gpu-intervals.xml | head -1
grep -o '<col>[^<]*</col>\|<mnemonic>[^<]*</mnemonic>' $D/gpu-intervals.xml | head -20
