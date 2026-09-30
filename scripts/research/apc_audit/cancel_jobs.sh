#!/bin/bash
# cancel_jobs.sh ID...   (gpuq cancel takes one id)
for id in "$@"; do
  /Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq cancel "$id"
done
