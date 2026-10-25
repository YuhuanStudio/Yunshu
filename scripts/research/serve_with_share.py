"""Measurement launcher: the engine CLI with DECODE_TAIL_SHARE overridden (a sweep adds no flag).

python serve_with_share.py SHARE <engine cli args>
"""

import runpy
import sys

from yunshu_engine.serving import work_scheduler

work_scheduler.DECODE_TAIL_SHARE = float(sys.argv[1])
sys.argv = ["yunshu_cli", *sys.argv[2:]]
runpy.run_module("yunshu_cli", run_name="__main__")
