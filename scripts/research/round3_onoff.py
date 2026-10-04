"""probe_checkpoint_stores.py with the round driver serving every text request
(a lone request included), so its allow_draft on/off digests compare the
driver's own spec-on and spec-off rows. Run under gpuq with YUNSHU_ROUND_DRIVER=1.
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))


def main() -> None:
    from yunshu_engine import vlm_batch_runner

    vlm_batch_runner.DRIVER_MIN_CONCURRENCY = 1
    runpy.run_path(str(HERE / "probe_checkpoint_stores.py"), run_name="__main__")


if __name__ == "__main__":
    main()
