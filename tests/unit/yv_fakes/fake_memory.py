"""Stand-in for memory_ab.py: one arm per job."""

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append")
ap.add_argument("--arm-env", action="append", default=[])
ap.add_argument("--model")
ap.add_argument("--port")
ap.add_argument("--reps", type=int)
ap.add_argument("--rep-offset", type=int, default=0)
ap.add_argument("--sizes", nargs="+")
ap.add_argument("--out")
a = ap.parse_args()
name = a.arm[0].split("=")[0]
env = dict(
    x.split(":", 1)[1].split("=", 1) for x in a.arm_env if x.startswith(name + ":")
)
hog = 3.0 if env.get("FAKE_MEM_HOG") == "1" else 0.0
with open(a.out, "a") as f:
    for step, fp in (
        ("ready", 5.0),
        ("32k-turn1", 9.0 + hog),
        ("idle20s", 6.0 + hog),
        ("idle-after", 5.5 + hog),
    ):
        f.write(
            json.dumps(
                {
                    "arm": name,
                    "rep": a.rep_offset,
                    "step": step,
                    "footprint_gib": fp,
                    "peak_footprint_gib": 9.5 + hog,
                }
            )
            + "\n"
        )
    f.write(json.dumps({"complete": True}) + "\n")
