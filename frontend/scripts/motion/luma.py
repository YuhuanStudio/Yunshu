"""Mean luminance of the content area per frame: a flash (blank or inverted frame) shows as a dip/spike.
Usage: python luma.py <scenario-dir> [x0 y0 x1 y1]"""
import glob, sys
from PIL import Image, ImageStat
d = sys.argv[1]
box = tuple(int(v) for v in sys.argv[2:6]) if len(sys.argv) >= 6 else (290, 80, 1400, 800)
prev = None
out = []
for p in sorted(glob.glob(d + "/f_*.jpg")):
    m = ImageStat.Stat(Image.open(p).convert("L").crop(box)).mean[0]
    out.append((int(p.rsplit("_", 1)[1][:5]), round(m, 1)))
print(" ".join(f"{t}:{m}" for t, m in out))
vals = [m for _, m in out]
print("min", min(vals), "max", max(vals))
