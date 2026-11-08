"""Does mx.save_safetensors on a worker thread stall other Python threads (GIL held)?"""

import tempfile
import threading
import time

import mlx.core as mx

a = {f"k{i}": mx.ones((1, 4, 8192, 128), dtype=mx.bfloat16) for i in range(24)}
mx.eval(list(a.values()))
d = tempfile.mkdtemp(dir=".")
gaps, stop = [], threading.Event()


def spin():
    last = time.perf_counter()
    while not stop.is_set():
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
        time.sleep(0.001)


t = threading.Thread(target=spin)
t.start()
time.sleep(0.2)
t0 = time.perf_counter()
w = threading.Thread(target=lambda: mx.save_safetensors(d + "/x.safetensors", a))
w.start()
w.join()
dt = time.perf_counter() - t0
stop.set()
t.join()
print(f"write {dt * 1e3:.0f} ms, worst gap of a 1 ms ticker {max(gaps) * 1e3:.0f} ms")
import shutil

shutil.rmtree(d)
