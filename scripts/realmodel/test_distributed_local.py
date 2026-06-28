"""Test mx.distributed with local multi-process (ring backend).

Uses the ring backend over TCP sockets. Requires MLX_HOSTFILE + MLX_RANK env vars.

Usage:
    PYTHONPATH=. uv run python scripts/test_distributed_local.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading

NUM_PROCS = 2
BASE_PORT = 29600

WORKER = r"""
import os
import sys
import mlx.core as mx

rank = int(os.environ.get("MLX_RANK", "-1"))
print(f'[ring] rank_env={rank} starting...', flush=True)

group = mx.distributed.init(backend='ring')
rank = group.rank()
size = group.size()
print(f'[ring] Connected: rank={rank} size={size}', flush=True)

if size <= 1:
    print(f'[ring] Singleton group — no distributed ops', flush=True)
    sys.exit(0)

passed = 0
failed = 0

# Test 1: all_sum (scalar)
try:
    x = mx.array([float(rank + 1)])
    result = mx.distributed.all_sum(x, group=group)
    mx.eval(result)
    expected = size * (size + 1) / 2.0
    assert abs(result.item() - expected) < 0.01, f"{result.item()} != {expected}"
    print(f'  PASS all_sum scalar: rank={rank} result={result.tolist()}', flush=True)
    passed += 1
except Exception as e:
    print(f'  FAIL all_sum scalar: rank={rank} {e}', flush=True)
    failed += 1

# Test 2: all_sum (vector)
try:
    v = mx.array([float(rank), float(rank * 2), float(rank * 3)])
    result_v = mx.distributed.all_sum(v, group=group)
    mx.eval(result_v)
    print(f'  PASS all_sum vector: rank={rank} result={result_v.tolist()}', flush=True)
    passed += 1
except Exception as e:
    print(f'  FAIL all_sum vector: rank={rank} {e}', flush=True)
    failed += 1

# Test 3: all_gather
try:
    y = mx.array([float(rank * 100)])
    gathered = mx.distributed.all_gather(y, group=group)
    mx.eval(gathered)
    expected = [float(r * 100) for r in range(size)]
    assert gathered.tolist() == expected, f"{gathered.tolist()} != {expected}"
    print(f'  PASS all_gather: rank={rank} result={gathered.tolist()}', flush=True)
    passed += 1
except Exception as e:
    print(f'  FAIL all_gather: rank={rank} {e}', flush=True)
    failed += 1

# Test 4: all_gather (vector)
try:
    z = mx.array([float(rank + 1), float((rank + 1) * 10)])
    gathered2 = mx.distributed.all_gather(z, group=group)
    mx.eval(gathered2)
    print(f'  PASS all_gather vector: rank={rank} result={gathered2.tolist()}', flush=True)
    passed += 1
except Exception as e:
    print(f'  FAIL all_gather vector: rank={rank} {e}', flush=True)
    failed += 1

# Test 5: all_max
try:
    m = mx.array([float(rank * 7)])
    result_m = mx.distributed.all_max(m, group=group)
    mx.eval(result_m)
    print(f'  PASS all_max: rank={rank} result={result_m.tolist()}', flush=True)
    passed += 1
except Exception as e:
    print(f'  FAIL all_max: rank={rank} {e}', flush=True)
    failed += 1

# Test 6: all_min
try:
    m = mx.array([float(rank * 7)])
    result_m = mx.distributed.all_min(m, group=group)
    mx.eval(result_m)
    print(f'  PASS all_min: rank={rank} result={result_m.tolist()}', flush=True)
    passed += 1
except Exception as e:
    print(f'  FAIL all_min: rank={rank} {e}', flush=True)
    failed += 1

mx.synchronize()
print(f'[ring] rank={rank} DONE: {passed} passed, {failed} failed', flush=True)
sys.exit(0 if failed == 0 else 1)
"""

if __name__ == "__main__":
    print(f"Testing mx.distributed ring backend with {NUM_PROCS} processes")
    print("=" * 60)

    # Create JSON hostfile: [ ["ip:port"], ["ip:port"], ... ]
    hostfile_data = [
        [f"127.0.0.1:{BASE_PORT + i}"]
        for i in range(NUM_PROCS)
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(hostfile_data, f)
        hostfile = f.name

    print(f"Hostfile: {json.dumps(hostfile_data)}")

    # Launch all processes simultaneously
    procs = []
    results = {}

    def run_proc(rank, p):
        try:
            stdout, stderr = p.communicate(timeout=30)
            results[rank] = (p.returncode, stdout.decode(), stderr.decode())
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            results[rank] = (
                -1,
                (stdout or b"").decode(),
                (stderr or b"").decode() + " TIMEOUT",
            )

    for rank in range(NUM_PROCS):
        env = os.environ.copy()
        env['MLX_HOSTFILE'] = hostfile
        env['MLX_RANK'] = str(rank)
        p = subprocess.Popen(
            [sys.executable, '-c', WORKER],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)
        print(f"  Launched rank={rank} pid={p.pid}")

    # Wait concurrently (avoid sequential timeout issues)
    threads = []
    for rank, p in enumerate(procs):
        t = threading.Thread(target=run_proc, args=(rank, p))
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=45)

    total_passed = 0
    total_failed = 0
    for rank in range(NUM_PROCS):
        if rank in results:
            rc, stdout, stderr = results[rank]
            print(f"\n--- Rank {rank} (exit={rc}) ---")
            print(stdout)
            if "TIMEOUT" in (stderr or ""):
                print("STDERR: TIMEOUT")
            elif stderr.strip():
                print(f"STDERR: {stderr[-300:]}")

            for line in stdout.split('\n'):
                if 'PASS' in line:
                    total_passed += 1
                elif 'FAIL' in line:
                    total_failed += 1

    os.unlink(hostfile)

    print(f"\n{'=' * 60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    if total_failed == 0 and total_passed > 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
