"""Hit-rate / TTFT model of the APC hierarchy under multi-session agent traffic (CPU only, seconds).

Parameters are the measured ones for Qwen3.8-27B oQ4e on the M5 Max (tier_roofline.py,
tier_breakeven.py): 154 MB of recurrent state + 64 KiB of K/V per token per checkpoint, prefill
885 tok/s, HOT clone 0.043 s per 2.1 GB, zstd(+byte planes) ratio 1.46 on K/V and 1.07 on the state,
int8 g32 K/V 0.5625x.

S sessions grow round-robin (random order each round) by STEP tokens per request up to LENGTH, then a
new session replaces the finished one. A request restores the session's latest checkpoint from
wherever it is (HOT, WARM, SSD, gone) and prefills the rest; its new checkpoint supersedes the old
one in HOT. Eviction demotes LRU HOT -> WARM (compact) -> SSD (raw, capped) -> gone.

    tier_capacity_model.py
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass

GIB = 1 << 30
STATE = 154e6
KV_TOK = 65536
PREFILL_TPS = 885.0
CLONE_S_PER_GB = 0.043 / 2.1


def raw_bytes(n: int) -> float:
    return STATE + KV_TOK * n


@dataclass
class Warm:
    name: str
    kv_ratio: float
    state_ratio: float
    decode_gbps: float  # restore throughput on the raw bytes (0 = fixed cost per GB below)
    lossy: bool = False

    def stored(self, n: int) -> float:
        return STATE / self.state_ratio + KV_TOK * n / self.kv_ratio

    def restore_s(self, n: int) -> float:
        return raw_bytes(n) / (self.decode_gbps * 1e9) + raw_bytes(n) / 1e9 * CLONE_S_PER_GB


WARM_LOSSLESS = Warm("warm-lossless", 1.458, 1.068, 6.0)  # 6 GB/s: 17 GB/s decode + unshuffle + copy
WARM_INT8 = Warm("warm-int8", 1.778, 1.0, 40.0, lossy=True)  # GPU dequantize


@dataclass
class Disk:
    name: str
    read_gbps: float
    latency_s: float
    cap_gib: float

    def restore_s(self, n: int) -> float:
        return self.latency_s + raw_bytes(n) / (self.read_gbps * 1e9)


INTERNAL = Disk("internal 10 GB/s", 10.0, 0.0001, 64)
TB4 = Disk("TB4 5 GB/s", 5.0, 0.0001, 64)
HDD = Disk("HDD 0.15 GB/s", 0.15, 0.012, 200)


def run(hot_gib: float, warm_gib: float, warm: Warm | None, disks: list[Disk], sessions: int,
        length: int, step: int = 3000, requests: int = 1500, seed: int = 1):
    rnd = random.Random(seed)
    hot: OrderedDict[int, int] = OrderedDict()  # session id -> checkpoint tokens
    wm: OrderedDict[int, int] = OrderedDict()
    dk: list[OrderedDict[int, int]] = [OrderedDict() for _ in disks]
    lens: dict[int, int] = {}
    nxt = 0
    active = []
    for _ in range(sessions):
        active.append(nxt)
        lens[nxt] = 7500
        nxt += 1
    served = {"hot": 0, "warm": 0, "disk": 0, "cold": 0}
    ttft: list[float] = []
    cached_tok = prompt_tok = 0

    def hot_bytes():
        return sum(raw_bytes(n) for n in hot.values())

    def warm_bytes():
        return sum(warm.stored(n) for n in wm.values()) if warm else 0.0

    def disk_bytes(i):
        return sum(raw_bytes(n) for n in dk[i].values())

    def demote():
        while hot and hot_bytes() > hot_gib * GIB:
            s, n = hot.popitem(last=False)
            put_warm(s, n)

    def put_warm(s, n):
        if warm is not None and warm.stored(n) <= warm_gib * GIB:
            wm[s] = n
            while warm_bytes() > warm_gib * GIB:
                s2, n2 = wm.popitem(last=False)
                if not warm.lossy:
                    put_disk(0, s2, n2)
            if warm.lossy:
                put_disk(0, s, n)  # lossy WARM: the SSD gets the exact copy at demotion
            return
        put_disk(0, s, n)

    def put_disk(i, s, n):
        if i >= len(disks):
            return
        if raw_bytes(n) > disks[i].cap_gib * GIB:
            return
        dk[i][s] = n
        dk[i].move_to_end(s)
        while disk_bytes(i) > disks[i].cap_gib * GIB:
            s2, n2 = dk[i].popitem(last=False)
            put_disk(i + 1, s2, n2)

    for r in range(requests):
        if r % sessions == 0:
            rnd.shuffle(active)
        s = active[r % sessions]
        n_prev = lens[s]
        n = n_prev + step if r >= sessions else n_prev
        prev_ck = n_prev - 1 if r >= sessions else 0
        t = 0.0
        where = "cold"
        if s in hot and hot[s] <= prev_ck + 1:
            where, t = "hot", raw_bytes(hot[s]) / 1e9 * CLONE_S_PER_GB
        elif s in wm:
            where, t = "warm", warm.restore_s(wm[s]) if warm else 0.0
        else:
            for i, d in enumerate(disks):
                if s in dk[i]:
                    where, t = "disk", d.restore_s(dk[i][s])
                    break
        have = {"hot": hot, "warm": wm}.get(where)
        ck = 0
        if where in ("hot", "warm"):
            ck = have[s]
        elif where == "disk":
            ck = next(dk[i][s] for i in range(len(dk)) if s in dk[i])
        # promote: the old checkpoint is superseded by the new one
        hot.pop(s, None)
        wm.pop(s, None)
        for d in dk:
            d.pop(s, None)
        served[where] += 1
        cached_tok += ck
        prompt_tok += n
        ttft.append(t + (n - ck) / PREFILL_TPS)
        hot[s] = n - 1
        hot.move_to_end(s)
        demote()
        lens[s] = n
        if n + step > length:  # session finished: a new one takes its place
            active[active.index(s)] = nxt
            lens[nxt] = 7500
            hot.pop(s, None)
            nxt += 1
    ttft.sort()
    tot = sum(served.values())
    return {
        "hot": served["hot"] / tot,
        "warm": served["warm"] / tot,
        "disk": served["disk"] / tot,
        "cold": served["cold"] / tot,
        "mean": sum(ttft) / len(ttft),
        "p90": ttft[int(0.9 * len(ttft))],
        "cached": cached_tok / prompt_tok,
    }


def main() -> None:
    cfgs = [
        ("RAM only", None, 0.0, []),
        ("warm-lossless, no SSD", WARM_LOSSLESS, 0.4, []),
        ("warm-int8, no SSD", WARM_INT8, 0.4, []),
        ("+ internal SSD", None, 0.0, [INTERNAL]),
        ("+ TB4 SSD", None, 0.0, [TB4]),
        ("internal + warm-lossless", WARM_LOSSLESS, 0.4, [INTERNAL]),
        ("internal + warm-int8", WARM_INT8, 0.4, [INTERNAL]),
        ("TB4 + warm-lossless", WARM_LOSSLESS, 0.4, [TB4]),
        ("TB4 + warm-int8", WARM_INT8, 0.4, [TB4]),
        ("internal(8G) > HDD", None, 0.0, [Disk("internal 10 GB/s", 10.0, 0.0001, 8), HDD]),
        ("HDD only", None, 0.0, [HDD]),
    ]
    for budget in (4.0, 16.0, 32.0):
        for sessions, length in ((3, 30000), (6, 30000), (3, 60000), (6, 60000)):
            print(f"\nAPC RAM {budget:g} GiB, {sessions} sessions to {length // 1000}K tokens")
            print(f"{'config':28} {'hot':>5} {'warm':>5} {'disk':>5} {'cold':>5} {'cached':>7} {'mean s':>7} {'p90 s':>7}")
            for name, warm, share, disks in cfgs:
                w_gib = budget * share if warm else 0.0
                r = run(budget - w_gib, w_gib, warm, disks, sessions, length)
                print(f"{name:28} {r['hot']:5.2f} {r['warm']:5.2f} {r['disk']:5.2f} {r['cold']:5.2f} {r['cached']:7.3f} {r['mean']:7.2f} {r['p90']:7.2f}")


if __name__ == "__main__":
    main()
