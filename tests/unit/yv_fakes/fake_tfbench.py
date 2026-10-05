"""Stand-in for scripts/research/tfbench.py: deterministic records, behaviour set by --env flags."""

import argparse
import hashlib
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--engine")
ap.add_argument("--part")
ap.add_argument("--rep", type=int, default=0)
ap.add_argument("--out")
ap.add_argument("--only-ctx", type=int, action="append")
ap.add_argument("--only-kind", action="append")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--tag", default="")
ap.add_argument("--model")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--decode-tokens", type=int, default=256)
ap.add_argument("--turn2-tokens", type=int, default=0)
ap.add_argument("--long-ask", action="store_true")
a = ap.parse_args()
env = dict(kv.split("=", 1) for kv in a.env)
if env.get("FAKE_CRASH") == "1":
    raise SystemExit("fake server crashed")
if (
    env.get("FAKE_CRASH_REP")
    and int(env["FAKE_CRASH_REP"]) == a.rep
    and a.part == "decode"
    and not a.smoke
):
    raise SystemExit("fake crash on this rep")
spec = "off" if env.get("YUNSHU_VLM_DRAFT", "").lower() in ("off", "none") else "dflash"
ctxs = [512] if a.smoke else a.only_ctx or [1024]
log_dir = os.path.join(os.environ.get("TFB_OUT", "."), "out")
os.makedirs(log_dir, exist_ok=True)
with open(os.path.join(log_dir, f"server-yunshu-decode-{a.rep}{a.tag}.log"), "w") as f:
    f.write(f"VLM batch runner: x draft={spec}\n")
    if env.get("FAKE_PATH") == "on":
        f.write("fast path engaged\n")
with open(a.out, "a") as out:

    def emit(**k):
        out.write(json.dumps(k) + "\n")

    emit(
        part="session",
        engine="yunshu",
        rep=a.rep,
        engaged_spec_mode=spec,
        src=os.environ.get("TFB_YUNSHU_SRC"),
    )
    if a.part == "needle":
        for ctx in ctxs:
            for i in range(10):
                bad = env.get("FAKE_NEEDLE_BAD") and i < int(env["FAKE_NEEDLE_BAD"])
                emit(part="needle", ctx=ctx, item=i, correct=not bad)
        ctxs = []
    if a.part == "conc32":
        slow = 2.0 if env.get("FAKE_SLOW") == "1" else 1.0
        for t in range(2):
            emit(
                part="conc32",
                trial=t,
                ttfts=[3.0 * slow, 3.1 * slow],
                per_req_dec=[40.0 / slow, 41.0 / slow],
                cached=[32768, 32768],
                pts=[34000, 34000],
                cts=[1024, 1024],
                shas=["a", "b"],
            )
        ctxs = []
    for ctx in ctxs:
        for kind in a.only_kind or ("prose", "code"):
            for phase in ("cold", "warm", "turn2"):
                grp = "turn2" if phase == "turn2" else "t1"
                key = f"{ctx}-{kind}-{grp}-{env.get('FAKE_DIVERGE', '')}"
                if phase == "warm" and env.get("FAKE_APC_BAD") == "1":
                    key += "-warm"
                dec = (
                    100.0 + a.rep * 0.2 - (25.0 if env.get("FAKE_SLOW") == "1" else 0.0)
                )
                emit(
                    part="decode",
                    ctx=ctx,
                    kind=kind,
                    phase=phase,
                    ct=a.decode_tokens,
                    pt=ctx,
                    ttft_s=1.0 + ctx / 10000.0,
                    dec_tps=dec,
                    finish="length",
                    cached=0 if phase == "cold" else ctx,
                    sha=hashlib.sha256(key.encode()).hexdigest()[:16],
                    text="hello",
                    xy={"speculative": spec},
                )
    emit(part="part_done", engine="yunshu", which=a.part, rep=a.rep, complete=True)
