"""Evict an APC working set, restart, compare SSD/RAM TTFT and rehearse next turns.

Use gpuq. Prompts/replies stay in memory; result files contain metrics and digests.
Diagnostic spans are collected separately by ttft_warm_probe --spill-context.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def validate(row, tier=None, reference=None):
    if row["ct"] < 1 or row["ttft_s"] <= 0:
        raise ValueError("missing generation")
    actual = (row.get("xy") or {}).get("cache", {}).get("tier")
    if tier and actual != tier:
        raise ValueError(
            f"expected {tier}, got {actual}: cached={row['cached']}/{row['pt']}"
        )
    if tier in ("ram", "ssd") and not row["cached"]:
        raise ValueError("missing cached tokens")
    if reference and row["sha"] != reference["sha"]:
        raise ValueError("cold/RAM/SSD or rehearsal output mismatch")


def run(tf, env, model, tag, contexts, tokens, emit, *, idle=1, spill_context=131072):
    """Transport is injected so CPU fixtures exercise the restart protocol."""
    server = tf.Srv("yunshu", env, tag + "-prime", model=model)
    references, replies = {}, {}

    def request(ctx, phase, body, tier=None, reference=None, **extra):
        row = tf.send(server.url, body)
        text = row.pop("_text")
        row.pop("head", None)
        validate(row, tier, reference)
        emit(dict(ctx=ctx, phase=phase, **extra, **row))
        return row, text

    try:
        for ctx in contexts:
            body = tf.req(server.model, tf.load_prompt(f"prose-{ctx}"), tokens)
            reference, reply = request(ctx, "cold", body, "none")
            references[ctx], replies[ctx] = reference, reply
            request(ctx, "ram_before_restart", body, "ram", reference)
        request(
            spill_context,
            "spill",
            tf.req(server.model, tf.load_prompt(f"prose-{spill_context}"), 1),
        )
        tf_log = server.log.read_text()
        mode = tf.engaged_spec_mode("yunshu", tf_log)
        if mode not in ("dflash", "off"):
            raise ValueError(f"unexpected engaged mode {mode}")
        emit(dict(event="server", phase="prime", engaged=mode))
        server.kill()
        server = tf.Srv("yunshu", env, tag + "-restart", model=model)
        for ctx in contexts:
            body = tf.req(server.model, tf.load_prompt(f"prose-{ctx}"), tokens)
            request(ctx, "ssd", body, "ssd", references[ctx])
            request(ctx, "ram_after_restart", body, "ram", references[ctx])
            followup = dict(
                body,
                messages=body["messages"]
                + [
                    {"role": "assistant", "content": replies[ctx]},
                    {
                        "role": "user",
                        "content": "Continue with the next part, at the same length.",
                    },
                ],
            )
            time.sleep(idle)
            plain, _ = request(ctx, "followup", followup)
            rehearsal = tf.rehearse(server, followup)
            request(ctx, "followup_rehearsed", followup, reference=plain, **rehearsal)
        log = server.log.read_text()
        mode = tf.engaged_spec_mode("yunshu", log)
        if mode not in ("dflash", "off"):
            raise ValueError(f"unexpected engaged mode {mode}")
        emit(
            dict(
                event="server",
                phase="restart",
                engaged=mode,
                drafter_restores=log.count(
                    "DFlash drafter window restored from APC hit"
                ),
            )
        )
        emit(dict(complete=True))
    finally:
        server.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 16384, 32768])
    parser.add_argument("--spill-context", type=int, default=131072)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument(
        "--cleanup-apc",
        action="store_true",
        help="remove this arm's explicitly configured scratch APC dir on exit",
    )
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    env = dict(item.split("=", 1) for item in args.env)
    if not env.get("YUNSHU_VLM_APC_DISK_DIR"):
        parser.error("each arm requires its own explicit APC disk dir")
    os.environ["TFB_YUNSHU_SRC"] = args.src
    os.environ["TFB_OUT"] = str(args.out.parent / args.tag)
    os.environ["TFB_PORT_LAST"] = "18996"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tfbench

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:

        def emit(row):
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(
                {
                    k: row.get(k)
                    for k in ("ctx", "phase", "cached", "ttft_s", "complete")
                },
                flush=True,
            )

        try:
            run(
                tfbench,
                env,
                args.model,
                args.tag,
                args.contexts,
                args.tokens,
                emit,
                spill_context=args.spill_context,
            )
        finally:
            if args.cleanup_apc:
                root = Path(env["YUNSHU_VLM_APC_DISK_DIR"])
                # Only bounded worker scratch roots; never remove a user's cache.
                if root.parent == Path(
                    "/Volumes/Micron/yunshu-apc"
                ) and root.name.startswith("ssdttft-"):
                    shutil.rmtree(root, ignore_errors=True)
                    shutil.rmtree(str(root) + ".dflash-window", ignore_errors=True)


if __name__ == "__main__":
    main()
