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
import urllib.request
from pathlib import Path


def disk_inventory(root):
    """Checkpoint sizes/counts only; never persist token IDs or prompt text."""
    entries = []
    for path in Path(root).rglob("exact_*.safetensors"):
        try:
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= 16 << 20:
                    continue
                metadata = json.loads(stream.read(size)).get("__metadata__", {})
            count = sum(bool(x) for x in metadata.get("token_ids", "").split(","))
            entries.append({"tokens": count, "bytes": path.stat().st_size})
        except (OSError, ValueError):
            continue  # an asynchronous writer may not have published its header yet
    return entries


def ensure_spilled(
    root, expected, fill, *, wait_s=3, max_fill=8, inventory=disk_inventory
):
    """Require the measured checkpoints on disk; bounded 32K fillers force LRU eviction."""
    expected = set(expected)
    for attempt in range(max_fill + 1):
        deadline = time.monotonic() + wait_s
        while True:
            entries = inventory(root)
            if expected <= {r["tokens"] for r in entries}:
                return entries
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        if attempt == max_fill:
            raise ValueError(
                f"eviction did not persist measured checkpoints: {sorted(expected)}"
            )
        fill(attempt)


def cleanup_scratch(root):
    path = Path(root)
    if path.parent == Path("/Volumes/Micron/yunshu-apc") and path.name.startswith(
        "ssdttft-"
    ):
        shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(str(path) + ".dflash-window", ignore_errors=True)


def rehearsal_completed(url):
    request = urllib.request.Request(
        url + "/metrics", headers={"Authorization": "Bearer k"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode()
    for line in text.splitlines():
        if (
            line.startswith("yunshu_next_turn_rehearsals_total{")
            and 'outcome="completed"' in line
        ):
            return int(float(line.rsplit(" ", 1)[1]))
    return 0


def validate_next_turn(source, followup, completed_delta):
    if source["ct"] < 2048 or completed_delta < 1:
        raise ValueError("post-restart stride rehearsal did not engage")
    if followup["cached"] <= source["pt"]:
        raise ValueError("followup did not reuse the rehearsed reply")


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


def run(
    tf,
    env,
    model,
    tag,
    contexts,
    tokens,
    emit,
    *,
    idle=1,
    spill_context=131072,
    check_next_turn=False,
):
    """Transport is injected so CPU fixtures exercise the restart protocol."""
    server = tf.Srv("yunshu", env, tag + "-prime", model=model)
    references, replies, expected = {}, {}, {}

    def request(ctx, phase, body, tier=None, reference=None, **extra):
        row = tf.send(server.url, body)
        text = row.pop("_text")
        row.pop("head", None)
        emit(dict(ctx=ctx, phase=phase, **extra, **row))
        validate(row, tier, reference)
        return row, text

    try:
        for ctx in contexts:
            body = tf.req(server.model, tf.load_prompt(f"prose-{ctx}"), tokens)
            reference, reply = request(ctx, "cold", body, "none")
            references[ctx], replies[ctx] = reference, reply
            warm, _ = request(ctx, "ram_before_restart", body, "ram", reference)
            expected[ctx] = warm["cached"]
        root = env.get("YUNSHU_VLM_APC_DISK_DIR")
        if root:
            emit(
                dict(
                    event="disk_inventory",
                    phase="before_spill",
                    entries=disk_inventory(root),
                )
            )
        request(
            spill_context,
            "spill",
            tf.req(server.model, tf.load_prompt(f"prose-{spill_context}"), 1),
        )
        if root:

            def fill(index):
                prompt = f"Independent eviction session {index}.\n" + tf.load_prompt(
                    "prose-32768"
                )
                request(32768, "eviction_fill", tf.req(server.model, prompt, 1))

            entries = ensure_spilled(root, expected.values(), fill)
            emit(dict(event="disk_inventory", phase="after_spill", entries=entries))
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
        if check_next_turn:
            ctx = max(contexts)
            source_body = tf.req(
                server.model, tf.load_prompt(f"prose-{ctx}") + tf.LONG_ASK, 2304
            )
            source, reply = request(ctx, "rehearsal_source", source_body)
            before = rehearsal_completed(server.url)
            deadline = time.monotonic() + 20
            completed = before
            while completed <= before and time.monotonic() < deadline:
                time.sleep(0.2)
                completed = rehearsal_completed(server.url)
            followup_body = tf.req(server.model, "", tokens)
            followup_body["messages"] = source_body["messages"] + [
                {"role": "assistant", "content": reply.strip()},
                {
                    "role": "user",
                    "content": "Continue with the next part, at the same length.",
                },
            ]
            followup, _ = request(ctx, "next_turn", followup_body)
            validate_next_turn(source, followup, completed - before)
            emit(
                dict(
                    event="next_turn_proof",
                    completed_delta=completed - before,
                    cached=followup["cached"],
                    source_prompt_tokens=source["pt"],
                )
            )
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
    parser.add_argument(
        "--check-next-turn",
        action="store_true",
        help="require an automatic stride rehearsal and reply-prefix reuse after restart",
    )
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
                check_next_turn=args.check_next_turn,
            )
        finally:
            if args.cleanup_apc:
                cleanup_scratch(env["YUNSHU_VLM_APC_DISK_DIR"])


if __name__ == "__main__":
    main()
