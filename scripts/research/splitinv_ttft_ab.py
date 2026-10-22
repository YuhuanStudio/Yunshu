"""Three interleaved same-checkpoint 32K follow-up A/Bs. Submit with gpuq --quiet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-tree", type=Path, required=True)
    p.add_argument("--cand-tree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--reply-tokens", type=int, nargs="+", default=[256, 512, 2048])
    p.add_argument("--kinds", nargs="+", default=["code", "prose"])
    p.add_argument(
        "--arms",
        nargs="+",
        default=["base", "cand", "cand-r1", "base-r1", "base-r2", "cand-r2"],
    )
    p.add_argument("--idle-s", type=float, default=8)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    import tfbench as t
    from splitinv_rehearsal import metrics

    a.out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with a.out.open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    failures = []
    for arm in a.arms:
        tree = a.base_tree if arm.startswith("base") else a.cand_tree
        t.YUNSHU_SRC = str(tree / "python")
        server = t.start_server(
            "yunshu",
            {
                "YUNSHU_VLM_DRAFT": t.D,
                "YUNSHU_NEXT_TURN_PREFILL": "stride",
                "YUNSHU_VLM_APC_DISK": "0",
                "YUNSHU_VLM_APC_DISK_DIR": f"/Volumes/Micron/yunshu-apc/splitinv-ttft-{arm}",
            },
            f"splitinv-ttft-{a.ctx}-{arm}",
        )
        try:
            for kind in a.kinds:
                prompt = t.decode_prompt(kind, a.ctx, True)
                for arrival in ["after", "immediate"]:
                    for tokens in a.reply_tokens:
                        text = (
                            prompt
                            + f"\nResponse budget: {tokens} tokens. Arrival case: {arrival}."
                        )
                        body = t.req(server.model, text, tokens)
                        before = metrics(t, server)
                        first = t.send(server.url, body)
                        t.check_decode_len(
                            first, tokens, f"{arm}/{kind}/{tokens}/{arrival}"
                        )
                        follow = t.req(server.model, text, 16)
                        follow["messages"] += [
                            {"role": "assistant", "content": first["_text"].strip()},
                            {
                                "role": "user",
                                "content": "Continue with the next part, at the same length.",
                            },
                        ]
                        if arrival == "after":
                            time.sleep(a.idle_s)
                        result = t.send(server.url, follow)
                        completed = metrics(t, server) - before
                        if not result["cached"]:
                            raise RuntimeError("follow-up did not hit APC")
                        if (
                            arm.startswith("cand")
                            and arrival == "after"
                            and completed < 1
                        ):
                            raise RuntimeError("candidate rehearsal did not complete")
                        for label, row in [("reply", first), ("turn2", result)]:
                            row.pop("_text", None)
                            emit(
                                dict(
                                    arm=arm,
                                    ctx=a.ctx,
                                    kind=kind,
                                    reply_tokens=tokens,
                                    arrival=arrival,
                                    phase=label,
                                    rehearsals_completed=completed,
                                    mode=server.engaged_spec_mode,
                                    **row,
                                )
                            )
            server.verify_spec_mode()
            emit(dict(arm=arm, arm_complete=True, mode=server.engaged_spec_mode))
        except Exception as error:
            failures.append(arm)
            emit(dict(arm=arm, error=repr(error)))
        finally:
            server.kill()
    emit(dict(complete=True, failures=failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
