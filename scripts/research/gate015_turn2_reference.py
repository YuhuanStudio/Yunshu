"""Cold replay of the release gate's identical follow-up (run only via gpuq)."""

import argparse
import hashlib
import json
from pathlib import Path


def followup_body(bench, model, prompt, rows):
    cold = [r for r in rows if r.get("part") == "decode" and r.get("phase") == "cold"]
    if len(cold) != 1 or cold[0].get("ct") != 2048 or cold[0].get("finish") != "length":
        raise ValueError("missing or incomplete release cold reply")
    reply = cold[0].get("text")
    if not isinstance(reply, str) or not reply:
        raise ValueError("missing release reply text")
    body = bench.req(model, prompt, 256)
    body["messages"] += [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "Continue with the next part, at the same length."},
    ]
    return body


def validate(result):
    if result.get("ct") != 256 or result.get("finish") != "length":
        raise ValueError("incomplete cold follow-up")
    if result.get("cached"):
        raise ValueError("cold replay unexpectedly hit cache")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--base-tree", type=Path, required=True)
    ap.add_argument("--cand-tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import tfbench as t

    rows = [json.loads(line) for line in a.fixture.read_text().splitlines()]
    prompt = t.decode_prompt("prose", 65536, True)
    body = followup_body(t, t.M, prompt, rows)
    request_sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    for tree in (a.base_tree, a.cand_tree):
        if not (tree / "python/yunshu_engine/vlm_engine.py").is_file():
            raise ValueError(f"missing pinned tree {tree}")
    if a.dry_run:
        print(json.dumps(dict(complete="dry-run", request_sha=request_sha)))
        return
    arms = [
        ("base-off", a.base_tree, "off", t.D, None),
        ("cand-off", a.cand_tree, "off", t.D, None),
        ("cand-boundary", a.cand_tree, "boundary", t.D, None),
        ("cand-serial", a.cand_tree, "off", "off", "0"),
    ]
    with a.out.open("x") as out:
        for name, tree, mode, draft, memory in arms:
            t.YUNSHU_SRC = str(tree / "python")
            env = {
                "YUNSHU_NEXT_TURN_PREFILL": mode,
                "YUNSHU_VLM_DRAFT": draft,
                "YUNSHU_VLM_APC_DISK": "0",
            }
            if memory is not None:
                env["YUNSHU_VLM_APC_MEMORY_GB"] = memory
            server = t.start_server("yunshu", env, f"gate015-cold-{name}")
            try:
                result = t.send(server.url, body)
                validate(result)
                server.verify_spec_mode()
                result.pop("_text")
                row = dict(
                    arm=name,
                    request_sha=request_sha,
                    mode=server.engaged_spec_mode,
                    **result,
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)
            finally:
                server.kill()
        out.write(
            json.dumps(dict(complete=True, arms=len(arms), request_sha=request_sha))
            + "\n"
        )


if __name__ == "__main__":
    main()
