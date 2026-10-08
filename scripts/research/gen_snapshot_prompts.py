"""Create one immutable, exactly tokenized cross-engine corpus on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

from snapshot_prompts import MODEL, assert_prompt, exact_prompt
from tfbench import LONG_ASK


def generate(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for ctx in (1024, 8192, 32768, 65536, 131072):
        for kind in ("prose", "code"):
            original = (source / f"{kind}-{ctx}.txt").read_text()
            split = original.rfind("\n\n---\n")
            if split < 0:
                raise ValueError("missing corpus instruction boundary")
            body, ask = original[:split], original[split:]
            text = exact_prompt(body * 2, ctx, ask + LONG_ASK)
            count = assert_prompt(text, ctx)
            (destination / f"{kind}-{ctx}.txt").write_text(text)
            records.append(
                {
                    "kind": kind,
                    "ctx": ctx,
                    "content_tokens": count,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            print(json.dumps(records[-1]), flush=True)
    for kind in ("prose", "code"):
        body = (source / f"{kind}-32768.txt").read_text()
        split = body.rfind("\n\n---\n")
        for n in (2, 4):
            for trial in range(2):
                for i in range(n):
                    header = f"Concurrent cohort n={n} trial={trial} request={i}.\n"
                    text = exact_prompt(
                        header + body[:split] * 2, 32768, body[split:] + LONG_ASK
                    )
                    path = destination / f"conc-{n}-{trial}-{kind}-{i}.txt"
                    path.write_text(text)
                    records.append(
                        {
                            "kind": kind,
                            "ctx": 32768,
                            "cohort_n": n,
                            "trial": trial,
                            "concurrent_variant": i,
                            "content_tokens": assert_prompt(text, 32768),
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
    manifest = {
        "complete": True,
        "checkpoint": str(MODEL),
        "tokenizer_sha256": hashlib.sha256(
            (MODEL / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "token_budget": "content including instructions; chat-template overhead recorded separately",
        "prompts": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/Volumes/P5Plus/yunshu-build/snapshot014/corpus/prompts"),
    )
    args = parser.parse_args()
    generate(args.source, args.destination)
