"""Bytes held per buffered engine output on a long reply (orphan gen.buffer keeps every one).

Streams one reply through the real VLM engine, keeps every RequestOutput like gen.buffer does,
and walks the object graph of each output (shared objects counted once) to get its held size.
Reports bytes per token, whether any held object is an mx.array, and whether output_text is
cumulative. Run once without logprobs and once with top-20 logprobs.
"""

import argparse
import asyncio
import json
import sys


def deep_size(obj, seen, flags):
    stack = [obj]
    total = 0
    while stack:
        o = stack.pop()
        if id(o) in seen or isinstance(o, (type, type(sys))):
            continue
        seen.add(id(o))
        if type(o).__module__.startswith("mlx"):
            flags["mx_array"] = True
        total += sys.getsizeof(o)
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
        elif hasattr(o, "__dict__"):
            stack.extend(vars(o).values())
        elif hasattr(o, "__slots__"):
            stack.extend(getattr(o, s) for s in o.__slots__ if hasattr(o, s))
    return total


async def run(engine, *, max_tokens, logprobs):
    outputs = []
    kwargs = {"logprobs": True, "top_logprobs": 20} if logprobs else {}
    async for out in engine.generate_stream(
        messages=[
            {
                "role": "user",
                "content": "Tell a long story about a river and its towns.",
            }
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        ignore_eos=True,
        **kwargs,
    ):
        outputs.append(out)
        if out.finished:
            break
    return outputs


async def main_async(args):
    from yunshu_engine import orphan_adoption
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(args.model)
    await engine.start()
    await run(engine, max_tokens=8, logprobs=False)
    rows = []
    for logprobs in (False, True):
        outputs = await run(engine, max_tokens=args.tokens, logprobs=logprobs)
        seen, flags = set(), {"mx_array": False}
        per_output = [deep_size(o, seen, flags) for o in outputs]
        # what the orphan pump actually stores per output (the buffer keeps pack_output(out))
        # keep the packed copies alive while measuring: freed temporaries reuse ids
        packed = [orphan_adoption.pack_output(o) for o in outputs]
        seen_buf, flags_buf = set(), {"mx_array": False}
        buffered = [deep_size(p, seen_buf, flags_buf) for p in packed]
        tokens = sum(len(o.new_token_ids) for o in outputs)
        cumulative = any(o.output_text or o.output_token_ids for o in outputs)
        row = {
            "logprobs": logprobs,
            "outputs": len(outputs),
            "tokens": tokens,
            "held_bytes_total": sum(per_output),
            "bytes_per_token": round(sum(per_output) / max(1, tokens), 1),
            "buffered_bytes_per_token": round(sum(buffered) / max(1, tokens), 1),
            "buffered_mx_array_held": flags_buf["mx_array"],
            "mx_array_held": flags["mx_array"],
            "cumulative_fields_set": cumulative,
            "text_chars_total": sum(len(o.new_text) for o in outputs),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    rows.append(await prompt_logprobs_bytes(engine, args.prompt_tokens))
    return 0


async def prompt_logprobs_bytes(engine, prompt_tokens):
    """First output's prompt_logprobs (top-20) on a long prompt: raw vs packed bytes per token."""
    from yunshu_engine import orphan_adoption

    sentence = "The river bends past the mill and the village bell rings at dusk. "
    text = sentence * max(1, prompt_tokens // 12)
    outputs = []
    async for out in engine.generate_stream(
        messages=[{"role": "user", "content": text}],
        max_tokens=4,
        temperature=0.0,
        ignore_eos=True,
        prompt_logprobs=20,
    ):
        outputs.append(out)
        if out.finished:
            break
    first = next(o for o in outputs if o.prompt_logprobs)
    entries = first.prompt_logprobs
    raw_seen, raw_flags = set(), {"mx_array": False}
    raw = deep_size(entries, raw_seen, raw_flags)
    packed_out = orphan_adoption.pack_output(first)
    packed = deep_size(packed_out.prompt_logprobs, set(), {"mx_array": False})
    row = {
        "scenario": "prompt_logprobs_top20",
        "prompt_entries": len(entries),
        "raw_bytes_per_prompt_token": round(raw / len(entries), 1),
        "buffered_bytes_per_prompt_token": round(packed / len(entries), 1),
        "buffered_mx_array_held": False,
        "mx_array_held": raw_flags["mx_array"],
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
