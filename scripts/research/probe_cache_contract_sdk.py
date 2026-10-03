"""SDK cache contract, stream accounting and arbitrary-seam identity (gpuq only)."""

import argparse
import json
import time
from pathlib import Path

import anthropic
import openai
import tfbench as t
from cache_probe_source import freeze
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source = freeze(a.out, t)
    records = []
    server = t.Srv(
        "yunshu",
        {"YUNSHU_VLM_APC_DISK": "0", "YUNSHU_VLM_DRAFT": t.D},
        "cachesf-contract-" + str(time.time_ns()),
    )
    try:
        ac = anthropic.Anthropic(base_url=server.url, api_key="k", timeout=900)
        oc = openai.OpenAI(base_url=server.url + "/v1", api_key="k", timeout=900)
        prefix = t.load_prompt("prose-8192")
        stream_body = {
            "model": server.model,
            "max_tokens": 4,
            "thinking": {"type": "disabled"},
            "system": [
                {
                    "type": "text",
                    "text": "Stream contract. " + prefix + "  ",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "extra_body": {"temperature": 0},
        }
        tokenizer = AutoTokenizer.from_pretrained(t.M, local_files_only=True)
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": stream_body["system"][0]["text"]},
                {"role": "user", "content": "Reply with OK."},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        expected_prompt = len(tokenizer.encode(rendered, add_special_tokens=False))
        usages = []
        for rep in range(2):
            with ac.messages.stream(**stream_body) as stream:
                result = stream.get_final_message()
            u = result.usage.model_dump()
            prompt = expected_prompt
            assert (
                u["input_tokens"]
                + u["cache_creation_input_tokens"]
                + u["cache_read_input_tokens"]
                == prompt
            ), (u, prompt)
            records.append({"kind": "anthropic-stream", "rep": rep, "usage": u})
            usages.append(u)
        assert usages[0]["cache_creation_input_tokens"] > 0
        assert (
            usages[1]["cache_read_input_tokens"]
            == usages[0]["cache_creation_input_tokens"]
        )
        assert usages[1]["cache_creation_input_tokens"] == 0
        body = {
            "model": server.model,
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": True,
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "OpenAI exact seam. " + prefix,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {"type": "text", "text": "\nReply with one short sentence."},
                    ],
                }
            ],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        pair = [oc.chat.completions.create(**body) for _ in range(2)]
        read = pair[1].usage.prompt_tokens_details.cached_tokens
        assert read > 8192
        assert pair[0].choices[0].logprobs == pair[1].choices[0].logprobs, (
            "explicit hit/miss logprob drift"
        )
        records.append(
            {
                "kind": "openai-explicit",
                "usage": [r.usage.model_dump() for r in pair],
                "identity": True,
                "cached": read,
            }
        )
        body["messages"][0]["content"][1]["text"] = "\nReply with two short sentences."
        changed = oc.chat.completions.create(**body)
        assert changed.usage.prompt_tokens_details.cached_tokens == read
        records.append(
            {"kind": "openai-changed-suffix", "usage": changed.usage.model_dump()}
        )
        # No breakpoint in explicit-only mode must neither read nor write APC.
        for block in body["messages"][0]["content"]:
            block.pop("prompt_cache_breakpoint", None)
        no_cache = oc.chat.completions.create(**body)
        assert no_cache.usage.prompt_tokens_details.cached_tokens == 0
        records.append(
            {
                "kind": "openai-explicit-no-breakpoint",
                "usage": no_cache.usage.model_dump(),
            }
        )
        # Responses content conversion must carry the same explicit block metadata.
        rb = {
            "model": server.model,
            "max_output_tokens": 4,
            "temperature": 0,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Responses exact seam. " + prefix,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {"type": "input_text", "text": "\nReply with OK."},
                    ],
                }
            ],
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "extra_body": {"enable_thinking": False},
        }
        responses = [oc.responses.create(**rb) for _ in range(2)]
        assert responses[1].usage.input_tokens_details.cached_tokens > 8192
        records.append(
            {
                "kind": "responses-explicit",
                "usage": [r.usage.model_dump() for r in responses],
            }
        )
    finally:
        server.kill()
    log = server.log.read_text(errors="replace")
    assert "Speculative decoding: dflash" in log
    assert "APC prefix-invariant dispatch engaged:" in log
    records.append({"complete": True, "source": source, "server_log": str(server.log)})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in records))
    for r in records:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
