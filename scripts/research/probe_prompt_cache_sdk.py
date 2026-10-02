"""27B official SDK usage/identity checks. Run through gpuq only."""

import argparse
import json
import time
from pathlib import Path

import anthropic
import openai
import tfbench as t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ctx", type=int, default=8192)
    a = p.parse_args()
    from cache_probe_source import freeze

    source = freeze(a.out, t)
    s = t.Srv(
        "yunshu", {"YUNSHU_VLM_APC_DISK": "0"}, "cachesf-sdk-" + str(time.time_ns())
    )
    records = []
    try:
        oc = openai.OpenAI(base_url=s.url + "/v1", api_key="k", timeout=900)
        ac = anthropic.Anthropic(base_url=s.url, api_key="k", timeout=900)
        oc.chat.completions.create(
            model=s.model,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=4,
            temperature=0,
        )
        prefix = t.load_prompt(f"prose-{a.ctx}")
        for kind in ("system", "message", "tools"):
            body = {
                "model": s.model,
                "max_tokens": 4,
                "extra_body": {"temperature": 0},
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "Answer with OK."}],
            }
            if kind == "system":
                body["system"] = [
                    {
                        "type": "text",
                        "text": "System check. " + prefix,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif kind == "message":
                body["messages"][0]["content"] = [
                    {
                        "type": "text",
                        "text": "Message check. " + prefix,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "\nAnswer with OK."},
                ]
            else:
                body["tools"] = [
                    {
                        "name": "lookup",
                        "description": "Tool check. " + prefix,
                        "input_schema": {"type": "object", "properties": {}},
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            pair = []
            for repeat in range(2):
                start = time.perf_counter()
                r = ac.messages.create(**body)
                rec = {
                    "kind": kind,
                    "repeat": repeat,
                    "seconds": time.perf_counter() - start,
                    "usage": r.usage.model_dump(),
                    "content": [b.model_dump() for b in r.content],
                }
                print(json.dumps(rec), flush=True)
                records.append(rec)
                pair.append(r.usage)
            assert pair[0].cache_creation_input_tokens > 0, (kind, pair)
            assert (
                pair[1].cache_read_input_tokens == pair[0].cache_creation_input_tokens
            ), (kind, pair)
            assert pair[1].cache_creation_input_tokens == 0, (kind, pair)
        body = {
            "model": s.model,
            "messages": [{"role": "user", "content": prefix + "\nOpenAI usage check."}],
            "max_tokens": 4,
            "temperature": 0,
            "logprobs": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        pair = [oc.chat.completions.create(**body) for _ in range(2)]
        assert pair[1].usage.prompt_tokens_details.cached_tokens > 0
        assert pair[0].choices[0].logprobs == pair[1].choices[0].logprobs, (
            "OpenAI APC hit/miss logprobs drift"
        )
        records.append(
            {
                "kind": "openai",
                "usage": [r.usage.model_dump() for r in pair],
                "identity": True,
            }
        )
    finally:
        s.kill()
    log = s.log.read_text(errors="replace")
    assert "Prompt cache rendered token boundaries:" in log
    records.append(
        {
            "complete": True,
            "server_log": str(s.log),
            "source": source,
        }
    )
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    main()
