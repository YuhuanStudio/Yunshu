# Native tool formats and structural tags

Yunshu detects the wire format from the model's chat template. Independent readers
cover DeepSeek V3/R1/V3.1 envelopes, DeepSeek V3.2 `function_calls` DSML and V4
`tool_calls` DSML, Harmony tool messages, Hermes JSON, Llama JSON and literal Python
calls, Mistral JSON / `[ARGS]`, GLM newline JSON / key-value tags, and Kimi K2.
Existing MLX readers continue to cover other formats. Python calls are never executed.
Tool names are filtered against the request; each call gets its own ID in streaming.
Arguments are buffered until a complete call is available, then sent after the name.

Forced `tool_choice` uses a native llguidance grammar. Native envelopes can consist
of multiple tokenizer tokens. The XML-like parameter grammars use schema order;
Python forced calls omit optional parameters. Unsupported grammar/tokenizer combinations
retain the existing unconstrained fallback (a parsed call is not a strict-schema proof).
New families without locally available checkpoints are tested with fixtures, not claimed
as real-model validated. See the worker report for the checkpoints actually exercised.

```json
{
  "response_format": {
    "type": "structural_tag",
    "structures": [
      {
        "begin": "<result>",
        "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "end": "</result>"
      }
    ],
    "triggers": ["<result>"]
  }
}
```

This legacy OpenAI/vLLM tags-and-triggers shape is also accepted as
`structured_outputs.structural_tag` (object or JSON string). Reasoning and ordinary
text stay free until a trigger; triggered bodies follow their JSON schema and closing
tag. A trigger is optional, so applications that require a structure must check it
appeared. Every trigger must have a matching structure, with unambiguous prefixes.
Both text and VLM paths use llguidance; tokenizer special reasoning markers are allowed
outside structures. Combining tools with response_format still follows the gateway's
existing conflict policy.

Fixture shapes are inspired by vLLM's `tests/tool_parsers` under Apache-2.0,
Copyright contributors to the vLLM project. Values and all parser/test implementation
are independent; the reviewed upstream revision and files are recorded in `vendor.json`.
Harmony follows OpenAI's published Harmony message format. Structural tags use the
installed MIT-licensed llguidance `StructTag` API. No upstream implementation was copied.
