# Native tool formats and structural tags

Yunshu detects the wire format from the model's chat template. Independent readers
cover DeepSeek V3/R1/V3.1 envelopes, DeepSeek V3.2 `function_calls` DSML and V4
`tool_calls` DSML, Harmony tool messages, Hermes JSON, Llama JSON and literal Python
calls, Mistral JSON / `[ARGS]`, GLM newline JSON / key-value tags, and Kimi K2.
Existing MLX readers continue to cover other formats. Python calls are never executed.
Tool names are filtered against the request; each call gets its own ID in streaming.
Arguments are buffered until a complete call is available, then sent after the name.

Forced `tool_choice` uses a native llguidance grammar. Native envelopes can consist
of multiple tokenizer tokens. The XML-like and Python parameter grammars use schema order;
Python optional parameters can be included or omitted. Unsupported grammar/tokenizer combinations
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
Triggers that combine special-token atoms with ordinary text are rejected at
tokenizer binding; use a text-only trigger or a whole special token. Prefixes
of a registered begin marker are supported. Both text and VLM paths use llguidance; tokenizer special reasoning markers are allowed
outside structures. Auto tools may be combined with structural tags. Forced tool choices with
structural tags, and VLM tools with ordinary JSON/CFG constraints, remain
rejected so their independent contracts cannot silently override each other.
Lazy tags preserve model thinking defaults and thinking budgets.

Fixture shapes are inspired by vLLM's `tests/tool_parsers` under Apache-2.0,
Copyright contributors to the vLLM project. Values and all parser/test implementation
are independent; the reviewed upstream revision and files are recorded in `vendor.json`.
Harmony follows OpenAI's published Harmony message format. Structural tags use the
installed MIT-licensed llguidance `StructTag` API. No upstream implementation was copied.

The optional capability stage runs four real HTTP checks (forced, streaming, lazy
output and lazy auto-tools). Run `scripts/dev/yv ab --base BASE_SHA --cand CAND_SHA
--suite preflight,toolparse --model /path/to/small-model --label toolparse-smoke-topic
--priority -1 --detach`; it is not added to existing release/performance ladders.
The lazy-output prompt asks for Tokyo while the hidden schema forces Taipei,
so the evidence must show the constraint engaged rather than natural adherence.
