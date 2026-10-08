# Local Evals

The OpenAI-compatible Evals API stores definitions, runs and output items locally.
Use the OpenAI SDK with Yunshu's `/v1` base URL. Twelve routes cover eval CRUD,
run create/list/retrieve/cancel/delete and output-item list/retrieve.

## Grade a supplied response

This example uses a lexical grader and does not invoke a model:

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
evaluation = client.evals.create(
    name="exact-answer",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "include_sample_schema": True,
    },
    testing_criteria=[{
        "type": "string_check", "name": "exact", "operation": "eq",
        "input": "{{sample.output_text}}", "reference": "{{item.answer}}",
    }],
)
run = client.evals.runs.create(
    evaluation.id,
    data_source={
        "type": "jsonl",
        "source": {"type": "file_content", "content": [{
            "item": {"answer": "hello"}, "sample": {"output_text": "hello"},
        }]},
    },
)
while run.status not in {"completed", "failed", "canceled"}:
    time.sleep(0.2)
    run = client.evals.runs.retrieve(run.id, eval_id=evaluation.id)
print(run.status)
print(client.evals.runs.output_items.list(run.id, eval_id=evaluation.id))
```

## Sources and graders

Runs accept inline rows, Files API JSONL (`file_id`), or stored chat completions filtered
by model, metadata and inclusive creation timestamps. Set `store=True` when creating
chat completions to make them available to this source. A `completions` run can sample
the local model using message templates or an item reference; rows are validated against
the eval schema before scheduling. The maximum is 10,000 rows per run.

Supported graders are `string_check`, `text_similarity`, `score_model` and `label_model`.
Lexical grading runs on a dedicated CPU worker; model sampling and model graders use
normal local inference. Unsupported graders return 400. This API does not provide
hosted Python execution or a cloud dashboard (`report_url` is empty).

## Persistence and cancellation

`YUNSHU_EVALS_DIR` defaults to `~/.yunshu/evals`. Atomic JSON files retain definitions,
run snapshots and completed output items. Credentials remain in memory. Cancel a run
with `client.evals.runs.cancel(run.id, eval_id=evaluation.id)`; cancellation stops active
inference/CPU grading and preserves completed items. Restart marks interrupted runs
failed instead of replaying them. Deleting an eval also deletes its runs.

List pages support cursors, order and limits from 1 to 100. Inspect output items to
understand failures; a submitted run is not evidence of a successful evaluation.
See [API surface](API_SURFACE.md) for implementation and real-server evidence.
