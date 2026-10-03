# Chat-template regression fixtures

CPU-only snapshots; no tokenizer vocabulary or model weights are included.

- `qwen35.jinja`: local `Qwen3.5-0.8B-MLX-bf16/chat_template.jinja` (Qwen3.5 template).
- `qwen38.jinja`: local `Jundot/Qwen3.8-27B-oQ4e-mtp/chat_template.jinja` (Qwen3.8 template).
- `glm53.jinja`: zai-org/GLM-5.3-Flash `chat_template.jinja`, HF revision
  `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`;
  https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/chat_template.jinja

Upstream authors retain ownership of the templates. Fixtures exercise reasoning and
empty tool history through both API conversions without loading a checkpoint.
