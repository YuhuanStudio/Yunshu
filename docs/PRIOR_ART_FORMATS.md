# Published retrieval formats

EmbeddingGemma 2 bf16 and affine 4bit checkpoints use the maintained
`mlx_vlm.embedding_loader.load_embedding_model` when its model module is installed.
mlx-vlm 0.7.6 does not include that module; Yunshu carries the MIT port from
`pc/embeddinggemma-2` at `3d87e88402f307efbf68e568971aa887ee7d9ed0` and a derived
strict loader. Quantized modules are constructed from checkpoint scales; vision
and audio towers can retain floating weights. The original float32 loader remains
available for unquantized checkpoints. Integer packed tensors are never cast to
float. See vendor.json for provenance.

Single-label BERT and XLM-RoBERTa use mlx-vlm’s maintained sequence-classification
loader, including published BGE converters with bare backbone names.
Sequence-classification checkpoints for BERT, RoBERTa and XLM-RoBERTa may contain
quantized encoder layers with floating or quantized trained classifier heads.
Missing backbone or head tensors remain a load error. Qwen3 yes/no rerankers use
the official instruction/prefix/suffix recipe and mlx-lm's quantized loader.
`JinaForRanking` (Jina v3) is a separate architecture and remains unsupported:
its trained ranking head must not be silently replaced by Qwen yes/no logits.

Qwen3-VL retrieval delegates to the installed mlx-embeddings `model.process` API.
No GPL source is copied into Yunshu. Empty embedding input returns an empty list;
adding a default instruction does not mutate caller input dictionaries.

## Verification

`scripts/dev/yv ab --base <sha> --cand <sha> --suite embedding --label priorfix-...`
loads bf16, 4bit and a real bf16 checkpoint repacked into two shards, compares
Yunshu's wrapper with direct upstream model invocation on M5, and applies the
0.99985 fp32 cosine floor to the 33 bf16 vectors. Quantization accuracy is reported
separately. `--suite priorart` runs bounded retrieval, Omni prepared-input and
fixed-seed diffusion comparisons. These are modality-specific audits, excluded
from the generic LLM full suite. A diffusion comparator records output differences
and unqualified durations; those durations do not decide performance or defaults.

## 0.1.4 shard impact

The 2026-10-08 HF inventory lists one `model.safetensors` for Google
EmbeddingGemma 2 and each of the eight mlx-community formats (bf16, affine
4/5/6/8bit, MXFP4/MXFP8 and NVFP4). Every converted index maps to that same single
file. Thus the prior raw-loader multi-shard defect was not reachable with those
published snapshots. Repacked/private multi-shard checkpoints could reach it.
This statement is about the shard defect only; it does not imply 0.1.4 could load
published quantized formats. Pinned snapshots and index evidence are kept in the
private priorfix notes.
