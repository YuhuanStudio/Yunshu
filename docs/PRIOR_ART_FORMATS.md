# Published retrieval formats

EmbeddingGemma 2 bf16 and affine 4bit checkpoints use the maintained
`mlx_vlm.embedding_loader.load_embedding_model` when its model module is installed.
mlx-vlm 0.7.6 does not include that module; Yunshu carries the MIT port from
`pc/embeddinggemma-2` at `3d87e88402f307efbf68e568971aa887ee7d9ed0` and a derived
strict loader. Quantized modules are constructed from checkpoint scales; vision
and audio towers can retain floating weights. The original float32 loader remains
available for unquantized checkpoints. Integer packed tensors are never cast to
float. See vendor.json for provenance.

BERT, RoBERTa and XLM-RoBERTa use mlx-vlm’s maintained encoder loader with
`SequenceClassificationModel`, including published BGE converters with bare
backbone names and both single-label and multi-label trained heads.
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

The wider HF search snapshot contains 101 matching repository names, including
GGUF/ONNX/CoreML and unrelated substring matches. No listed model shard filenames
were found. The two repositories with multiple root safetensors are Sakura
Gemma4 DualMode: config.model_type is `gemma4`, and the files are embedding head
and LoRA adapters, not EmbeddingGemma 2 weight shards. Split-component files under
subdirectories likewise do not reach the raw root-shard loader. This is a dated
inventory, not a claim about future uploads.

The published Qwen3 0.6B 4bit chat template expects `query` and `document` roles.
The pinned aperepel service supplies `user`; that template drops the query and
document, so distinct pairs produce identical IDs and logits. Yunshu’s official
recipe matches the converted role template. Verification records this raw
negative result separately from the upstream readout with fixed identical IDs;
the latter isolates yes/no logit extraction without claiming service-template
parity. The Apache-2.0 converted template is pinned as a CPU fixture in vendor.json.

After a successful diffusion pilot, `--suite priorart --env
PRIORART_KINDS=diffusion-timing` runs three interleaved M5 pairs under gpuq quiet
admission. Both arms materialize PNGs at 256px / 2 steps / seed 7; loading is
excluded. Results report medians, output difference and the exact workload, not
a default-changing speed verdict. The CPU validator rejects incomplete pairs,
mixed devices and non-finite times.

The andrevp diffusers affine4 checkpoint has no mflux safetensors metadata. The
reviewed mflux HF mapping drops its scales/biases, leaving packed projection words
in ordinary Linear modules. The independent probe therefore validates the complete
affine tensor triplets and Qwen projection dimensions on CPU, then decodes weights
before the unchanged mflux HF mapper/applier. Packed pad tokens are decoded too.
Evidence explicitly labels this as a floating reference with a format bridge;
it is not native mflux quantized loading or a lossless serving replacement. The
bridge is isolated to research scripts and changes no serving defaults.
