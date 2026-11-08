# Round-driver migration conformance (M0)

This harness freezes the current implementations before the shared-blocks moves.
It changes no engine behavior. Receipts contain synthetic token IDs and exact
`float.hex` per-token logprobs, device/chip and package versions, prompt checksum,
seed recipe, exercised drafting counts, and module-state assertions.

CPU generic (M3; stage the new files first so m3run includes them):

```sh
scripts/dev/m3run --no-lock -- python -m pytest tests/conformance -q -p no:cacheprovider
```

Metal, using pinned ancestor/candidate SHAs, the main checkout's gpuq, and an
external `GPUQ_DIR`:

```sh
GPUQ_OWNER=m0-golden GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq \
YV_GPUQ=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq \
scripts/dev/yv ab --base BASE_SHA --cand CAND_SHA --suite round-golden \
  --env M0_LAYER=qwen3_5 --label m0-golden-UNIQUE --priority -1 --detach
scripts/dev/yv wait RUN_DIR
```

`M0_LAYER` selects `generic`, `qwen3_5`, `qwen3_5_moe`, or `flash_next`.
`M0_TINY_ONLY=1` runs only the random fixture. Otherwise the tiny cell must pass
before the real `--model` checkpoint cell starts. Generic is CPU-only. GPU calls
require `GPUQ_JOB_ID`; correctness jobs do not request quiet timing admission.
Every output is one JSON line with `complete: true`; a dry run is not evidence.

Archive successful receipts under the gitignored `docs/research/m0-golden/`,
named `LAYER-tiny.jsonl` and `LAYER-checkpoint.jsonl`, with separate device
folders. Record their SHA256 manifest in the commit message. Never force-add
research data. Later migrations pass `--env M0_REFERENCE_DIR=/absolute/device/folder`
to compare the entire deterministic payload with those receipts. Missing
references, truncated matrices, nonfinite reports, changed bits, absent drafting,
zero cache hits or absent replay fail closed. Compare only the same hardware,
packages, checkpoint and fixture recipe. The first M0 capture establishes the
reference; it is not an A/B engine performance claim.

The matrix covers greedy and seeded solo/burst/stagger/cancel, equivalent token
embeddings through the media fields, repeated logprob reporting, spec on/off
where a head is built, cold/hit and replay for checkpoint-capable packed layers.
Int8 compares the existing tile implementation with its dequantized reference
at three shapes; it does not assert int8 output equals bf16 output. The media
fixture checks the driver's embedding interface, not image-processor correctness.

The stock dense generic fixture selects upstream block-cache strategy; the
current driver admits checkpoint APC only. That baseline capability is explicitly
unavailable in the receipt, not counted as a passing APC test. No engine fix is
included here.

Globals checked after every step: `ragged_kv._STATE`,
`batch_invariant._STATE`, `gdn_prefill._STATE`, both `gdn_prefill` context scopes,
`lane_linear.STOCK_ROWS`, `_SUM_POLICY`, and `_SUM_REUSE`. Nested exception scope
restoration has a separate CPU conformance test. Installed class patches are not
pretended to be uninstalled between model builds; yv captures each family in a
fresh process.
