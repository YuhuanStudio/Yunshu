"""Engine registry for the cross-engine snapshot (tfbench.py / bench_snapshot.py).

Pure data + pure functions (no MLX, no subprocess at import): how each engine is launched, which
speculative mode it must engage (fail closed when the log or a status probe says otherwise), and
the metadata row every JSONL record carries. Unit tests: tests/unit/test_bench_snapshot.py.

Engines (ids are what ``tfbench.py --engine`` takes):
  yunshu-new / yunshu-base  Yunshu served from a pinned tree (TFB_YUNSHU_SRC = <tree>/python) under an
                            isolated HOME that holds a ``yunshu pull`` style models dir, so the DFlash2 drafter
                            is AUTO-DISCOVERED exactly as for a real user (no YUNSHU_VLM_DRAFT override)
  tf-new                    TensorFold 0.6.1, explicit --drafter (its only mode)  [legacy path in tfbench.py]
  mlxlm                     stock mlx-lm server, autoregressive
  omlx                      oMLX 0.7.0 in its own venv + own --base-path, DFlash2 on
  splash                    Splash 1.1.0, its own official checkpoint (different weights), DFlash2 automatic
  mtplx                     MTPLX 2.12.0, native MTP, same checkpoint
  llamacpp                  llama.cpp server, UD-Q4_K_M GGUF (different weights) + MTP head
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

CHECKPOINT = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
GGUF = "/Volumes/P5Plus/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
GGUF_MTP = (
    "/Volumes/P5Plus/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
)
ENVS = Path("/Volumes/P5Plus/yunshu-test-envs")
MAIN_PY = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
YUNSHU_BIN = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
# Benchmark servers start the CLI as `<venv python> -m yunshu_cli`, never through the shared
# .venv/bin/yunshu console script: a plain `uv run` re-syncs .venv and rewrites that script
# while gpuq jobs launch it (FileNotFoundError at job start).
YUNSHU_MODULE = "yunshu_cli"


def is_yunshu_command(cmd) -> bool:
    """True for a server command built here: the module form or the legacy console script."""
    return bool(cmd) and (
        Path(cmd[0]).name == "yunshu" or list(cmd[1:3]) == ["-m", YUNSHU_MODULE]
    )


def yunshu_args(cmd) -> list:
    """Arguments after the CLI entry (`serve ...`) of a command from is_yunshu_command."""
    return list(cmd[3:] if list(cmd[1:3]) == ["-m", YUNSHU_MODULE] else cmd[1:])


def yunshu_entry(environ) -> list:
    """argv prefix that starts the CLI with the main checkout's venv interpreter."""
    override = environ.get("TFB_YUNSHU_BIN")
    if override:
        return [override]
    return [MAIN_PY, "-m", YUNSHU_MODULE]


TF_BIN = str(ENVS / "tensorfold-0.6.1/bin/tensorfold")
MTPLX_BIN = str(ENVS / "mtplx/.venv/bin/mtplx")
OMLX_BIN = str(ENVS / "omlx-bench/venv/bin/omlx")
LLAMA_BIN = str(ENVS / "llamacpp/build/bin/llama-server")
SPLASH_BIN = "/opt/homebrew/bin/splash"
# refsync (2026-10-12): latest engines in their own envs under ENVS (the ids above stay for the 0.1.4 snapshot)
FN_CHECKPOINT = "/Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ3e-mtp"
FN_CHECKPOINT_PLE = "/Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ3e-mtp"
FN_GGUF = "/Volumes/Micron/yunshu-refsync/gguf-flashnext/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
FN_GGUF_MTP = (
    "/Volumes/Micron/yunshu-refsync/gguf-flashnext/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
)
TF_NATIVE_BIN = str(
    ENVS / "tensorfold-1.0.7/tensorfold-1.0.7-macos-arm64/bin/tensorfold-native"
)
TF_NATIVE_MODEL = "/Volumes/P5Plus/models/TensorFold/Qwen3.8-27B-MLX-4bit"
OMLX_BIN_LATEST = str(ENVS / "omlx-latest/bin/omlx")
LLAMA_BIN_LATEST = str(ENVS / "llamacpp-b11565/build/bin/llama-server")
DFLASH_GGUF = "/Volumes/P5Plus/models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q8_0.gguf"
MTPLX_BIN_LATEST = str(ENVS / "mtplx-2.12.2/bin/mtplx")
SPLASH_BIN_LATEST = str(ENVS / "splash-1.3.0/bin/splash")
# full comparison (2026-11-03): installs under the competitors dir, paths from compsync census.json
COMP = Path("/Volumes/P5Plus/yunshu-build/competitors")
TF_121_BIN = str(
    COMP / "tensorfold-1.2.1/tensorfold-1.2.1-macos-arm64/bin/tensorfold-native"
)
TF_118_BIN = str(
    COMP / "tensorfold-1.1.8/tensorfold-1.1.8-macos-arm64/bin/tensorfold-native"
)
SPLASH_BIN_150 = str(COMP / "splash-1.5.0/splash-1.5.0-arm64-macos26/engine/splash")
# 1.5.0 HTTP server = its own launcher (install/launcher.py serve, run by the bundled python);
# engine/splash alone only offers serve-native (no HTTP port), see server/server.py _native_command.
SPLASH_PY_150 = str(COMP / "splash-1.5.0/splash-1.5.0-arm64-macos26/python/bin/python3")
SPLASH_LAUNCHER_150 = str(
    COMP / "splash-1.5.0/splash-1.5.0-arm64-macos26/install/launcher.py"
)
OMLX_BIN_073 = str(COMP / "venvs/omlx-0.7.3/bin/omlx")
OMLX_BIN_073RC1 = str(COMP / "venvs/omlx-0.7.3rc1/bin/omlx")
OMLX_BIN_074DEV1 = str(COMP / "venvs/omlx-0.7.4.dev1/bin/omlx")
MLXSERVE_BIN_26111 = str(COMP / "mlxserve-bin/26.11.1/mlx-serve-macos-arm64/mlx-serve")
MLXLM_SERVER_BIN = str(COMP / "venvs/mlx-lm-0.32.2/bin/mlx_lm.server")
MLXVLM_PY_0711 = str(
    COMP / "venvs/mlx-vlm-0.7.11/bin/python"
)  # python -m mlx_vlm.server
LLAMA_BIN_B12061 = str(COMP / "llama.cpp-b12061/llama-b12061/llama-server")
KAPITSA_BIN_0226 = str(COMP / "kapitsa-0.2.26/kapitsa-0.2.26-darwin-arm64/kapitsa")
# 0.2.26 self-updated to 0.2.31 inside its own directory during real-1104 (four binary shas under one label), so
# the final table uses a read-only copy of the 0.2.31 binary, pinned by sha256 and launched with auto-update off.
KAPITSA_BIN_0231 = str(COMP / "kapitsa-0.2.31/kapitsa-0.2.31-darwin-arm64/kapitsa")
KAPITSA_PIN_0231 = "cd869a9cfebecdcb3d1c2b47680b4fd1516790ef850d1142c4a7705b9547b569"
# engines superseded by a re-run on a pinned binary (history rows stay; the final table uses the successor)
SUPERSEDED_BY = {"kapitsa-0.2.26": "kapitsa-0.2.31"}
KAPITSA_MODEL_DIR = "/Volumes/P5Plus/models/kapitsa/models/qwen3.8-27b-4bit"
SPLASH_MODEL = "incoai/Qwen3.8-27B-Splash"
NO_THINK_KWARGS = '{"enable_thinking": false}'
# env vars that would change a server's behaviour under a benchmark
SCRUB_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "CLAUDE",
    "CODEX",
    "YUNSHU_",
    "MLX_",
    "LLAMA_",
    "MTPLX_",
    "OMLX_",
    "TENSORFOLD_",
    "TF_",
    "KAPITSA_",
    "NI_",
    "SPLASH_",
)


@dataclass(frozen=True)
class Engine:
    id: str
    product: str  # human name
    expected_mode: str  # dflash | mtp | ar
    weights: str  # same | different
    checkpoint: str
    drafter: str | None
    kind: str  # yunshu | external
    note: str = ""
    # GPU-time model (estimate only, see bench_snapshot.estimate): relative prefill speed, decode tok/s
    prefill_factor: float = 1.0
    decode_tps: float = 60.0
    extra: dict = field(default_factory=dict)


ENGINES: dict[str, Engine] = {
    e.id: e
    for e in [
        Engine(
            "yunshu-new",
            "Yunshu 0.1.4",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "yunshu",
            "pinned verify tree; drafter auto-discovered",
            1.0,
            75.0,
        ),
        Engine(
            "yunshu-base",
            "Yunshu 0.1.3",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "yunshu",
            "tag v0.1.3 tree; drafter auto-discovered",
            1.0,
            70.0,
        ),
        Engine(
            "tf-new",
            "TensorFold 0.6.1",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "explicit --drafter",
            1.0,
            80.0,
        ),
        Engine(
            "mlxlm",
            "mlx-lm server",
            "ar",
            "same",
            CHECKPOINT,
            None,
            "external",
            "stock, autoregressive, no speculative decoding",
            0.8,
            28.0,
        ),
        Engine(
            "omlx",
            "oMLX 0.7.0",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "own venv + own --base-path; DFlash2 via model_settings.json",
            0.9,
            60.0,
        ),
        Engine(
            "splash",
            "Splash 1.1.0",
            "dflash",
            "different",
            SPLASH_MODEL,
            "automatic",
            "external",
            "official incoai/Qwen3.8-27B-Splash weights (own quantization, INT8 KV); DFlash2 automatic",
            1.1,
            75.0,
        ),
        Engine(
            "mtplx",
            "MTPLX 2.12.0",
            "mtp",
            "same",
            CHECKPOINT,
            None,
            "external",
            "native MTP, profile turbo (recommended fastest lossless path)",
            0.7,
            60.0,
        ),
        Engine(
            "llamacpp",
            "llama.cpp 0.5.0-dev (836d571)",
            "mtp",
            "different",
            GGUF,
            GGUF_MTP,
            "external",
            "UD-Q4_K_M GGUF is NOT the oQ4e weights; MTP head Q4_0",
            0.4,
            22.0,
        ),
        # ---- full comparison: the Phase B preview row (unmerged phaseb2-main 4f99270, own pin) ----
        Engine(
            "yunshu-pb2",
            "Yunshu Phase B preview (unmerged, 4f99270)",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "yunshu",
            "phaseb2-main tree 4f99270 (TFB_YUNSHU_SRC of its own pin); same launch as yunshu-main; drafter auto-discovered",
            1.0,
            75.0,
        ),
        # ---- full comparison: the pb2-generic candidate (detached tree 15ab1a0f, own pin) ----
        Engine(
            "yunshu-next",
            "Yunshu next (pb2-generic, 15ab1a0f)",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "yunshu",
            "pb2-generic tree 15ab1a0f (TFB_YUNSHU_SRC of its own pin); same launch as yunshu-pb2; default settings; drafter auto-discovered",
            1.0,
            75.0,
        ),
        # ---- refsync: latest engines, 27B cells ----
        Engine(
            "yunshu-main",
            "Yunshu main",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "yunshu",
            "main HEAD tree (TFB_YUNSHU_SRC); drafter auto-discovered",
            1.0,
            75.0,
        ),
        Engine(
            "yunshu-tf4",
            "Yunshu main (TensorFold q4g64 pack)",
            "dflash",
            "same",
            TF_NATIVE_MODEL,
            DRAFTER,
            "yunshu",
            "same weights as tf-native/splash-latest: TensorFold/Qwen3.8-27B-MLX-4bit + DFlash2",
            1.0,
            75.0,
        ),
        Engine(
            "tf-native",
            "TensorFold 1.0.7 (native Zig)",
            "dflash",
            "different",
            TF_NATIVE_MODEL,
            DRAFTER,
            "external",
            "TensorFold/Qwen3.8-27B-MLX-4bit (affine q4 g64, NOT oQ4e) + DFlash2; native runtime, no Python",
            1.0,
            75.0,
        ),
        Engine(
            "omlx-latest",
            "oMLX main",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "own venv from reference/omlx main; DFlash2 via model_settings.json",
            0.9,
            60.0,
        ),
        Engine(
            "mtplx-latest",
            "MTPLX 2.12.2",
            "mtp",
            "same",
            CHECKPOINT,
            None,
            "external",
            "native MTP, profile turbo",
            0.7,
            60.0,
        ),
        Engine(
            "llamacpp-mtp",
            "llama.cpp b11565 (MTP)",
            "mtp",
            "different",
            GGUF,
            GGUF_MTP,
            "external",
            "UD-Q4_K_M GGUF is NOT the oQ4e weights; MTP head Q4_0",
            0.4,
            22.0,
        ),
        Engine(
            "llamacpp-dflash",
            "llama.cpp b11565 (DFlash2)",
            "dflash",
            "different",
            GGUF,
            DFLASH_GGUF,
            "external",
            "UD-Q4_K_M GGUF is NOT the oQ4e weights; z-lab DFlash2 GGUF Q8_0 drafter, draft-dflash",
            0.4,
            22.0,
        ),
        Engine(
            "splash-latest",
            "Splash 1.3.0",
            "dflash",
            "different",
            TF_NATIVE_MODEL,
            DRAFTER,
            "external",
            "separate bottle copy of 1.3.0; loads the MLX q4g64 checkpoint TensorFold/Qwen3.8-27B-MLX-4bit (same weights as tf-native, NOT oQ4e) from an offline HF cache view + local DFlash2 draft; bf16 KV",
            1.1,
            75.0,
        ),
        # ---- refsync: Flash-Next cells (Qwen3.8-Flash-Next: MoE + PLE + MTP) ----
        Engine(
            "fn-yunshu-main",
            "Yunshu main (Flash-Next)",
            "mtp",
            "same",
            FN_CHECKPOINT_PLE,
            None,
            "yunshu",
            "Jundot oQ4e-mtp through the -ple-ssd view (PLE on SSD)",
            1.0,
            40.0,
        ),
        Engine(
            "fn-omlx-latest",
            "oMLX main (Flash-Next)",
            "mtp",
            "same",
            FN_CHECKPOINT,
            None,
            "external",
            "Jundot oQ4e-mtp, Lightning MTP",
            0.9,
            40.0,
        ),
        Engine(
            "fn-llamacpp-mtp",
            "llama.cpp b11565 (Flash-Next MTP)",
            "mtp",
            "different",
            FN_GGUF,
            FN_GGUF_MTP,
            "external",
            "unsloth UD-IQ4_XS GGUF (87 GiB) is NOT oQ4e; MTP head Q8_0",
            0.4,
            30.0,
        ),
        # ---- full comparison 2026-11-03 (FULLCMP_SPEC.md): manifest ids, see MANIFEST_ENGINES ----
        Engine(
            "tf-1.2.1",
            "TensorFold 1.2.1 (native Zig)",
            "dflash",
            "different",
            TF_NATIVE_MODEL,
            DRAFTER,
            "external",
            "TensorFold/Qwen3.8-27B-MLX-4bit + DFlash2; tf-native launch recipe, competitors install",
            1.0,
            75.0,
        ),
        Engine(
            "tf-1.1.8",
            "TensorFold 1.1.8 (native Zig, lone loop)",
            "dflash",
            "different",
            TF_NATIVE_MODEL,
            DRAFTER,
            "external",
            "best-1.1.8 lone loop (no batching); tf-native launch recipe",
            1.0,
            75.0,
        ),
        Engine(
            "splash-1.5.0",
            "Splash 1.5.0",
            "dflash",
            "different",
            TF_NATIVE_MODEL,
            DRAFTER,
            "external",
            "splash-latest HTTP recipe on the 1.5.0 binary (census: HTTP launcher flags unverified for 1.5.0)",
            1.1,
            75.0,
        ),
        Engine(
            "omlx-0.7.3",
            "oMLX 0.7.3",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "omlx recipe, 0.7.3 wheel venv; DFlash2 via model_settings.json",
            0.9,
            60.0,
        ),
        Engine(
            "omlx-0.7.3rc1",
            "oMLX 0.7.3rc1",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "omlx recipe, 0.7.3rc1 wheel venv; DFlash2 via model_settings.json",
            0.9,
            60.0,
        ),
        Engine(
            "omlx-0.7.4.dev1",
            "oMLX 0.7.4.dev1",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "omlx recipe, 0.7.4.dev1 wheel venv; DFlash2 via model_settings.json; INT8-activation prefill OFF "
            "(a8_prefill_enabled=false: it is ON by default in 0.7.4.dev1 and changes numerics)",
            0.9,
            60.0,
        ),
        Engine(
            "mlx-serve-26.11.1",
            "mlx-serve 26.11.1",
            "mtp",
            "unverified",
            CHECKPOINT,
            None,
            "external",
            "census start_cmd only; MTP is the configured mode (the Metal default, --mtp); weights not yet established",
            1.0,
            60.0,
        ),
        Engine(
            "mlx-lm-server",
            "mlx-lm 0.32.2 server (stock)",
            "ar",
            "same",
            CHECKPOINT,
            None,
            "external",
            "stock mlx_lm.server, no speculative decoding",
            0.8,
            28.0,
        ),
        Engine(
            "mlx-vlm-server",
            "mlx-vlm 0.7.11 server (DFlash2 drafter)",
            "dflash",
            "same",
            CHECKPOINT,
            DRAFTER,
            "external",
            "python -m mlx_vlm.server --draft-model DFlash2 --draft-kind dflash; engaged only when the log shows 'Drafter ready; speculative decoding enabled.' (an incompatible drafter falls back to AR with a warning)",
            0.8,
            28.0,
        ),
        Engine(
            "llama-cpp-b12061",
            "llama.cpp b12061 llama-server (Metal)",
            "mtp",
            "different",
            GGUF,
            GGUF_MTP,
            "external",
            "llama-server b12061, Unsloth UD-Q4_K_M GGUF + MTP head (--spec-type draft-mtp); not the same bits as the MLX checkpoint",
            1.0,
            60.0,
        ),
        Engine(
            "kapitsa-0.2.26",
            "Kapitsa 0.2.26",
            "dflash",
            "unverified",
            KAPITSA_MODEL_DIR,
            None,
            "external",
            "census start_cmd only (KAPITSA_HOME isolated); DFlash2 is the configured drafter (--dflash auto); weights not yet established",
            1.0,
            60.0,
        ),
        Engine(
            "kapitsa-0.2.31",
            "Kapitsa 0.2.31",
            "dflash",
            "unverified",
            KAPITSA_MODEL_DIR,
            None,
            "external",
            "pinned read-only binary (sha256 cd869a9c...), auto-update off (KAPITSA_NO_AUTO_UPDATE=1, CI=true); replaces the self-updated 0.2.26 rows; DFlash2 is the configured drafter (--dflash auto)",
            1.0,
            60.0,
        ),
    ]
}
# the legacy tfbench engines keep their old code path
LEGACY = ("yunshu", "tf-old", "tf-new")

# ---- full comparison: manifest engine ids (compsync/fullcmp-1103/manifest.jsonl) -> harness ids ----
# "yunshu" is a LEGACY tfbench id, so the pinned-tree Yunshu row runs as "yunshu-main".
MANIFEST_ENGINES = {
    "yunshu": "yunshu-main",
    "yunshu-pb2": "yunshu-pb2",
    "yunshu-next": "yunshu-next",
    "tf-1.2.1": "tf-1.2.1",
    "tf-1.1.8": "tf-1.1.8",
    "splash-1.5.0": "splash-1.5.0",
    "omlx-0.7.3": "omlx-0.7.3",
    "omlx-0.7.3rc1": "omlx-0.7.3rc1",
    "omlx-0.7.4.dev1": "omlx-0.7.4.dev1",
    "mlx-serve-26.11.1": "mlx-serve-26.11.1",
    "mlx-lm-server": "mlx-lm-server",
    "kapitsa-0.2.26": "kapitsa-0.2.26",
    "kapitsa-0.2.31": "kapitsa-0.2.31",
    "mlx-vlm-server": "mlx-vlm-server",
    "llama-cpp-b12061": "llama-cpp-b12061",
}
# Engines with no harness support yet: reason text only (the stub returns "unsupported (harness: ...)").
UNSUPPORTED_ENGINES = {
    "lmstudio": "lms CLI is bootstrapped into $HOME/.lmstudio; isolated-HOME headless start not verified (TODO: verify lms server start with isolated HOME)",
    "ollama": "Ollama imports GGUF only; the MLX checkpoint cannot be loaded (census n/a, not a separate row)",
    "best-competitor (TBD)": "placeholder row, not an engine; the best release is chosen after measurement (TODO: remove the row)",
}
TODO_LIST = (
    "lmstudio: verify headless lms server with isolated HOME",
    "mlx-serve-26.11.1 / kapitsa-0.2.26: establish spec mode and weights (expected_mode unverified fails closed)",
    "splash-1.5.0: HTTP launch verified from source (install/launcher.py serve), not yet run",
    "mlx-serve-26.11.1 / kapitsa-0.2.26 / mlx-vlm-server / llama-cpp-b12061: first run must engage the expected mode (see expected_mode_probe in the audit manifest)",
)


def stub_reason(manifest_id: str) -> str | None:
    """'unsupported (harness: reason)' for an engine without harness support, else None."""
    reason = UNSUPPORTED_ENGINES.get(manifest_id)
    return None if reason is None else f"unsupported (harness: {reason})"


# ---- weights each engine actually loads: the one source for the cell header and the audit ----
# sha256 of the GGUF files llama-server loads: the census (meta/model_sha256_gguf.txt) has the main
# file; the MTP head was hashed 2026-11-04 (shasum -a 256 on the file).
GGUF_SHA256 = {
    "Qwen3.8-27B-UD-Q4_K_M.gguf": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
    "mtp-Qwen3.8-27B-Q4_0.gguf": "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e",
}
GGUF_QUANT = {"Qwen3.8-27B-UD-Q4_K_M.gguf": "UD-Q4_K_M", "mtp-Qwen3.8-27B-Q4_0.gguf": "MTP Q4_0"}
# census weights key of the checkpoint each engine's launch serves (bench id)
WEIGHTS_CENSUS_KEY = {
    "llama-cpp-b12061": "qwen3.8-27b-gguf-unsloth",
    "kapitsa-0.2.26": "qwen3.8-27b-4bit-kapitsa",
    "kapitsa-0.2.31": "qwen3.8-27b-4bit-kapitsa",
    "tf-1.2.1": "qwen3.8-27b-4bit-tensorfold",
    "tf-1.1.8": "qwen3.8-27b-4bit-tensorfold",
    "splash-1.5.0": "qwen3.8-27b-4bit-tensorfold",
}
DEFAULT_WEIGHTS_KEY = "qwen3.8-27b-4bit-jundot"  # CHECKPOINT: Yunshu, oMLX, mlx-lm, mlx-vlm, mlx-serve


def launched_gguf_files(engine: str) -> list[str]:
    """GGUF files the launch loads (-m, -md); [] for MLX engines."""
    if engine == "llama-cpp-b12061":
        return [GGUF, GGUF_MTP]
    return []


def served_model_dir(engine: str, default: str) -> str:
    """Directory of the safetensors the launch serves (for MLX engines); default for the rest."""
    if engine in ("kapitsa-0.2.26", "kapitsa-0.2.31"):
        return KAPITSA_MODEL_DIR
    if engine in ("tf-1.2.1", "tf-1.1.8", "splash-1.5.0"):
        return TF_NATIVE_MODEL
    return default


def gguf_weights_record(engine: str) -> list[dict]:
    """[{file, sha256, quant, size}] for the GGUF files the launch loads (the MLX fields do not apply)."""
    import os

    out = []
    for path in launched_gguf_files(engine):
        name = Path(path).name
        out.append(
            {
                "file": name,
                "sha256": GGUF_SHA256[name],
                "quant": GGUF_QUANT[name],
                "size": os.path.getsize(path) if os.path.exists(path) else None,
            }
        )
    return out


def weights_note(engine: str) -> str:
    """Footnote for the audit and the table: 'same weights' only when the launch serves Yunshu's checkpoint."""
    if launched_gguf_files(engine):
        return "different weights (GGUF UD-Q4_K_M + MTP Q4_0)"
    key = WEIGHTS_CENSUS_KEY.get(engine, DEFAULT_WEIGHTS_KEY)
    if key == DEFAULT_WEIGHTS_KEY:
        return "same weights as Yunshu (Jundot oQ4e-mtp)"
    return f"different weights ({key})"


def manifest_id(harness_id: str) -> str:
    """Manifest engine id for a tfbench (harness) id: yunshu-main -> yunshu. Unknown ids come back
    unchanged, so the attribution schema lookup (fullcmp_attribution.engine_attr) fails closed."""
    inverse = {v: k for k, v in MANIFEST_ENGINES.items()}
    return inverse.get(harness_id, harness_id)


def tfbench_engine(manifest_id: str) -> str:
    """tfbench --engine id for a manifest engine id; KeyError for stubs and unknown ids."""
    if manifest_id in UNSUPPORTED_ENGINES:
        raise KeyError(stub_reason(manifest_id))
    return MANIFEST_ENGINES[manifest_id]


TF_ENGINES = ("tf-new", "tf-old")


def is_new_engine(engine: str) -> bool:
    """Engines launched by build_launch() (everything except tfbench's original three)."""
    return engine in ENGINES and engine not in LEGACY


class PinMismatch(RuntimeError):
    """The file a launch would run is not the pinned binary (or is missing): the cell must not run."""


def verify_pinned_binary(path: str, sha256: str) -> None:
    """Fail closed before a server starts: the launched file's sha256 must be the pin (the audit's
    pinned_versions guard checks the same after the fact)."""

    real = str(Path(path).resolve())
    if not os.path.isfile(real):
        raise PinMismatch(f"pinned binary {path} is missing")
    h = hashlib.sha256()
    with open(real, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != sha256:
        raise PinMismatch(f"{path} sha256 {h.hexdigest()} is not the pin {sha256}")


def scrubbed_env(environ: dict, extra: dict | None = None) -> dict:
    env = {
        k: v
        for k, v in environ.items()
        if not k.startswith(SCRUB_PREFIXES) and k not in ("PYTHONPATH", "PYTHONHOME")
    }
    env.update(extra or {})
    return env


def omlx_model_settings(model_id: str, a8_off: bool = False) -> dict:
    # mtp_enabled is written explicitly false: an absent key defaults to False in oMLX too, but
    # the written file must state it so only the DFlash2 drafter can produce MTP[uid] stats lines.
    # a8_off (0.7.4.dev1): INT8-activation prefill is on by default there (None = on); the lossless measured
    # configuration saves it off. The file is version 2: a version-1 file's saved False is dropped on load where the
    # kernels cannot run (model_settings.py:871-876), which is not the case on this M5, but version 2 states it.
    settings = {"dflash_enabled": True, "dflash_draft_model": DRAFTER, "mtp_enabled": False}
    if a8_off:
        settings["a8_prefill_enabled"] = False
    return {"version": 2 if a8_off else 1, "models": {model_id: settings}}


@dataclass
class Launch:
    cmd: list[str]
    env: dict
    flags: dict
    files: dict  # path -> text content written before launch
    links: dict  # link path -> target (symlinks created before launch)
    probe: str | None = (
        None  # status URL path fetched after the warm-up (engaged-mode evidence)
    )


def yunshu_home_links(home: str | Path, checkpoint: str = CHECKPOINT) -> dict:
    """A ``yunshu pull`` layout inside the isolated HOME: models dir org/name entries. The drafter lives
    next to the checkpoint under ~/.yunshu/models, which is where spec_select looks, so discovery is real."""
    base = Path(home) / ".yunshu" / "models"
    links = {str(base / "Jundot" / Path(checkpoint).name): checkpoint}
    if checkpoint in (CHECKPOINT, TF_NATIVE_MODEL):
        links[str(base / "incoai" / "Qwen3.8-27B-DFlash2")] = DRAFTER
    return links


def build_launch(
    engine: str,
    port: int,
    home: str | Path,
    environ: dict,
    ctx_tokens: int = 140000,
    parallel: int = 1,
) -> Launch:
    """Command, environment and pre-launch files for one server session. Pure."""
    home = str(home)
    p = str(port)
    e = ENGINES[engine]
    base_env = {"HOME": home, "HF_HUB_OFFLINE": "1", "NO_PROXY": "127.0.0.1"}
    if e.kind == "yunshu":
        links = yunshu_home_links(home, e.checkpoint)
        model = str(Path(home) / ".yunshu/models/Jundot" / Path(e.checkpoint).name)
        return Launch(
            [
                *yunshu_entry(environ),
                "serve",
                "-m",
                model,
                "--port",
                p,
            ],
            scrubbed_env(environ, base_env),
            {
                "yunshu_vlm_draft": "unset (auto-discovery)",
                "models_dir": "$HOME/.yunshu/models",
            },
            {},
            links,
        )
    if engine == "mlxlm":
        return Launch(
            [
                environ.get("TFB_MLXLM_PY", MAIN_PY),
                "-m",
                "mlx_lm",
                "server",
                "--model",
                CHECKPOINT,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--chat-template-args",
                NO_THINK_KWARGS,
                "--decode-concurrency",
                "8",
                "--prompt-concurrency",
                "8",
            ],
            scrubbed_env(environ, base_env),
            {"speculative": "none", "decode_concurrency": 8, "prompt_concurrency": 8},
            {str(Path(home) / ".cache/huggingface/hub/.snapshot014"): ""},
            {},
        )
    if engine == "omlx":
        mid = "Qwen3.8-27B-oQ4e-mtp"
        settings = Path(home) / "omlx-base" / "model_settings.json"
        settings_text = json.dumps(omlx_model_settings(mid))
        flags = {
            "dflash_enabled": True,
            "dflash_draft_model": DRAFTER,
            "mtp_enabled": False,
            "model_settings_sha256": hashlib.sha256(settings_text.encode()).hexdigest(),
            "memory_guard_gb": 96,
            "max_concurrent_requests": max(8, parallel),
            "parallel": parallel,
            "ssd_cache": "20GB",
            "hot_cache": "8GB",
        }
        return Launch(
            [
                OMLX_BIN,
                "serve",
                "--model-dir",
                str(Path(home) / "omlx-models"),
                "--host",
                "127.0.0.1",
                "--port",
                p,
                # default 8 (omlx serve --help); a cell serving more clients raises it to N
                "--max-concurrent-requests",
                str(max(8, parallel)),
                "--memory-guard-gb",
                "96",
                "--paged-ssd-cache-dir",
                str(Path(home) / "omlx-ssd"),
                "--paged-ssd-cache-max-size",
                "20GB",
                "--hot-cache-max-size",
                "8GB",
                "--no-hf-cache",
                "--base-path",
                str(Path(home) / "omlx-base"),
                "--log-level",
                "info",
            ],
            scrubbed_env(environ, base_env),
            flags,
            {str(settings): settings_text},
            {str(Path(home) / "omlx-models" / mid): CHECKPOINT},
        )
    if engine == "splash":
        # Splash keeps its models under the user's ~/Library/Application Support/Splash (read only for us);
        # a separate instance on our port, SSD cache off (its default), nothing written to the user's service.
        env = scrubbed_env(environ, base_env)
        return Launch(
            [
                SPLASH_BIN,
                "serve",
                "--model",
                SPLASH_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--no-webui",
                "--kv-format",
                "bf16",
                "--default-reasoning-effort",
                "none",
            ],
            env,
            {
                "kv_format": "bf16 (lossless)",
                "max_cache_disk": "0 (default)",
                "reasoning_effort": "none",
            },
            {},
            {
                str(
                    Path(home)
                    / "Library/Application Support/Splash/models"
                    / SPLASH_MODEL
                ): "/Users/yuhuan/Library/Application Support/Splash/models/"
                + SPLASH_MODEL
            },
            probe="/status",
        )
    if engine == "mtplx":
        return Launch(
            [
                MTPLX_BIN,
                "serve",
                "--model",
                CHECKPOINT,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--profile",
                "turbo",
                "--kv-quant",
                "off",
                "--no-auth",
                "--cache-dir",
                str(Path(home) / "mtplx-cache"),
            ],
            scrubbed_env(environ, base_env),
            {
                "profile": "turbo",
                "generation_mode": "mtp (default)",
                "kv_quant": "off (default)",
            },
            {},
            {},
        )
    if engine == "llamacpp":
        ctx = int(ctx_tokens)
        return Launch(
            [
                LLAMA_BIN,
                "-m",
                GGUF,
                "--spec-type",
                "draft-mtp",
                "--spec-draft-model",
                GGUF_MTP,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "-c",
                str(ctx),
                "-np",
                str(parallel),
                "-ngl",
                "99",
                "-fa",
                "on",
                "--jinja",
                "--chat-template-kwargs",
                NO_THINK_KWARGS,
                "--no-webui",
            ],
            scrubbed_env(environ, base_env),
            {
                "ctx": ctx,
                "parallel": parallel,
                "flash_attn": "on",
                "spec_type": "draft-mtp",
            },
            {},
            {},
            probe="/props",
        )
    if engine in REFSYNC_ALIASES:
        # same launch recipe as the pinned-snapshot engine, newer binary
        base = REFSYNC_ALIASES[engine]
        launch = build_launch(base, port, home, environ, ctx_tokens, parallel)
        if engine in OMLX_BIN_BY_ENGINE:
            launch.cmd[0] = OMLX_BIN_BY_ENGINE[engine]
        if engine in OMLX_A8_OFF_ENGINES:
            # same model settings file, rewritten with A8 prefill saved off (settings version 2, the file version of
            # 0.7.4.dev1); the recorded flags carry the new hash and the setting
            mid = "Qwen3.8-27B-oQ4e-mtp"
            for path in list(launch.files):
                if path.endswith("model_settings.json"):
                    text = json.dumps(omlx_model_settings(mid, a8_off=True))
                    launch.files[path] = text
                    launch.flags["model_settings_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            launch.flags["a8_prefill_enabled"] = False
        elif engine == "mtplx-latest":
            launch.cmd[0] = MTPLX_BIN_LATEST
        return launch
    if engine == "splash-1.5.0":
        # Splash 1.5.0 HTTP server: its launcher serves the local TensorFold MLX directory
        # (--model-dir, offline) with the local DFlash2 drafter. No HF snapshot links needed.
        return Launch(
            [
                SPLASH_PY_150,
                SPLASH_LAUNCHER_150,
                "serve",
                "--model",
                "TensorFold/Qwen3.8-27B-MLX-4bit",
                "--model-dir",
                TF_NATIVE_MODEL,
                "--offline",
                "--draft-model",
                DRAFTER,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--no-webui",
                "--kv-format",
                "bf16",
                "--default-reasoning-effort",
                "none",
                "--max-context",
                str(int(ctx_tokens)),
                # "keep the SSD cache across restarts" (server/serve_options.py:555-562; files under
                # ~/Library/Caches/Splash/prefix-cache, :564-571, inside the isolated HOME): automatic restore,
                # no client call, so a user who restarts the server turns it on and the restart cell measures it
                "--persistent-cache",
            ],
            scrubbed_env(
                environ,
                {**base_env, "HF_HOME": str(Path(home) / "hf")},
            ),
            {
                "kv_format": "bf16 (lossless)",
                "reasoning_effort": "none",
                "draft": "local DFlash2",
                "persistent_cache": "on (SSD cache kept across restarts)",
            },
            {},
            {},
            probe="/status",
        )
    if engine in ("splash-latest",):
        rev = "a" * 40
        hub = (
            Path(home)
            / ".cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit"
        )
        return Launch(
            [
                SPLASH_BIN_LATEST if engine == "splash-latest" else SPLASH_BIN_150,
                "serve",
                "--model",
                "TensorFold/Qwen3.8-27B-MLX-4bit",
                "--offline",
                "--draft-model",
                DRAFTER,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--no-webui",
                "--kv-format",
                "bf16",
                "--default-reasoning-effort",
                "none",
                "--max-context",
                str(int(ctx_tokens)),
            ],
            scrubbed_env(environ, base_env),
            {
                "kv_format": "bf16 (lossless)",
                "reasoning_effort": "none",
                "draft": "local DFlash2",
            },
            {str(hub / "refs" / "main"): rev},
            {str(hub / "snapshots" / rev): TF_NATIVE_MODEL},
            probe="/status",
        )
    if engine in TF_NATIVE_FAMILY:
        return Launch(
            [
                TF_NATIVE_BIN_BY_ENGINE[engine],
                "serve",
                TF_NATIVE_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--drafter",
                DRAFTER,
                "--no-update-check",
                "--no-thinking",
                "--parallel",
                str(parallel),
                "--context",
                str(int(ctx_tokens)),
            ],
            scrubbed_env(environ, base_env),
            # --parallel N streams, each of --context tokens (its own startup line: 'Qwen3.8-27B: 1 streams of 140000
            # tokens'), so a concurrent cell passes N and a context sized to the cell, not 140000
            {"drafter": "DFlash2 (explicit --drafter)", "thinking": "off", "parallel": parallel},
            {},
            {},
        )
    if engine in ("llamacpp-mtp", "llamacpp-dflash"):
        dflash = engine == "llamacpp-dflash"
        ctx = int(ctx_tokens) * parallel
        return Launch(
            [
                LLAMA_BIN_LATEST,
                "-m",
                GGUF,
                "--spec-type",
                "draft-dflash" if dflash else "draft-mtp",
                "--spec-draft-model",
                DFLASH_GGUF if dflash else GGUF_MTP,
                "--spec-draft-n-max",
                "7" if dflash else "3",
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "-c",
                str(ctx),
                "-np",
                str(parallel),
                "-ngl",
                "99",
                "-fa",
                "on",
                "--jinja",
                "--chat-template-kwargs",
                NO_THINK_KWARGS,
                "--no-webui",
            ],
            scrubbed_env(environ, base_env),
            {
                "ctx": ctx,
                "parallel": parallel,
                "flash_attn": "on",
                "spec_type": "draft-dflash" if dflash else "draft-mtp",
                "n_max": 7 if dflash else 3,
            },
            {},
            {},
            probe="/props",
        )
    if engine == "fn-omlx-latest":
        mid = Path(FN_CHECKPOINT).name
        settings = Path(home) / "omlx-base" / "model_settings.json"
        cfg = {"version": 1, "models": {mid: {"mtp_enabled": True}}}
        return Launch(
            [
                OMLX_BIN_LATEST,
                "serve",
                "--model-dir",
                str(Path(home) / "omlx-models"),
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--max-concurrent-requests",
                "8",
                "--memory-guard-gb",
                "120",
                "--paged-ssd-cache-dir",
                str(Path(home) / "omlx-ssd"),
                "--paged-ssd-cache-max-size",
                "20GB",
                "--hot-cache-max-size",
                "4GB",
                "--no-hf-cache",
                "--base-path",
                str(Path(home) / "omlx-base"),
                "--log-level",
                "info",
            ],
            scrubbed_env(environ, base_env),
            {
                "mtp": "model_settings mtp_enabled=true (verify in log)",
                "memory_guard_gb": 120,
            },
            {str(settings): json.dumps(cfg)},
            {str(Path(home) / "omlx-models" / mid): FN_CHECKPOINT},
        )
    if engine == "fn-llamacpp-mtp":
        ctx = int(ctx_tokens)
        return Launch(
            [
                LLAMA_BIN_LATEST,
                "-m",
                FN_GGUF,
                "--spec-type",
                "draft-mtp",
                "--spec-draft-model",
                FN_GGUF_MTP,
                "--spec-draft-n-max",
                "3",
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "-c",
                str(ctx),
                "-np",
                str(parallel),
                "-ngl",
                "99",
                "-fa",
                "on",
                "--jinja",
                "--chat-template-kwargs",
                NO_THINK_KWARGS,
                "--no-webui",
            ],
            scrubbed_env(environ, base_env),
            {"ctx": ctx, "spec_type": "draft-mtp", "n_max": 3},
            {},
            {},
            probe="/props",
        )
    if engine == "mlx-serve-26.11.1":
        return Launch(
            [
                MLXSERVE_BIN_26111,
                "serve",
                "--model",
                CHECKPOINT,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--ctx-size",
                str(int(ctx_tokens)),
                # the documented SSD prefix tier (docs/cli.md:90, src/main.zig:376, default off): prefixes survive
                # restarts. A user who restarts a long-conversation server turns it on; root ~/.mlx-serve/kv-cache
                # (src/kv_disk_cache.zig:3640) falls under the isolated HOME
                "--prefix-cache-disk",
                "20GB",
                # --max-concurrent N: sizes the request queue and caps how many decoders a long prefill admits a
                # newcomer beside (default 1; mlx-serve serve --help). A concurrent cell serves N clients
                *(["--max-concurrent", str(parallel)] if parallel > 1 else []),
            ],
            scrubbed_env(environ, base_env),
            {
                "ctx_size": int(ctx_tokens),
                "spec_mode": "mtp",
                "prefix_cache_disk": "20GB (documented SSD tier)",
                "parallel": parallel,
            },
            {},
            {},
        )
    if engine == "mlx-vlm-server":
        return Launch(
            [
                MLXVLM_PY_0711,
                "-m",
                "mlx_vlm.server",
                "--model",
                CHECKPOINT,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--draft-model",
                DRAFTER,
                "--draft-kind",
                "dflash",
            ],
            # prefix caching (APC) is OFF by default (server/runtime_config.py:76, :321; read from APC_ENABLED at
            # :360): without it every request re-prefills and warm / follow-up cells measure nothing. A user who
            # serves long conversations turns it on; disk persistence is then on by default (:78-91, :322, files
            # under ~/.cache/mlx-vlm/apc, apc.py:84-89, inside the isolated HOME)
            scrubbed_env(environ, {**base_env, "APC_ENABLED": "1"}),
            {"speculative": "dflash", "draft_kind": "dflash", "apc": "APC_ENABLED=1 (prefix cache on, disk persistence default)"},
            {},
            {},
        )
    if engine == "llama-cpp-b12061":
        return Launch(
            [
                LLAMA_BIN_B12061,
                "-m",
                GGUF,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                # a concurrent cell (parallel N > 1): -np N slots with the context per slot = the cell's size, so
                # -c is the total, N x the cell (llama-server --help: -np number of server slots, default auto; with an
                # explicit slot count the unified KV buffer is off and -c is split across the slots)
                "-c",
                str(int(ctx_tokens) * max(1, parallel)),
                *(["-np", str(parallel)] if parallel > 1 else []),
                "--jinja",
                "--reasoning",
                "off",
                "--spec-type",
                "draft-mtp",
                "-md",
                GGUF_MTP,
                "--metrics",
            ],
            scrubbed_env(environ, base_env),
            {"spec_type": "draft-mtp", "reasoning": "off", "parallel": parallel},
            {},
            {},
        )
    if engine == "mlx-lm-server":
        return Launch(
            [
                MLXLM_SERVER_BIN,
                "--model",
                CHECKPOINT,
                "--host",
                "127.0.0.1",
                "--port",
                p,
                "--max-tokens",
                "8192",
                "--chat-template-args",
                NO_THINK_KWARGS,
            ],
            scrubbed_env(environ, base_env),
            {"speculative": "none"},
            # mlx_lm.server lists the HF cache on /v1/models (scan_cache_dir raises CacheNotFound
            # when the hub dir is missing: the 2026-11-03 smoke's 500). An empty hub dir in the
            # isolated HOME is enough; no model is downloaded.
            {str(Path(home) / ".cache/huggingface/hub/.keep"): ""},
            {},
        )
    if engine in ("kapitsa-0.2.26", "kapitsa-0.2.31"):
        pinned = engine == "kapitsa-0.2.31"
        if pinned:
            verify_pinned_binary(KAPITSA_BIN_0231, KAPITSA_PIN_0231)
        kh = str(Path(home) / "kapitsa-home")
        # --private: listen on 127.0.0.1 only and do not announce on the network (kapitsa serve --help); by default
        # the server is visible to every Mac on the LAN. A concurrent cell (parallel N > 1) is launched the way a user
        # serving N clients would: --slots N (requests that run at once; default up to 4) and --context sized to the
        # cell (context per conversation; default 256K x slots)
        conc_args = (
            ["--slots", str(parallel), "--context", str(int(ctx_tokens))] if parallel > 1 else []
        )
        return Launch(
            [
                KAPITSA_BIN_0231 if pinned else KAPITSA_BIN_0226,
                "serve", "--model", "27b", "--dflash", "auto", "--port", p, "--private", *conc_args,
            ],
            # the pinned run launches with self-update off: 0.2.26 replaced its own binary mid-run
            scrubbed_env(
                environ,
                {**base_env, "KAPITSA_HOME": kh, **({"KAPITSA_NO_AUTO_UPDATE": "1", "CI": "true"} if pinned else {})},
            ),
            {"model": "27b", "spec_mode": "dflash", "private": True, "parallel": parallel},
            {},
            {str(Path(kh) / "models" / "qwen3.8-27b-4bit"): KAPITSA_MODEL_DIR},
        )
    raise KeyError(engine)


# The launch flag that sets how many requests an engine serves at once, per manifest engine id (the audit fails a
# concurrent cell whose header shows a value below the cell's N). Engines absent here have no such flag or admit
# concurrently without one: Yunshu / pb2 (YUNSHU_MAX_CONCURRENT unset: adaptive, starts at 8; settings.py:133),
# Splash (--queue-size 32), mlx-vlm (--max-num-seqs unbounded), mlx-lm (--decode-concurrency 32).
PARALLEL_FLAG = {
    "tf-1.2.1": "--parallel",
    "tf-1.1.8": "--parallel",
    "mlx-serve-26.11.1": "--max-concurrent",
    "kapitsa-0.2.26": "--slots",
    "kapitsa-0.2.31": "--slots",
    "llama-cpp-b12061": "-np",
    "omlx-0.7.3": "--max-concurrent-requests",
    "omlx-0.7.3rc1": "--max-concurrent-requests",
    "omlx-0.7.4.dev1": "--max-concurrent-requests",
}

REFSYNC_ALIASES = {
    "omlx-latest": "omlx",
    "mtplx-latest": "mtplx",
    "omlx-0.7.3": "omlx",
    "omlx-0.7.3rc1": "omlx",
    "omlx-0.7.4.dev1": "omlx",
}
OMLX_BIN_BY_ENGINE = {
    "omlx-latest": OMLX_BIN_LATEST,
    "omlx-0.7.3": OMLX_BIN_073,
    "omlx-0.7.3rc1": OMLX_BIN_073RC1,
    "omlx-0.7.4.dev1": OMLX_BIN_074DEV1,
}
# oMLX versions whose INT8-activation (A8) prefill is ON by default on M5 and changes numerics: the measured
# configuration writes a8_prefill_enabled=false in the model settings (lossless; model_settings.py:216-222, :337-343
# of 0.7.4.dev1). 0.7.3 and earlier default to off.
OMLX_A8_OFF_ENGINES = frozenset({"omlx-0.7.4.dev1"})
TF_NATIVE_FAMILY = ("tf-native", "tf-1.2.1", "tf-1.1.8")
TF_NATIVE_BIN_BY_ENGINE = {
    "tf-native": TF_NATIVE_BIN,
    "tf-1.2.1": TF_121_BIN,
    "tf-1.1.8": TF_118_BIN,
}


# ---- engaged-speculative-mode detection (fail closed) -----------------------------------------------


def detect_mode(engine: str, log: str, probe_text: str | None = None) -> str | None:
    """The speculative mode a started server actually runs, from its log (and a status probe), or None
    when there is no evidence. Callers compare against Engine.expected_mode and refuse to measure."""
    e = ENGINES[engine]
    if e.kind == "yunshu":
        modes = re.findall(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b", log)
        return modes[-1] if modes else None
    if engine in TF_NATIVE_FAMILY:
        # 1.2.0+ serving line names the drafter: "drafts: DFlash2 (<path>)" (or "drafts: DFlash2 (...) + copies").
        if re.search(r"drafts: DFlash2\b", log):
            return "dflash"
        # 1.0.x native log: a drafter line naming DFlash2 and "drafts: on" (confirmed by the pilot)
        return (
            "dflash"
            if re.search(r"DFlash2 drafts for|drafter[^\n]*(DFlash|dflash)", log)
            and re.search(r"drafts: on", log)
            else None
        )
    if engine == "llamacpp-dflash":
        # b11565 logs "adding speculative implementation 'draft-dflash'" when the drafter is wired
        wired = re.search(r"adding speculative implementation 'draft-dflash'", log)
        configured = re.search(r"draft-dflash", probe_text or "")
        return "dflash" if wired or configured else None
    if engine == "fn-omlx-latest":
        return (
            "mtp"
            if re.search(
                r"Lightning MTP enabled for|MTP path activated|Speculative backend selected for [^\n]*Lightning MTP",
                log,
            )
            else None
        )
    legacy_engine = engine
    engine = {
        "omlx-latest": "omlx",
        "mtplx-latest": "mtplx",
        "splash-latest": "splash",
        "splash-1.5.0": "splash",
        "omlx-0.7.3": "omlx",
        "omlx-0.7.3rc1": "omlx",
        "omlx-0.7.4.dev1": "omlx",
        "mlx-lm-server": "mlxlm",
        "mlx-vlm-server": "mlxvlm",
        "llama-cpp-b12061": "llamacpp",
        "llamacpp-mtp": "llamacpp",
        "fn-llamacpp-mtp": "llamacpp",
    }.get(engine, engine)
    if engine == "tf-new":
        return (
            "dflash"
            if re.search(r"\[tensorfold\] drafter [^\n]*DFlash[^\n]*block=", log)
            else None
        )
    if engine == "omlx":
        if re.search(r"DFlash enabled for [^\n]*draft=", log):
            return "dflash"
        return None
    if engine == "mtplx":
        m = re.search(r"Mode\s+(\S[^\n│]*?)\s*│", log) or re.search(
            r"Mode\s+([^\n]+)", log
        )
        if not m:
            return None
        text = m.group(1).lower()
        return "mtp" if "mtp" in text else "ar" if "ar" in text.split() else None
    if engine == "llamacpp":
        loaded_mtp = re.search(
            r"loading draft model [^\n]*[/\\]MTP[/\\]mtp-", log, re.I
        )
        drafted = re.search(
            r"draft acceptance\s*=\s*[0-9.]+\s*\(\s*\d+ accepted /\s*([1-9]\d*) generated",
            log,
        )
        configured = re.search(
            r'"speculative\.types"\s*:\s*"draft-mtp"', probe_text or ""
        )
        # b11565 prints no per-request speculative stats at the default log level: the MTP head file
        # loaded as the draft model, or "draft-mtp" in /props, is the engagement evidence.
        # b12061 prints per-request "draft acceptance" lines, so only a drafted reply counts there
        loose = legacy_engine not in ("llamacpp", "llama-cpp-b12061")
        if (
            (loaded_mtp and drafted)
            or (loose and loaded_mtp)
            or configured
            or re.search(r"draft-mtp|speculative decoding.*mtp|mtp.*draft", log, re.I)
            or re.search(r"draft-mtp", probe_text or "")
        ):
            return "mtp"
        return None
    if engine == "splash":
        text = (probe_text or "").lower()
        return "dflash" if "draft" in text or "dflash" in text else None
    if engine == "mlxvlm":
        # mlx_vlm/server/generation.py logs "Loading speculative drafter (<kind>): <path>", then
        # "Drafter ready; speculative decoding enabled." once it validated against the target.
        # An incompatible drafter logs a warning and falls back to AR, so "ready" is the evidence.
        kind = re.search(r"Loading speculative drafter \((dflash|mtp)\)", log)
        if kind and re.search(r"Drafter ready; speculative decoding enabled\.", log):
            return kind.group(1)
        return None
    if engine == "mlxlm":
        # no speculative path is configured; "ar" is by construction once the server answered
        return "ar" if re.search(r"Starting httpd|http://127\.0\.0\.1", log) else None
    return None


def check_engaged(engine: str, engaged: str | None) -> None:
    want = ENGINES[engine].expected_mode
    if engaged != want:
        raise RuntimeError(
            f"{engine}: expected spec mode {want!r}, engaged {engaged!r}"
        )


def meta_row(
    engine: str,
    *,
    version: str,
    git_sha: str | None,
    engaged: str | None,
    flags: dict,
    requested_mode: str | None = None,
) -> dict:
    """The fields every JSONL row of a snapshot carries."""
    e = ENGINES[engine]
    return {
        "engine": engine,
        "product": e.product,
        "version": version,
        "git_sha": git_sha,
        "flags": flags,
        "drafter": e.drafter,
        "spec_mode": engaged,
        "spec_mode_expected": e.expected_mode,
        "checkpoint": e.checkpoint,
        "weights_vs_oQ4e": e.weights,
    }


# ---- versions / preflight ---------------------------------------------------------------------------

STATIC_VERSIONS = {
    "tf-new": ("0.6.1", "17c73e189f5e6a5304cda7ea37f086f9c49b4788"),
    "mlxlm": ("mlx-lm 0.32.0 (snapshot014 isolated venv)", None),
    "omlx": (
        "0.7.0 (git tag v0.7.0, own venv)",
        "4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40",
    ),
    "splash": ("1.1.0 (Homebrew)", None),
    "mtplx": ("2.12.0 (own venv, mlx-lm 0.31.3)", None),
    "llamacpp": ("0.5.0-dev build 242", "836d57176dc699a726c55418e4f96b8ca628e1bf"),
    "tf-native": ("1.0.7 (native Zig, release binary)", None),
    "omlx-latest": ("main 0b07e88e (0.7.0 + 171 commits)", "0b07e88e"),
    "mtplx-latest": ("2.12.2 (own venv, mlx 0.32.2)", "9882703f"),
    "llamacpp-mtp": ("b11565 (0.6.0-dev build 427)", "2f160ba59"),
    "llamacpp-dflash": ("b11565 (0.6.0-dev build 427)", "2f160ba59"),
    "fn-omlx-latest": ("main 0b07e88e (0.7.0 + 171 commits)", "0b07e88e"),
    "fn-llamacpp-mtp": ("b11565 (0.6.0-dev build 427)", "2f160ba59"),
    "splash-latest": ("1.3.0 (bottle copy)", None),
    # full comparison: exact census versions (the audit compares header version == pin)
    "tf-1.2.1": ("1.2.1", None),
    "tf-1.1.8": ("1.1.8", None),
    "splash-1.5.0": ("1.5.0", None),
    "omlx-0.7.3": ("0.7.3", None),
    "omlx-0.7.3rc1": ("0.7.3rc1", None),
    "omlx-0.7.4.dev1": ("0.7.4.dev1", None),
    "mlx-serve-26.11.1": ("26.11.1", None),
    "mlx-lm-server": ("0.32.2", None),
    "kapitsa-0.2.26": ("0.2.26", None),
    "kapitsa-0.2.31": ("0.2.31", None),
    "mlx-vlm-server": ("0.7.11", None),
    "llama-cpp-b12061": ("b12061", None),
}


def tree_version(src: str) -> tuple[str, str | None]:
    """(pyproject version, git sha) of a pinned Yunshu tree given its python/ directory."""
    import subprocess

    root = Path(src.rstrip("/")).parent
    version = "unknown"
    pp = root / "pyproject.toml"
    if pp.is_file():
        m = re.search(r'^version\s*=\s*"([^"]+)"', pp.read_text(), re.M)
        version = m.group(1) if m else version
    try:
        sha = (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001
        sha = None
    return version, sha


def engine_version(engine: str, yunshu_src: str = "") -> tuple[str, str | None]:
    if ENGINES[engine].kind == "yunshu":
        return tree_version(yunshu_src) if yunshu_src else ("unknown", None)
    return STATIC_VERSIONS.get(engine, ("unknown", None))


def required_paths(engine: str) -> list[str]:
    """Files that must exist before a job is worth submitting."""
    common = [CHECKPOINT + "/config.json"]
    e = ENGINES[engine]
    if engine == "fn-yunshu-main":
        return [FN_CHECKPOINT_PLE + "/config.json", YUNSHU_BIN]
    if e.kind == "yunshu":
        return common + [DRAFTER + "/config.json", YUNSHU_BIN]
    return {
        "tf-native": [
            TF_NATIVE_MODEL + "/config.json",
            DRAFTER + "/config.json",
            TF_NATIVE_BIN,
        ],
        "omlx-latest": common + [DRAFTER + "/config.json", OMLX_BIN_LATEST],
        "mtplx-latest": common + [MTPLX_BIN_LATEST],
        "llamacpp-mtp": [GGUF, GGUF_MTP, LLAMA_BIN_LATEST],
        "llamacpp-dflash": [GGUF, DFLASH_GGUF, LLAMA_BIN_LATEST],
        "splash-latest": [
            SPLASH_BIN_LATEST,
            TF_NATIVE_MODEL + "/config.json",
            DRAFTER + "/config.json",
        ],
        "fn-omlx-latest": [FN_CHECKPOINT + "/config.json", OMLX_BIN_LATEST],
        "fn-llamacpp-mtp": [FN_GGUF, FN_GGUF_MTP, LLAMA_BIN_LATEST],
        "tf-1.2.1": [
            TF_NATIVE_MODEL + "/config.json",
            DRAFTER + "/config.json",
            TF_121_BIN,
        ],
        "tf-1.1.8": [
            TF_NATIVE_MODEL + "/config.json",
            DRAFTER + "/config.json",
            TF_118_BIN,
        ],
        "splash-1.5.0": [
            SPLASH_PY_150,
            SPLASH_LAUNCHER_150,
            TF_NATIVE_MODEL + "/config.json",
            DRAFTER + "/config.json",
        ],
        "mlx-vlm-server": common + [MLXVLM_PY_0711, DRAFTER + "/config.json"],
        "llama-cpp-b12061": [GGUF, GGUF_MTP, LLAMA_BIN_B12061],
        "omlx-0.7.3": common + [DRAFTER + "/config.json", OMLX_BIN_073],
        "omlx-0.7.3rc1": common + [DRAFTER + "/config.json", OMLX_BIN_073RC1],
        "omlx-0.7.4.dev1": common + [DRAFTER + "/config.json", OMLX_BIN_074DEV1],
        "mlx-serve-26.11.1": [CHECKPOINT + "/config.json", MLXSERVE_BIN_26111],
        "mlx-lm-server": common + [MLXLM_SERVER_BIN],
        "kapitsa-0.2.26": [KAPITSA_BIN_0226, KAPITSA_MODEL_DIR + "/config.json"],
        "kapitsa-0.2.31": [KAPITSA_BIN_0231, KAPITSA_MODEL_DIR + "/config.json"],
        "tf-new": common + [DRAFTER + "/config.json", TF_BIN],
        "mlxlm": common + [MAIN_PY],
        "omlx": common + [DRAFTER + "/config.json", OMLX_BIN],
        "splash": [
            SPLASH_BIN,
            "/Users/yuhuan/Library/Application Support/Splash/models/" + SPLASH_MODEL,
        ],
        "mtplx": common + [MTPLX_BIN],
        "llamacpp": [GGUF, GGUF_MTP, LLAMA_BIN],
    }[engine]


def missing_paths(engine: str) -> list[str]:
    return [p for p in required_paths(engine) if not Path(p).exists()]
