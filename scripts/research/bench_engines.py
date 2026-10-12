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

import json
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
TF_BIN = str(ENVS / "tensorfold-0.6.1/bin/tensorfold")
MTPLX_BIN = str(ENVS / "mtplx/.venv/bin/mtplx")
OMLX_BIN = str(ENVS / "omlx-bench/venv/bin/omlx")
LLAMA_BIN = str(ENVS / "llamacpp/build/bin/llama-server")
SPLASH_BIN = "/opt/homebrew/bin/splash"
# refsync (2026-10-12): latest engines in their own envs under ENVS (the ids above stay for the 0.1.4 snapshot)
FN_CHECKPOINT = "/Volumes/P5Plus/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"
FN_CHECKPOINT_PLE = "/Volumes/P5Plus/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp-ple-ssd"
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
    ]
}
# the legacy tfbench engines keep their old code path
LEGACY = ("yunshu", "tf-old", "tf-new")
TF_ENGINES = ("tf-new", "tf-old")


def is_new_engine(engine: str) -> bool:
    """Engines launched by build_launch() (everything except tfbench's original three)."""
    return engine in ENGINES and engine not in LEGACY


def scrubbed_env(environ: dict, extra: dict | None = None) -> dict:
    env = {
        k: v
        for k, v in environ.items()
        if not k.startswith(SCRUB_PREFIXES) and k not in ("PYTHONPATH", "PYTHONHOME")
    }
    env.update(extra or {})
    return env


def omlx_model_settings(model_id: str) -> dict:
    return {
        "version": 1,
        "models": {model_id: {"dflash_enabled": True, "dflash_draft_model": DRAFTER}},
    }


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
                environ.get("TFB_YUNSHU_BIN", YUNSHU_BIN),
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
        flags = {
            "dflash_enabled": True,
            "dflash_draft_model": DRAFTER,
            "memory_guard_gb": 96,
            "max_concurrent_requests": 8,
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
                "--max-concurrent-requests",
                "8",
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
            {str(settings): json.dumps(omlx_model_settings(mid))},
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
        if engine == "omlx-latest":
            launch.cmd[0] = OMLX_BIN_LATEST
        elif engine == "mtplx-latest":
            launch.cmd[0] = MTPLX_BIN_LATEST
        return launch
    if engine == "splash-latest":
        rev = "a" * 40
        hub = (
            Path(home)
            / ".cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit"
        )
        return Launch(
            [
                SPLASH_BIN_LATEST,
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
    if engine == "tf-native":
        return Launch(
            [
                TF_NATIVE_BIN,
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
                "--context",
                str(int(ctx_tokens)),
            ],
            scrubbed_env(environ, base_env),
            {"drafter": "DFlash2 (explicit --drafter)", "thinking": "off"},
            {},
            {},
        )
    if engine in ("llamacpp-mtp", "llamacpp-dflash"):
        dflash = engine == "llamacpp-dflash"
        ctx = int(ctx_tokens)
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
    raise KeyError(engine)


REFSYNC_ALIASES = {
    "omlx-latest": "omlx",
    "mtplx-latest": "mtplx",
}


# ---- engaged-speculative-mode detection (fail closed) -----------------------------------------------


def detect_mode(engine: str, log: str, probe_text: str | None = None) -> str | None:
    """The speculative mode a started server actually runs, from its log (and a status probe), or None
    when there is no evidence. Callers compare against Engine.expected_mode and refuse to measure."""
    e = ENGINES[engine]
    if e.kind == "yunshu":
        modes = re.findall(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b", log)
        return modes[-1] if modes else None
    if engine == "tf-native":
        # 1.0.x native log: a drafter line naming DFlash2 and its block size (confirmed by the pilot)
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
            if re.search(r"Lightning MTP enabled for|MTP path activated|Speculative backend selected for [^\n]*Lightning MTP", log)
            else None
        )
    legacy_engine = engine
    engine = {
        "omlx-latest": "omlx",
        "mtplx-latest": "mtplx",
        "splash-latest": "splash",
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
        loose = legacy_engine != "llamacpp"
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
