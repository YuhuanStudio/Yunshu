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
    ]
}
# the legacy tfbench engines keep their old code path
LEGACY = ("yunshu", "tf-old", "tf-new")
TF_ENGINES = ("tf-new", "tf-old")


def is_new_engine(engine: str) -> bool:
    """Engines launched by build_launch() (everything except tfbench's original three)."""
    return engine in ENGINES and engine not in LEGACY


def scrubbed_env(environ: dict, extra: dict | None = None) -> dict:
    env = {k: v for k, v in environ.items() if not k.startswith(SCRUB_PREFIXES)}
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


def yunshu_home_links(home: str | Path) -> dict:
    """A ``yunshu pull`` layout inside the isolated HOME: models dir org/name entries. The drafter lives
    next to the checkpoint under ~/.yunshu/models, which is where spec_select looks, so discovery is real."""
    base = Path(home) / ".yunshu" / "models"
    return {
        str(base / "Jundot" / "Qwen3.8-27B-oQ4e-mtp"): CHECKPOINT,
        str(base / "incoai" / "Qwen3.8-27B-DFlash2"): DRAFTER,
    }


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
        links = yunshu_home_links(home)
        model = str(Path(home) / ".yunshu/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
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
            {},
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
        env = scrubbed_env(environ, {"NO_PROXY": "127.0.0.1"})
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
            {},
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
    raise KeyError(engine)


# ---- engaged-speculative-mode detection (fail closed) -----------------------------------------------


def detect_mode(engine: str, log: str, probe_text: str | None = None) -> str | None:
    """The speculative mode a started server actually runs, from its log (and a status probe), or None
    when there is no evidence. Callers compare against Engine.expected_mode and refuse to measure."""
    e = ENGINES[engine]
    if e.kind == "yunshu":
        modes = re.findall(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b", log)
        return modes[-1] if modes else None
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
        if re.search(r"draft-mtp|speculative decoding.*mtp|mtp.*draft", log, re.I):
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
    "tf-new": ("0.6.1", None),
    "mlxlm": ("mlx-lm 0.32.0 (main venv)", None),
    "omlx": ("0.7.0 (git tag v0.7.0, own venv)", None),
    "splash": ("1.1.0 (Homebrew)", None),
    "mtplx": ("2.12.0 (own venv, mlx-lm 0.31.3)", None),
    "llamacpp": ("0.5.0-dev build 242", "836d57176dc699a726c55418e4f96b8ca628e1bf"),
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
    if e.kind == "yunshu":
        return common + [DRAFTER + "/config.json", YUNSHU_BIN]
    return {
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
