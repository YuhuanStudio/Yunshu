"""Launch review (2026-11-04): each competitor's launch recipe against its own help / docs.

Rule: a launch carries the settings a normal user following the engine's own documentation would use for speed
(lossless), and none of the engine's LOSSY switches (lossy options are user choices with a lossless default).
Every entry below was read from the pinned binary's --help (or the cited source file) and compared with the
launch value. Changes to a recipe must update this table on purpose; it is the record of the review.

Documented speed flags with NO recommendation in the engine's own docs stay at the engine's default (listed in
DEFAULTS_KEPT): the review did not guess a better value.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import bench_engines as be  # noqa: E402

ENV = {}


def _launch(engine, **kw):
    return be.build_launch(engine, 18990, "/tmp/h", ENV, **kw)


def _val(cmd, flag):
    cmd = list(cmd)
    return cmd[cmd.index(flag) + 1]


# launch values that matter for speed, with the source of the choice
PRESENT = {
    # TensorFold 1.2.1 / 1.1.8: serve --help carries names only (no descriptions); RUNTIME.md names no speed setting.
    "tf-1.2.1": {"--drafter": be.DRAFTER, "--no-thinking": None, "--parallel": "1", "--context": "140000"},
    "tf-1.1.8": {"--drafter": be.DRAFTER, "--no-thinking": None, "--parallel": "1", "--context": "140000"},
    # Splash 1.5.0 serve --help: KV format default int8 (lossy), launch bf16 = lossless; DFlash2 draft explicit;
    # --max-context sized to the cell; persistent SSD cache on (serve_options.py:555-562)
    "splash-1.5.0": {"--kv-format": "bf16", "--draft-model": be.DRAFTER, "--persistent-cache": None},
    # oMLX 0.7.3 serve --help: --max-concurrent-requests default 8 (same), hot cache default 0 (launch 8GB, a
    # documented speed option), SSD cache on; DFlash via model settings (dflash_verify_mode adaptive, block size and
    # draft window at the checkpoint defaults: model_settings.py:321-339)
    "omlx-0.7.3": {"--max-concurrent-requests": "8", "--hot-cache-max-size": "8GB", "--paged-ssd-cache-max-size": "20GB"},
    # mlx-serve 26.11.1 serve --help: MTP head auto-loaded (default on Metal), depth adaptive, prefix cache RAM default
    # on, SSD tier opt-in (docs/cli.md:90): on
    "mlx-serve-26.11.1": {"--ctx-size": "140000", "--prefix-cache-disk": "20GB"},
    # llama.cpp b12061 llama-server --help: --spec-type draft-mtp with the MTP GGUF head, flash attention auto
    # (default), --cache-ram 8192 MiB default, --spec-draft-n-max 3 default
    "llama-cpp-b12061": {"--spec-type": "draft-mtp", "-c": "140000", "--reasoning": "off"},
    # mlx-vlm 0.7.11: DFlash drafter explicit; prefix cache is OFF by default (runtime_config.py:76) -> APC_ENABLED=1
    "mlx-vlm-server": {"--draft-kind": "dflash", "--draft-model": be.DRAFTER},
    # kapitsa 0.2.26 serve --help: DFlash2 drafter auto (default), slots default up to 4, disk cache auto (default on),
    # context default = the model's limit (256K here, log line 'Context: 256K tokens per conversation')
    "kapitsa-0.2.26": {"--dflash": "auto"},
}

# flags a launch must NOT carry: the engine's lossy switches (user options, never a benchmark default)
LOSSY_ABSENT = {
    "mlx-serve-26.11.1": ["--ane-prefill", "--int8-qk", "--mtp-typical", "--mtp-tokenv3", "--kv-quant", "--lm-head-mxfp8", "--ane"],
    "splash-1.5.0": ["--state-format"],  # bf16 state is lossy; the default fp32 stays
    "mlx-vlm-server": ["--kv-bits", "--kv-quant-scheme", "--quantized-kv-start"],
    "llama-cpp-b12061": ["-ctk", "-ctv", "--cache-type-k", "--cache-type-v"],
    "tf-1.2.1": ["--drafter-bits"],
    "omlx-0.7.3": [],
}

# documented speed flags the launch leaves at the engine's default because its own docs recommend no other value
DEFAULTS_KEPT = {
    "llama-cpp-b12061": ["-b (2048)", "-ub (512)", "-fa (auto)", "-np (auto)", "--cache-ram (8192 MiB)", "--spec-draft-n-max (3)", "-t / -tb"],
    "mlx-serve-26.11.1": ["--max-concurrent (1)", "--prefill-chunk (8192)", "--prefill-decode-share (0.25)", "--mtp-max-depth (adaptive)"],
    "splash-1.5.0": ["--decode-share (0.5)", "--queue-size (32)", "--verify-rows (auto)", "--disable-ane (off)"],
    "kapitsa-0.2.26": ["--slots (up to 4)", "--prefill-chunk (2048)", "--spec", "--mem-budget"],
    "mlx-vlm-server": ["--prefill-step-size (2048)", "--max-num-seqs (unbounded)", "--draft-block-size"],
    "omlx-0.7.3": ["--memory-guard (launch 96 GB, a ceiling)", "--initial-cache-blocks (256)"],
    "tf-1.2.1": ["--speed-up", "--keep-warm", "--prompt-cache-gib (16 GiB)"],
}


def test_each_launch_carries_the_reviewed_speed_settings():
    for engine, want in PRESENT.items():
        cmd = _launch(engine).cmd
        for flag, value in want.items():
            assert flag in cmd, (engine, flag)
            if value is not None:
                assert _val(cmd, flag) == value, (engine, flag, _val(cmd, flag), value)


def test_no_launch_carries_a_lossy_switch_of_its_engine():
    for engine, flags in LOSSY_ABSENT.items():
        cmd = _launch(engine).cmd
        for flag in flags:
            assert flag not in cmd, (engine, flag)


def test_mlx_vlm_enables_prefix_caching_and_every_engine_with_a_documented_disk_tier_turns_it_on():
    assert _launch("mlx-vlm-server").env["APC_ENABLED"] == "1"
    assert "--persistent-cache" in _launch("splash-1.5.0").cmd
    assert "--prefix-cache-disk" in _launch("mlx-serve-26.11.1").cmd
    assert "--paged-ssd-cache-dir" in _launch("omlx-0.7.3").cmd


def test_defaults_kept_are_listed_for_every_reviewed_engine():
    assert set(DEFAULTS_KEPT) >= {"llama-cpp-b12061", "mlx-serve-26.11.1", "splash-1.5.0", "kapitsa-0.2.26", "mlx-vlm-server"}
