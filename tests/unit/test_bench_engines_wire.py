"""CPU gates for the mlx-vlm-server and llama-cpp-b12061 harness entries (no GPU, no server).

The launch lines are checked as pure data; the engaged-mode regexes are checked on the log lines the
real servers print (mlx_vlm/server/generation.py for mlx-vlm 0.7.11, llama.cpp b12061 server).
"""

from scripts.research import bench_engines as be


def _launch(engine, tmp_path, ctx=1024):
    return be.build_launch(engine, 18995, tmp_path / "home", {"PATH": "/usr/bin"}, ctx)


def test_mlx_vlm_server_launch_uses_dflash2_drafter(tmp_path):
    lc = _launch("mlx-vlm-server", tmp_path)
    cmd = lc.cmd
    assert cmd[0] == be.MLXVLM_PY_0711
    assert cmd[1:3] == ["-m", "mlx_vlm.server"]
    assert cmd[cmd.index("--model") + 1] == be.CHECKPOINT
    assert cmd[cmd.index("--port") + 1] == "18995"
    assert cmd[cmd.index("--draft-model") + 1] == be.DRAFTER
    assert cmd[cmd.index("--draft-kind") + 1] == "dflash"
    # real-1104 fairness fix: mlx-vlm runs with its prefix cache on (APC_ENABLED=1), like every other engine
    assert lc.flags == {
        "speculative": "dflash",
        "draft_kind": "dflash",
        "apc": "APC_ENABLED=1 (prefix cache on, disk persistence default)",
    }
    assert be.ENGINES["mlx-vlm-server"].expected_mode == "dflash"
    assert be.ENGINES["mlx-vlm-server"].drafter == be.DRAFTER
    assert be.stub_reason("mlx-vlm-server") is None


def test_mlx_vlm_server_engaged_only_on_drafter_ready():
    loaded = (
        "Loading speculative drafter (dflash): /Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2\n"
        "Auto-detected speculative draft kind: dflash\n"
        "Drafter ready; speculative decoding enabled.\n"
    )
    assert be.detect_mode("mlx-vlm-server", loaded) == "dflash"
    # incompatible drafter: AR fallback, must not count as engaged
    fallback = (
        "Loading speculative drafter (dflash): /x\n"
        "Speculative drafter is incompatible with the target model; falling back to autoregressive "
        "generation: bad\n"
    )
    assert be.detect_mode("mlx-vlm-server", fallback) is None
    # plain start (no draft) is not engaged either
    assert (
        be.detect_mode(
            "mlx-vlm-server", "Language model ready, continuous batching enabled.\n"
        )
        is None
    )


def test_mlx_vlm_server_drafter_is_a_required_path():
    paths = be.required_paths("mlx-vlm-server")
    assert be.DRAFTER + "/config.json" in paths
    assert be.MLXVLM_PY_0711 in paths


def test_llama_cpp_b12061_launch_is_ud_q4_k_m_with_mtp_head(tmp_path):
    lc = _launch("llama-cpp-b12061", tmp_path, ctx=131072)
    cmd = lc.cmd
    assert cmd[0] == be.LLAMA_BIN_B12061
    assert cmd[cmd.index("-m") + 1] == be.GGUF
    assert be.GGUF.endswith("Qwen3.8-27B-UD-Q4_K_M.gguf")
    assert cmd[cmd.index("-md") + 1] == be.GGUF_MTP
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("-c") + 1] == "131072"
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert cmd[cmd.index("--port") + 1] == "18995"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    # the launch records its slot count (-np, concurrency cells raise it)
    assert lc.flags == {"spec_type": "draft-mtp", "reasoning": "off", "parallel": 1}
    assert be.ENGINES["llama-cpp-b12061"].checkpoint == be.GGUF
    assert be.ENGINES["llama-cpp-b12061"].expected_mode == "mtp"


def test_llama_cpp_b12061_engaged_needs_drafted_reply():
    # b12061 prints per-request acceptance; a loaded MTP head alone is not evidence for this build
    loaded_only = (
        "loading draft model '/Volumes/P5Plus/models/unsloth/Qwen3.8-27B-GGUF/MTP/"
        "mtp-Qwen3.8-27B-Q4_0.gguf'\n"
    )
    assert be.detect_mode("llama-cpp-b12061", loaded_only) is None
    drafted = loaded_only + "draft acceptance = 0.71 (  5 accepted /  7 generated)\n"
    assert be.detect_mode("llama-cpp-b12061", drafted) == "mtp"


def test_both_engines_registered_for_manifest():
    for eng in ("mlx-vlm-server", "llama-cpp-b12061"):
        assert eng in be.MANIFEST_ENGINES
        assert eng in be.ENGINES
        assert eng not in be.UNSUPPORTED_ENGINES
        assert be.stub_reason(eng) is None
