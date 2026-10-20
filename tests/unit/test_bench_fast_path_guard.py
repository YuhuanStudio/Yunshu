"""A bench run whose server turned a tuned fast path off fails instead of measuring it."""


def test_disabled_fast_path_is_found_in_the_server_log():
    import tfbench

    log = (
        "INFO VLM batch runner: ... draft=dflash\n"
        "WARNING yunshu_engine.kernels.ragged_kv: ragged KV off: Qwen3.5 attention computes"
        " in mlx.core.float16, the kernels take bf16\n"
    )
    assert tfbench.disabled_fast_path(log).startswith(
        "ragged KV off: Qwen3.5 attention"
    )
    assert tfbench.disabled_fast_path("INFO all good\n") is None
