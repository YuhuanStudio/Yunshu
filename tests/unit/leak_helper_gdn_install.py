"""Run explicitly by test_prefill_kernel_state_probe (not collected on its own): installs the GDN prefill kernel."""


def test_install():
    import mlx.core as mx

    if not hasattr(mx.fast, "gated_delta_update"):
        return
    from yunshu_engine.kernels import gdn_prefill

    assert gdn_prefill.install()
