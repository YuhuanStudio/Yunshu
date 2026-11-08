import sys, pytest
class P:
    def pytest_runtest_setup(self, item):
        if "apc_manager_e2e" in item.nodeid and not getattr(P, "done", 0):
            P.done = 1
            from yunshu_engine.kernels import gdn_prefill, nax_prefill, attn_prefill, lane_linear
            from yunshu_engine import mtp_lane
            import mlx.core as mx
            import os
            act = os.environ.get("ACT", "")
            if act == "stock0": lane_linear.set_stock_rows(0)
            if act == "gdnon": gdn_prefill._STATE["enabled"] = True
            if act == "gdnuninstalled": print("n/a")
            from mlx_vlm.models.qwen3_5 import gated_delta as gd
            import mlx_vlm.models.qwen3_5.language as lang
            if act == "unwrap":
                w = gd.gated_delta_kernel
                cl = {n: c.cell_contents for n, c in zip(w.__code__.co_freevars, w.__closure__)}
                print("FREE", list(cl), flush=True)
                gd.gated_delta_kernel = cl["step"]
            if act == "nopacked":
                gd._gated_delta_kernel_packed = None
            if act == "reset":
                w = gd.gated_delta_kernel
                if getattr(w, "__closure__", None):
                    cl = {n: c.cell_contents for n, c in zip(w.__code__.co_freevars, w.__closure__)}
                    gd.gated_delta_kernel = cl["step"]
                gdn_prefill._STATE.update(installed=False, enabled=False)
                lane_linear.set_stock_rows(0)
            print("GD", gd.gated_delta_kernel.__qualname__, gd._gated_delta_kernel_packed, getattr(lang,'gated_delta_update',None), getattr(lang.gated_delta_update,'__module__',None) if hasattr(lang,'gated_delta_update') else None, flush=True)
            print("STATE", dict(gdn=dict(gdn_prefill._STATE), nax=nax_prefill._enabled, attn=getattr(attn_prefill,'_STATE',getattr(attn_prefill,'_enabled',None)),
               stock=lane_linear.STOCK_ROWS, sp=lane_linear._SUM_POLICY, sr=lane_linear._SUM_REUSE,
               mtp={k:(v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in mtp_lane._STATE.items()}, dev=str(mx.default_device())), flush=True)
sys.exit(pytest.main(["-q","-p","no:cacheprovider","-p","no:randomly","-s",*sys.argv[1:]],plugins=[P()]))
