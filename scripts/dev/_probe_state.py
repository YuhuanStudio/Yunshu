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
            print("STATE", dict(gdn=dict(gdn_prefill._STATE), nax=nax_prefill._enabled, attn=getattr(attn_prefill,'_STATE',getattr(attn_prefill,'_enabled',None)),
               stock=lane_linear.STOCK_ROWS, sp=lane_linear._SUM_POLICY, sr=lane_linear._SUM_REUSE,
               mtp={k:(v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in mtp_lane._STATE.items()}, dev=str(mx.default_device())), flush=True)
sys.exit(pytest.main(["-q","-p","no:cacheprovider","-p","no:randomly","-s",*sys.argv[1:]],plugins=[P()]))
