import sys, pytest, json, types, os
class P:
    def pytest_runtest_setup(self, item):
        if "apc_manager_e2e" in item.nodeid and not getattr(P, "done", 0):
            P.done = 1
            out = {}
            for name, m in list(sys.modules.items()):
                if not (name.startswith("yunshu_") or name.startswith("mlx_vlm") or name.startswith("mlx_lm")):
                    continue
                for k, v in list(vars(m).items()):
                    if k.startswith("__"): continue
                    if isinstance(v, (int, float, str, bool, type(None))):
                        out[f"{name}.{k}"] = repr(v)
                    elif isinstance(v, (dict, list, set, tuple)):
                        try: out[f"{name}.{k}"] = f"{type(v).__name__}[{len(v)}]" + (repr(v)[:150] if len(repr(v)) < 400 else "")
                        except Exception: pass
                    elif callable(v) and hasattr(v, "__qualname__"):
                        out[f"{name}.{k}"] = "fn:" + v.__qualname__ + ":" + str(getattr(v, "__module__", ""))
                    if isinstance(v, type) and name.startswith(("mlx_vlm.models.qwen3_5", "yunshu_engine")):
                        for a, f in list(vars(v).items()):
                            if callable(f) and hasattr(f, "__qualname__"):
                                out[f"{name}.{k}.{a}"] = "fn:" + f.__qualname__
            out = {k: v for k, v in out.items()}
            json.dump(out, open(os.environ["OUT"], "w"), indent=0, sort_keys=True, default=str)
sys.exit(pytest.main(["-q","-p","no:cacheprovider","-p","no:randomly",*sys.argv[1:]],plugins=[P()]))
