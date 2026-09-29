"""Apple Neural Engine latency / throughput for drafter-sized workloads.

Runs in an isolated venv (coremltools; not the repo .venv):

    /Volumes/P5Plus/yunshu-test-envs/ane/bin/python scripts/research/hw/ane_probe.py \
        --work /Volumes/P5Plus/yunshu-test-cache/ane --cases linear,mlp,layer

Builds MIL programs directly (no torch): a linear of MLP-up size, a SwiGLU MLP, and a full
transformer layer with the Qwen3.8 / DFlash2 drafter dimensions (hidden 5120, inter 17408,
GQA 32/8 x hd128), with fp16 and int8 (per-channel) weights. Reports per-call latency for
1..16 tokens on CPU_AND_NE vs CPU_AND_GPU vs CPU_ONLY, and (via MLComputePlan) which ops
actually landed on the ANE.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import coremltools as ct  # noqa: E402
from _common import Out  # noqa: E402
from coremltools.converters.mil import Builder as mb  # noqa: E402, N813
from coremltools.converters.mil.mil import types  # noqa: E402

H, INTER = 5120, 17408
# attention shape of the block: DFlash2 drafter (32/8 x 128) or the Qwen3.8-27B target
# (24/4 x 256 with the q output gate)
ARCHS = {"dflash": (32, 8, 128, 4096), "q27": (24, 4, 256, 12288)}
NH, NKV, HD, QOUT = ARCHS["dflash"]


def set_arch(name):
    global NH, NKV, HD, QOUT
    NH, NKV, HD, QOUT = ARCHS[name]


def w(rng, *shape, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float16)


def build_linear(M, K, N, rng):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, M, K), dtype=types.fp16)],
              opset_version=ct.target.iOS18)
    def prog(x):
        return mb.linear(x=x, weight=w(rng, N, K), name="y")

    return prog


def rms(x, g, name):
    ms = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
    r = mb.rsqrt(x=mb.add(x=ms, y=np.float16(1e-6)), epsilon=np.float16(1e-6))
    return mb.mul(x=mb.mul(x=x, y=r), y=g, name=name)


def mlp(x, rng, i):
    g = mb.linear(x=x, weight=w(rng, INTER, H), name=f"gate{i}")
    u = mb.linear(x=x, weight=w(rng, INTER, H), name=f"up{i}")
    a = mb.mul(x=mb.silu(x=g), y=u)
    return mb.linear(x=a, weight=w(rng, H, INTER), name=f"down{i}")


def attention(x, rng, i, M, L):
    """GQA attention over M new tokens + L cached context tokens (K/V as constants of
    zeros: only the matmul volume matters here)."""
    q = mb.linear(x=x, weight=w(rng, QOUT, H), name=f"q{i}")
    gate = None
    if QOUT > NH * HD:  # Qwen3.5 output gate: q_proj emits [q | gate]
        q, gate = mb.split(x=q, num_splits=2, axis=-1)
    k = mb.linear(x=x, weight=w(rng, NKV * HD, H), name=f"k{i}")
    v = mb.linear(x=x, weight=w(rng, NKV * HD, H), name=f"v{i}")
    q = mb.transpose(x=mb.reshape(x=q, shape=[1, M, NH, HD]), perm=[0, 2, 1, 3])
    k = mb.transpose(x=mb.reshape(x=k, shape=[1, M, NKV, HD]), perm=[0, 2, 1, 3])
    v = mb.transpose(x=mb.reshape(x=v, shape=[1, M, NKV, HD]), perm=[0, 2, 1, 3])
    rep = NH // NKV
    k = mb.reshape(x=mb.tile(x=mb.expand_dims(x=k, axes=[2]), reps=[1, 1, rep, 1, 1]),
                   shape=[1, NH, M, HD])
    v = mb.reshape(x=mb.tile(x=mb.expand_dims(x=v, axes=[2]), reps=[1, 1, rep, 1, 1]),
                   shape=[1, NH, M, HD])
    o = mb.scaled_dot_product_attention(query=q, key=k, value=v)
    o = mb.reshape(x=mb.transpose(x=o, perm=[0, 2, 1, 3]), shape=[1, M, NH * HD])
    if gate is not None:
        o = mb.mul(x=o, y=mb.sigmoid(x=gate))
    return mb.linear(x=o, weight=w(rng, H, NH * HD), name=f"o{i}")


def build_layers(M, n_layers, rng):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, M, H), dtype=types.fp16)],
              opset_version=ct.target.iOS18)
    def prog(x):
        for i in range(n_layers):
            g1 = np.ones(H, dtype=np.float16)
            x = mb.add(x=x, y=attention(rms(x, g1, f"n1_{i}"), rng, i, M, 0))
            x = mb.add(x=x, y=mlp(rms(x, g1, f"n2_{i}"), rng, i))
        return x

    return prog


def convert(prog):
    return ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18)


def quantize(ml, bits, block=0):
    import coremltools.optimize.coreml as cto

    if bits == "pal4":
        return cto.palettize_weights(ml, cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(nbits=4, mode="uniform")))

    if bits == 8:
        cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                          granularity="per_channel")
    else:
        cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
                                          granularity="per_block", block_size=block or 32)
    return cto.linear_quantize_weights(ml, cto.OptimizationConfig(global_config=cfg))


def ane_placement(path):
    """Fraction of ops that MLComputePlan reports as ANE-preferred."""
    try:
        from coremltools.models.compute_plan import MLComputePlan

        compiled = ct.utils.compile_model(str(path))
        plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        prog = plan.model_structure.program
        counts = {"ane": 0, "gpu": 0, "cpu": 0}
        for fn in prog.functions.values():
            for op in fn.block.operations:
                if op.operator_name in ("const", "constexpr_affine_dequantize"):
                    continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    continue
                name = type(usage.preferred_compute_device).__name__.lower()
                counts["ane" if "neural" in name else "gpu" if "gpu" in name else "cpu"] += 1
        return counts
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:120]}


def bench(model_path, M, units, iters):
    t0 = time.time()
    m = ct.models.MLModel(str(model_path), compute_units=units)
    load_s = time.time() - t0
    spec = m.get_spec()
    in_name = spec.description.input[0].name
    shape = tuple(spec.description.input[0].type.multiArrayType.shape)
    feed = {in_name: np.random.randn(*shape).astype(np.float32)}
    for _ in range(3):
        m.predict(feed)
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        m.predict(feed)
        s.append(time.perf_counter() - t)
    return load_s, statistics.median(s), min(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/Volumes/P5Plus/yunshu-test-cache/ane")
    ap.add_argument("--cases", default="linear,mlp,layer")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--units", default="ne,gpu,cpu")
    ap.add_argument("--ms", default="1,8,16")
    ap.add_argument("--arch", default="dflash", choices=list(ARCHS))
    ap.add_argument("--precs", default="fp16,int8,int4")
    a = ap.parse_args()
    out = Out("ane_probe")
    set_arch(a.arch)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    U = {"ne": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU,
         "cpu": ct.ComputeUnit.CPU_ONLY}
    rng = np.random.default_rng(0)
    cases = a.cases.split(",")
    for M in [int(v) for v in a.ms.split(",")]:
        specs = []
        if "linear" in cases:
            specs.append(("linear_5120x17408", lambda M=M: build_linear(M, H, INTER, rng), 2 * M * H * INTER, H * INTER))
        if "layer" in cases:
            tag = "layer_x1" if a.arch == "dflash" else "layer27_x1"
            specs.append((tag, lambda M=M: build_layers(M, 1, rng), None, None))
        if "layers5" in cases:
            specs.append(("layer_x5", lambda M=M: build_layers(M, 5, rng), None, None))
        for name, mk, _flops, _nparams in specs:
            for prec in a.precs.split(","):
                pth = work / f"{name}_M{M}_{prec}.mlpackage"
                try:
                    if not pth.exists():
                        ml = convert(mk())
                        if prec == "int8":
                            ml = quantize(ml, 8)
                        elif prec == "int4":
                            ml = quantize(ml, 4)
                        elif prec == "pal4":
                            ml = quantize(ml, "pal4")
                        ml.save(str(pth))
                        del ml
                    placement = ane_placement(pth)
                    row = {}
                    for u in a.units.split(","):
                        ld, med, mn = bench(pth, M, U[u], a.iters)
                        row[u] = {"load_s": round(ld, 1), "ms_median": round(med * 1e3, 3),
                                  "ms_min": round(mn * 1e3, 3)}
                    size_mb = sum(f.stat().st_size for f in pth.rglob("*") if f.is_file()) / 2**20
                    out(kind="ane", case=name, M=M, weights=prec, pkg_MB=round(size_mb, 1),
                        placement=placement, **row)
                except Exception as e:  # noqa: BLE001
                    out(kind="ane", case=name, M=M, weights=prec, err=str(e)[:300])


if __name__ == "__main__":
    main()
