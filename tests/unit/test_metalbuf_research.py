import importlib.util
import pathlib

P = pathlib.Path(__file__).parents[2] / "scripts/research"


def load(name):
    spec = importlib.util.spec_from_file_location(name, P / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class A:
    def __init__(self, nb):
        self.nbytes = nb


def test_classify():
    m = load("metalbuf_serve")
    c = m.classify([A(10), A(5000), A(70000), A(5 << 20)])
    assert (c["n"], c["lt4k"], c["lt64k"], c["lt4m"], c["ge4m"]) == (4, 1, 1, 1, 1)


def test_take_tokens_size():
    m = load("metalbuf_repro")
    g = iter(["x" * 100] * 50)
    assert len(m.take_tokens(g, 450)) == 450


class B(A):
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "f"


def test_top_groups():
    m = load("metalbuf_serve")
    g = m.top_groups([B((2,)), B((2,)), B((3,))])
    assert g[0] == [[2], "f", 2]
