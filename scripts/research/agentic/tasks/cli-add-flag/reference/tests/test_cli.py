import subprocess
import sys


def run(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "wcx", *args], capture_output=True, text=True, cwd=cwd)


def test_single_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one two\nthree\n")
    r = run(str(f))
    assert r.returncode == 0
    assert r.stdout == f"2\t3\t14\t{f}\n"


def test_total_line(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x\n")
    b.write_text("y z\n")
    r = run(str(a), str(b))
    assert r.stdout.splitlines()[-1] == "2\t3\t6\ttotal"


def test_top(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a b a\n")
    assert run("--top", "1", str(f)).stdout.splitlines()[1:] == ["2\ta"]


def test_top_ignore_case(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("A a\n")
    assert run("--top", "1", "-i", str(f)).stdout.splitlines()[1:] == ["2\ta"]
