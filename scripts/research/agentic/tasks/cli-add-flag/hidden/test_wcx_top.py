import ast
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(os.environ["AGENTIC_TASK_DIR"])


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "wcx", *args], capture_output=True, text=True, cwd=TASK
    )


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_default_output_unchanged(tmp_path):
    f = write(tmp_path, "a.txt", "one two\nthree\n")
    r = run(f)
    assert r.returncode == 0 and r.stdout == f"2\t3\t14\t{f}\n"


def test_top_basic_sorted(tmp_path):
    f = write(tmp_path, "a.txt", "the cat the dog the cat bird\n")
    r = run("--top", "2", f)
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    assert lines[0].endswith(f)
    assert lines[1:] == ["3\tthe", "2\tcat"]


def test_top_tie_break_alphabetical(tmp_path):
    f = write(tmp_path, "a.txt", "b a c b a c d\n")
    r = run("--top", "4", f)
    assert r.stdout.splitlines()[1:] == ["2\ta", "2\tb", "2\tc", "1\td"]


def test_top_across_files_and_total(tmp_path):
    a = write(tmp_path, "a.txt", "x y x\n")
    b = write(tmp_path, "b.txt", "y x z\n")
    r = run("--top", "2", a, b)
    lines = r.stdout.splitlines()
    assert lines[2].endswith("total")
    assert lines[3:] == ["3\tx", "2\ty"]


def test_case_sensitive_by_default(tmp_path):
    f = write(tmp_path, "a.txt", "Go go GO Go\n")
    r = run("--top", "5", f)
    assert r.stdout.splitlines()[1:] == ["2\tGo", "1\tGO", "1\tgo"]


def test_ignore_case(tmp_path):
    f = write(tmp_path, "a.txt", "Go go GO Go\n")
    for flag in ("--ignore-case", "-i"):
        r = run("--top", "5", flag, f)
        assert r.stdout.splitlines()[1:] == ["4\tgo"]


def test_word_definition(tmp_path):
    f = write(tmp_path, "a.txt", "don't stop, don't! a1 a1-b\n")
    r = run("--top", "5", f)
    assert r.stdout.splitlines()[1:] == ["2\ta1", "2\tdon't", "1\tb", "1\tstop"]


def test_top_invalid(tmp_path):
    f = write(tmp_path, "a.txt", "a\n")
    for bad in ("0", "-3", "x"):
        r = run("--top", bad, f)
        assert r.returncode == 2, bad


def test_readme_documents_options():
    text = (TASK / "README.md").read_text()
    assert "--top" in text and "--ignore-case" in text


def test_agent_added_tests():
    tree = ast.parse((TASK / "tests" / "test_cli.py").read_text())
    names = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
    ]
    assert len(names) >= 4
    src = (TASK / "tests" / "test_cli.py").read_text()
    assert "--top" in src
