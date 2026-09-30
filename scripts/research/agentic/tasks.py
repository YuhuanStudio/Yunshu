"""Task fixtures: custom tasks in ``tasks/`` plus Aider-polyglot Exercism Python exercises.

A task is a directory::

    task.json      {"id", "title", "prompt", "test": [argv...], "protected": [files], "generate": "gen.py"?}
    repo/          starting files copied into the agent's working directory
    hidden/        tests the agent never sees; copied to ``_hidden_tests/`` only at grading time
    reference/     an overlay that solves the task (used by ``check-tasks`` to validate the tests)

Polyglot exercises are materialized from the cloned dataset (never committed): the exercise's
docs, stub and test file go in the working directory (as in Aider, the agent sees the tests), and
the pristine test file is restored before grading so it cannot be edited into passing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
POLYGLOT_ROOT = Path(
    os.environ.get(
        "AGENTIC_POLYGLOT",
        "/Volumes/P5Plus/datasets/agentic/polyglot-benchmark/python/exercises/practice",
    )
)
PYTHON = "/Volumes/P5Plus/yunshu-test-envs/agentic-clis/pyenv/bin/python"

POLYGLOT = [
    "proverb",
    "pig-latin",
    "phone-number",
    "wordy",
    "transpose",
    "list-ops",
    "robot-name",
    "book-store",
    "two-bucket",
    "bowling",
    "forth",
    "grep",
]

POLYGLOT_PROMPT = (
    "Implement the exercise in this directory. Read .docs/instructions.md"
    "{append} for the requirements. Write your solution in {solution}; the tests are in "
    "{test} (do not edit the tests). Run the tests with `python -m pytest -q {test}` and "
    "keep working until they all pass."
)


@dataclass
class Task:
    id: str
    title: str
    prompt: str
    test: list[str]
    src: Path  # directory holding repo/ (and hidden/, reference/)
    protected: list[str] = field(default_factory=list)
    generate: str | None = None
    kind: str = "custom"
    timeout_s: int | None = None

    def stage(self, workdir: Path):
        """Copy the starting repo into ``workdir`` (fresh)."""
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        repo = self.src / "repo"
        if repo.exists():
            shutil.copytree(repo, workdir, dirs_exist_ok=True)
        if self.generate:
            subprocess.run(
                [PYTHON, str(self.src / self.generate), str(workdir)],
                check=True,
                cwd=workdir,
            )

    def grade(self, workdir: Path, timeout: int = 300) -> tuple[bool, str]:
        """Restore protected files, add hidden tests, run them. Returns (passed, output tail)."""
        for rel in self.protected:
            shutil.copy2(self.src / "repo" / rel, workdir / rel)
        hidden = self.src / "hidden"
        target = workdir / "_hidden_tests"
        if target.exists():
            shutil.rmtree(target)
        if hidden.exists():
            shutil.copytree(hidden, target)
        env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": str(workdir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(workdir.parent / "grade-home"),
            "NO_PROXY": "127.0.0.1",
            "AGENTIC_TASK_DIR": str(workdir),
            "AGENTIC_TASK_SRC": str(self.src),
        }
        Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
        try:
            p = subprocess.run(
                [PYTHON if a == "python" else a for a in self.test],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (p.stdout + p.stderr)[-3000:]
            return p.returncode == 0, out
        except subprocess.TimeoutExpired as e:
            return (
                False,
                f"grading timeout after {timeout}s\n{(e.stdout or b'')[-500:]!r}",
            )


def _load_custom(d: Path) -> Task:
    j = json.loads((d / "task.json").read_text())
    return Task(
        id=j["id"],
        title=j["title"],
        prompt=j["prompt"],
        test=j["test"],
        src=d,
        protected=j.get("protected", []),
        generate=j.get("generate"),
        timeout_s=j.get("timeout_s"),
    )


def custom_tasks() -> list[Task]:
    return [
        _load_custom(d)
        for d in sorted(TASKS_DIR.iterdir())
        if (d / "task.json").exists()
    ]


def materialize_polyglot(slug: str, cache: Path) -> Task:
    """Build a task directory for one Exercism Python exercise under ``cache``."""
    ex = POLYGLOT_ROOT / slug
    meta = json.loads((ex / ".meta" / "config.json").read_text())
    solution = meta["files"]["solution"][0]
    test = meta["files"]["test"][0]
    d = cache / f"polyglot-{slug}"
    if d.exists():
        shutil.rmtree(d)
    repo = d / "repo"
    repo.mkdir(parents=True)
    shutil.copytree(ex / ".docs", repo / ".docs")
    shutil.copy2(ex / solution, repo / solution)
    shutil.copy2(ex / test, repo / test)
    append = (
        " and .docs/instructions.append.md"
        if (ex / ".docs" / "instructions.append.md").exists()
        else ""
    )
    prompt = POLYGLOT_PROMPT.format(append=append, solution=solution, test=test)
    (d / "task.json").write_text(
        json.dumps(
            dict(
                id=f"polyglot-{slug}",
                title=f"Exercism python/{slug}",
                prompt=prompt,
                test=["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
                protected=[test],
            )
        )
    )
    # reference overlay = the exercise's example solution
    ref = d / "reference"
    ref.mkdir()
    shutil.copy2(ex / ".meta" / "example.py", ref / solution)
    t = _load_custom(d)
    t.kind = "polyglot"
    return t


def all_tasks(cache: Path) -> dict[str, Task]:
    out = {t.id: t for t in custom_tasks()}
    if POLYGLOT_ROOT.exists():
        for slug in POLYGLOT:
            t = materialize_polyglot(slug, cache)
            out[t.id] = t
    return out


def dataset_commit() -> str | None:
    head = POLYGLOT_ROOT.parents[3] / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            refp = head.parent / ref.split()[1]
            if refp.exists():
                return refp.read_text().strip()
            for line in (head.parent / "packed-refs").read_text().splitlines():
                if line.endswith(ref.split()[1]):
                    return line.split()[0]
        return ref
    except OSError:
        return None


def select(spec: str, tasks: dict[str, Task]) -> list[Task]:
    if spec in ("all", "*"):
        return list(tasks.values())
    if spec == "custom":
        return [t for t in tasks.values() if t.kind == "custom"]
    if spec == "polyglot":
        return [t for t in tasks.values() if t.kind == "polyglot"]
    out = []
    for name in spec.split(","):
        name = name.strip()
        if name not in tasks:
            sys.exit(f"unknown task {name!r}; have {sorted(tasks)}")
        out.append(tasks[name])
    return out
