"""scripts/vendor/check_upstream.py: nothing credited or depended on can go untracked."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_upstream", ROOT / "scripts/vendor/check_upstream.py"
)
cu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cu)

MANIFEST = json.loads((ROOT / "vendor.json").read_text())


def test_pyproject_packages_reads_base_extras_and_groups(tmp_path):
    f = tmp_path / "pyproject.toml"
    f.write_text(
        """
[project]
name = "demo"
dependencies = ["Foo_Bar>=1.0", "uvicorn[standard]>=0.34"]
[project.optional-dependencies]
vision = ["pillow>=10", "demo[audio]"]
audio = ["mlx-audio>=0.5; sys_platform == 'darwin'"]
[dependency-groups]
dev = ["pytest>=9", {include-group = "other"}]
"""
    )
    pk = cu.pyproject_packages(f)
    assert pk["foo-bar"] == [">=1.0"]
    assert pk["uvicorn"] == [">=0.34"]
    assert pk["mlx-audio"] == [">=0.5"]
    assert pk["pytest"] == [">=9"]
    assert "demo" not in pk


def test_repo_pyproject_packages_cover_every_declared_dependency():
    pk = cu.pyproject_packages(ROOT / "pyproject.toml")
    for name in ("llguidance", "openai", "anthropic", "mlx-audio", "mlx-embeddings"):
        assert name in pk


def test_unregistered_header_is_flagged(tmp_path):
    (tmp_path / "python").mkdir()
    (tmp_path / "python/ported.py").write_text(
        "# Upstream (derived): a/b (MIT) x.py @ abc\nx = 1\n"
    )
    (tmp_path / "python/docstring.py").write_text(
        '"""Thing.\n\nAdapted from oMLX but simplified.\n"""\n'
    )
    (tmp_path / "python/clean.py").write_text('"""Derived from the config."""\n')
    manifest = {"derived": [{"path": "python/other.py"}]}
    assert cu.unregistered_headers(tmp_path, manifest) == [
        "python/docstring.py",
        "python/ported.py",
    ]
    manifest = {
        "derived": [{"path": "python/ported.py"}, {"path": "python/docstring.py"}]
    }
    assert cu.unregistered_headers(tmp_path, manifest) == []


def test_init_next_to_registered_files_is_a_package_note(tmp_path):
    (tmp_path / "python/k").mkdir(parents=True)
    (tmp_path / "python/k/__init__.py").write_text('"""Vendored from X."""\n')
    (tmp_path / "python/k/a.py").write_text("# Upstream (vendored): x\n")
    manifest = {"vendored": [{"path": "python/k/a.py"}]}
    assert cu.unregistered_headers(tmp_path, manifest) == []


def test_repo_has_no_unregistered_upstream_headers():
    assert cu.unregistered_headers(ROOT, MANIFEST) == []


def test_unclassified_clones(tmp_path):
    for n in ("a", "b", "c"):
        (tmp_path / n / ".git").mkdir(parents=True)
    manifest = {
        "watch": [{"clone": "reference/a"}],
        "watch_excluded": {"reference/b": "parked"},
    }
    assert cu.unclassified_clones(tmp_path, manifest) == ["reference/c"]
    assert cu.unclassified_clones(tmp_path / "missing", manifest) == []


def test_every_tracked_clone_is_watched():
    watched = {w["clone"] for w in MANIFEST["watch"]}
    for kind in cu.REGISTERED_KINDS:
        for e in MANIFEST[kind]:
            assert e["clone"] in watched, (e["path"], e["clone"])


def test_watch_and_excluded_are_disjoint_and_justified():
    watched = {w["clone"] for w in MANIFEST["watch"]}
    excluded = MANIFEST["watch_excluded"]
    assert not watched & set(excluded)
    assert all(len(why) > 10 for why in excluded.values())
    for w in MANIFEST["watch"]:
        assert w["globs"] and w["why"]


def test_reference_clones_are_all_classified():
    ref = cu.clone_dir("reference")
    if not ref.is_dir():
        pytest.skip("no reference/ clones in this checkout")
    assert cu.unclassified_clones(ref, MANIFEST) == []


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


def test_added_files_and_check_watch_work_for_any_repo(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "reference/other"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "pkg").mkdir()
    (repo / "pkg/a.py").write_text("a = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (repo / "pkg/a.py").write_text("a = 2\n")
    (repo / "pkg/b.py").write_text("b = 1\n")
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "change a, add b")

    assert cu.added_files(repo, base, "HEAD", ["pkg/*.py"], set()) == ["pkg/b.py"]
    assert cu.added_files(repo, base, "HEAD", ["pkg/*.py"], {"pkg/b.py"}) == []

    monkeypatch.setattr(cu, "ROOT", tmp_path)
    watch = [
        {
            "repo": "https://example.com/other",
            "clone": "reference/other",
            "globs": ["pkg/*.py"],
            "why": "test",
            "commit": base,
        }
    ]
    assert cu.check_watch(watch, set(), {}) == 1
    out = capsys.readouterr().out
    assert "1 commits on watched paths" in out
    assert "new files not vendored (1): pkg/b.py" in out


@pytest.mark.parametrize("kind", ["vendored", "derived"])
def test_review_does_not_rewrite_provenance_or_hide_future_changes(
    tmp_path, monkeypatch, kind
):
    repo = tmp_path / "reference/review"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    source = repo / "source.py"
    source.write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "copied")
    base = cu.git(repo, "rev-parse", "HEAD")
    source.write_text("x = 2\n")
    _git(repo, "commit", "-qam", "reviewed but deliberately retained")
    reviewed = cu.git(repo, "rev-parse", "HEAD")
    local = tmp_path / "local.py"
    local.write_text("x = 1\n")
    entry = dict(
        path="local.py",
        repo="https://example.com/review",
        clone="reference/review",
        commit=base,
        reviewed_commit=reviewed,
        review_reason="Keep the original arithmetic until a same-checkpoint gate.",
        upstream_path="source.py",
        upstream_paths=["source.py"],
        license="MIT",
    )
    monkeypatch.setattr(cu, "ROOT", tmp_path)

    def check():
        if kind == "vendored":
            return cu.check_vendored([entry], set())
        return cu.check_history([entry])

    assert check() == 0
    assert entry["commit"] == base
    source.write_text("x = 3\n")
    _git(repo, "commit", "-qam", "future change must be reported")
    assert check() == 1


def test_package_watch_uses_reviewed_sha_without_erasing_release_pin(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(cu, "clone_dir", lambda _: tmp_path)
    monkeypatch.setattr(cu, "upstream_ref", lambda _: "HEAD")
    monkeypatch.setattr(cu.importlib.metadata, "version", lambda _: "1.0")

    def git(_, *args):
        calls.append(args)
        if args[0] == "rev-parse":
            return "exists"
        if args[0] == "ls-tree":
            return "pkg/a.py"
        return ""

    monkeypatch.setattr(cu, "git", git)
    watch = dict(
        repo="example",
        clone="reference/x",
        globs=["pkg/*"],
        why="test",
        commit="release-base",
        pin_package="x",
        reviewed_commit="reviewed-sha",
        review_reason="Reviewed unreleased changes; retained released dependency.",
    )
    assert cu.check_watch([watch], set(), {}) == 0
    assert any("reviewed-sha..HEAD" in args for args in calls)
    assert watch["commit"] == "release-base"


def test_main_fails_for_unreviewed_watch_changes(monkeypatch):
    monkeypatch.setattr(cu, "check_patches", lambda *a, **kw: 0)
    monkeypatch.setattr(cu, "check_vendored", lambda *a: 0)
    monkeypatch.setattr(cu, "check_history", lambda *a: 0)
    monkeypatch.setattr(cu, "check_watch", lambda *a: 1)
    monkeypatch.setattr(cu, "check_packages", lambda *a, **kw: 0)
    monkeypatch.setattr(cu, "unregistered_headers", lambda *a: [])
    monkeypatch.setattr(cu, "unclassified_clones", lambda *a: [])
    monkeypatch.setattr("sys.argv", ["check_upstream", "--no-fetch"])
    with pytest.raises(SystemExit) as got:
        cu.main()
    assert got.value.code == 1


@pytest.mark.parametrize(
    "latest,blocked", [("2.1.1", True), ("2.1.2", True), ("2.1.1", False)]
)
def test_package_review_only_covers_exact_version_with_live_blocker(
    monkeypatch, latest, blocked
):
    import io

    monkeypatch.setattr(cu.importlib.metadata, "version", lambda _: "1.33.0")
    monkeypatch.setattr(
        cu.importlib.metadata,
        "requires",
        lambda _: ["huggingface-hub<2"] if blocked else ["huggingface-hub>=1"],
    )
    monkeypatch.setattr(
        cu.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps({"info": {"version": latest}}).encode()),
    )
    reviews = {
        "huggingface-hub": {
            "installed_version": "1.33.0",
            "latest_version": "2.1.1",
            "blocked_by": ["tokenizers"],
            "reason": "Current tokenizer requires hub<2.",
        }
    }
    assert cu.check_packages({"huggingface-hub": [">=1"]}, reviews) == (
        0 if latest == "2.1.1" and blocked else 1
    )


@pytest.mark.parametrize(
    ("installed", "latest", "spec", "expected"),
    [
        ("2.1.0", "2.3.1", "==2.1.0", ["pyproject: ==2.1.0"]),
        ("2.1.0", "2.3.1", ">=2.1.0", []),  # pin lifted: the review no longer applies
        ("2.1.0", "2.4.0", "==2.1.0", []),  # newer release than the one reviewed
    ],
)
def test_package_review_covers_a_deliberate_pyproject_pin(
    installed, latest, spec, expected
):
    reviews = {
        "trafilatura": {
            "installed_version": "2.1.0",
            "latest_version": "2.3.1",
            "blocked_by": ["pyproject"],
            "reason": "2.3.1 escapes underscores in extracted markdown.",
        }
    }
    assert (
        cu.reviewed_package_blockers("trafilatura", installed, latest, reviews, spec)
        == expected
    )
