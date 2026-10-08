# Releasing Yunshu

Versions go up in small steps. A patch (`0.1.1`, `0.1.2`) carries fixes and improvements. A minor
(`0.2.0`) marks a real capability jump. Nothing is published except by the steps below, run by a
maintainer.

## One-time setup (maintainer)

1. **PyPI trusted publishing.** On pypi.org, add a trusted publisher for project `yunshu`:
   - owner `YuhuanStudio`, repository `Yunshu`
   - workflow `release.yml`, environment `pypi`

   No API token is stored anywhere.
2. **GitHub environment.** Create an environment named `pypi` under the repository settings and
   add yourself as a required reviewer. Every publish then waits for your approval.

## Cut a release

```bash
# 1. Start from a green main.
git switch main && git pull
just test && just lint
# ... and a passing release gate on this commit (docs/guides/RELEASE_GATE.md):
zsh scripts/release/gate.sh

# 2. Bump the version in pyproject.toml (one place; the CLI, /version and
#    MCP read it from the installed package metadata).
#    version = "0.1.1"

# 3. Turn "## [Unreleased]" in CHANGELOG.md into "## [0.1.1] - YYYY-MM-DD"
#    and start a new "## [Unreleased]" above it with the required headings,
#    an empty performance table and a compare footer to main. Leave lists empty
#    until changes land; do not invent highlights for an empty draft.

# 4. Build and check locally.
rm -rf dist && uv build
uvx twine check dist/*
uv tool install --force "yunshu[vision] @ file://$PWD/$(ls dist/yunshu-*.whl)" && yunshu doctor

# 5. Commit, tag, push. The tag triggers .github/workflows/release.yml.
git commit -am "Release 0.1.1"
git tag -a v0.1.1 -m "Yunshu 0.1.1"
git push origin main v0.1.1
```

The release workflow then does the following:

1. Checks that the tag matches `pyproject.toml` and that `CHANGELOG.md` has a `## [0.1.1]` entry.
2. Lints and runs the unit tests on an Apple Silicon runner.
3. Builds the sdist and wheel, runs `twine check`, and installs the wheel into a clean tool env.
4. Waits for approval on the `pypi` environment, then publishes to PyPI.
5. Opens a **draft** GitHub release with the changelog section as its notes. Review the draft and
   publish it. Publishing the release also runs the macOS unit-test job in `ci.yml`.

If something fails before step 4, nothing was published. Delete the tag, fix the problem, and tag
again:

```bash
git push --delete origin v0.1.1 && git tag -d v0.1.1
```

A version that reached PyPI cannot be re-uploaded. Fix forward with the next patch version.

## After the release

- Check that the install works: `uv tool install "yunshu[vision]"` in a clean shell, then
  `yunshu doctor`.
- Update the Homebrew formula in the shared tap: the `sha256` of the sdist PyPI now serves, then
  copy `packaging/homebrew/yunshu.rb` to `Formula/yunshu.rb` in
  [YuhuanStudio/homebrew-tap](https://github.com/YuhuanStudio/homebrew-tap)
  ([steps](packaging/homebrew/README.md)).


## Start the next cycle: sync dependencies and upstream

Right after a release, before new feature work, the next version starts from current upstream:

1. **Dependencies.** `uv lock --upgrade` (MLX, mlx-lm, mlx-vlm, transformers, llguidance and the rest), then
   `just test && just lint`. MLX releases can change numerics: re-check speculative output equals plain output on
   the 27B, the paired accuracy set agrees within one question, and decode / TTFT on the standard cells did not
   regress. Record the before / after in `docs/reports/PERF_TREND.md`.
2. **Vendored and patched code.** Pull every reference clone and run `just vendor-check`: each vendored kernel,
   ported file and upstream monkeypatch in `vendor.json` whose upstream source changed is reviewed, re-synced or
   re-justified. Drop local patches that upstream now makes unnecessary.
3. **Upstream changes worth taking.** Read the release notes / merged PRs since the last sync of MLX, mlx-lm,
   mlx-vlm and the engines we compare with or learn from. List bug fixes that affect us, new kernels or features,
   and new ideas, with links, as backlog items for the cycle.
4. Land the sync as its own merge with the measurements above, so later regressions can be bisected to it.


## Release notes contract

Keep the next draft under `[Unreleased]` until a maintainer cuts the release.
Use this order (including headings with no changes):

1. One paragraph explaining what users gain and which package contains it.
2. `### Highlights`: 3–6 short bullets, user benefit first; link every measured
   number to a dated benchmark and identify its workload.
3. `### Upgrade notes / breaking changes`: commands or configuration users must
   change, changed defaults, errors, cache invalidation and removed settings.
4. `### Performance`: machine, checkpoint, metric/workload, before → after,
   source link. Separate cold, repeated and follow-up TTFT from decode speed.
5. `### Added`, `### Changed`, `### Fixed`, `### Security`: one or two lines per
   item in user language. Use public setting names where users need them.
6. A compare footer from the previous version to the new version (main for drafts).

This combines curated highlights used by vLLM and mlx-lm, concise user-facing
changes used by Ollama, and upgrade instructions used by SGLang. Their formats
vary; Yunshu's template is a local convention, not a claim that every project
uses identical headings. Avoid implementation nicknames and unshipped claims.

### Notes review checklist

- [ ] Review `git log <previous-tag>..main`; distinguish landed changes from branches.
- [ ] Reconcile numbers with `docs/reports/PERF_TREND.md` and link `docs/BENCHMARKS.md`.
- [ ] State machine, checkpoint, modes, sample/repetition count and correctness limits.
- [ ] Include workload regressions; never add gains across separate experiments.
- [ ] Resolve every TODO before a release; do not bump a draft's package version.
- [ ] Explain migration actions, dependency floors and deprecations explicitly.
- [ ] Run public-doc tests and inspect rendered notes for the intended release:

  ```bash
  uv run pytest tests/unit/test_public_docs.py tests/unit/test_release_notes.py -q
  uv run python scripts/release/notes.py 0.1.2 --output /path/to/review-notes.md
  ```

- [ ] Inspect the draft's install commands and comparison before publishing.
  `scripts/release/notes.py` adds pinned pip/uv commands, Homebrew instructions
  (with tap-delay caveat), and the previous-version comparison automatically.
- [ ] Treat historical note rewrites as editorial proposals: preserve published
  behavior, dates and measurement caveats, and review their diff separately.

## Compatibility and deprecation policy

Pre-1.0 does not make silent breakage acceptable. A user-facing removal or changed
API/default needs an upgrade note naming its replacement and effect. Prefer a
warning and at least one published release of overlap for stable settings and CLI
options; if security or correctness requires immediate removal, explain why.
Experimental settings may be retired after their recorded decision, but still
list user-visible removals. Python internals are not a stable embedding API.
Security fixes target main and the latest release, as described in SECURITY.md.

## Verify the exact release candidate

Run `nice -n 15 scripts/dev/ci-local <candidate-sha>` before tagging to catch
clean-install and sandbox/path-dependent failures. Use committed SHAs for `yv`
arms and inspect the first line of `yv.log`; the base and candidate must differ.
Read every completed job's exit status and final evidence record. Resume a failed
stage after fixing infrastructure instead of rerunning already passed stages.

On the shared GPU queue, priorities are non-positive: designated snapshot/sync
jobs use 0, other work -1 (>=80 GB: -2). `--gate` admits a gate ahead of the same
priority's backlog. Timing cells require quiet admission; correctness cells do not.
A CPU-only documentation candidate can use `yv ab --suite preflight`; it does not
certify engine output or performance. See [Contributing](CONTRIBUTING.md) and
[verification](docs/guides/VERIFY.md) for the development checks.
