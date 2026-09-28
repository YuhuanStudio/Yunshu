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

# 2. Bump the version in pyproject.toml (one place; the CLI, /version and
#    MCP read it from the installed package metadata).
#    version = "0.1.1"

# 3. Turn "## [Unreleased]" in CHANGELOG.md into "## [0.1.1] - YYYY-MM-DD"
#    and start a new empty "## [Unreleased]" above it.

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
