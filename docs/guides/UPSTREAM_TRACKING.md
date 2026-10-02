# Tracking upstream code

Yunshu builds on other projects. `vendor.json` records every place it does, so upstream changes can be followed.

| kind | meaning | sync |
|---|---|---|
| `vendored` | file copied from upstream (`local_changes` lists deliberate edits) | re-copy, re-apply `local_changes` |
| `derived` | rewritten from specific upstream files (`upstream_paths`) | read the upstream diff, port what matters |
| `inspired` | same algorithm, independent code | read the upstream diff for ideas or bug fixes |
| `patches` | we monkeypatch an upstream symbol (`module`, `symbol`, `signature`, `source_sha256`) | re-check our replacement against the new source |

Every entry has `repo`, `clone` (a checkout under `reference/`), `commit` (the upstream commit or release tag we based
it on; for older entries the oldest commit in a shallow clone), `license`, and the local `path`.

## Run

    just vendor-check                                           # fetch the clones, report
    uv run python scripts/vendor/check_upstream.py --no-fetch   # offline

Sections: patched symbols, vendored files, derived, inspired, watched paths, packages, self-checks. Exit code 2 when a
patched symbol changed or is missing or a self-check fails, 1 when anything else is behind, 0 when clean.

## What is watched

- `watch`: every reference clone we follow for ideas or fixes, with the globs that matter and `commit` = the upstream
  commit we have reviewed up to (move it forward after reading the report). The report lists commits touching the
  globs since then and files added upstream that match them. Repos pinned to an installed package (`pin_package`) are
  compared against that release tag.
- `watch_excluded`: every other clone under `reference/`, with the reason it is not followed (parked modality,
  non-goal, superseded). A clone in neither list fails the self-check, so a new clone cannot go unclassified.
- Packages: read from `pyproject.toml` (base, every extra, every dependency group); `packages` in vendor.json only
  adds names that are not declared there (mlx-metal). The report shows installed, latest on PyPI and the declared
  specifier, so a new dependency cannot be missed.
- Self-check: a file under `python/` whose first lines say `# Upstream`, "ported/adapted/studied from" or
  "Inspired by" must be in vendor.json (a package `__init__.py` next to registered files is exempt).

A patched symbol is hashed (normalized AST source) in the installed package. `!! ... UPSTREAM SOURCE CHANGED` means the
code our replacement wraps or copies changed: read the new source, fix `python/yunshu_engine/...`, run the tests and
the parity checks, then record the new hash:

    uv run python scripts/vendor/check_upstream.py --update-hashes

## Sync a derived / inspired entry

1. `just vendor-check` lists upstream commits on its paths since `commit`.
2. Read them (`git -C reference/<clone> show <sha>`) and port what applies.
3. Set `commit` to the upstream commit (or tag) you reviewed up to.

## Add a new one

Add the entry to the right list in `vendor.json` (patches: run `--update-hashes`), start the source file with an
`# Upstream (kind): repo (license) paths @ commit` comment, and add the license to `THIRD_PARTY_NOTICES.md`. If the
clone is missing, clone the repo into `reference/<name>` (`reference/` is gitignored).
