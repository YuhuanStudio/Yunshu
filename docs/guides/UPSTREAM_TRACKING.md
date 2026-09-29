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

Sections: patched symbols, vendored files, derived, inspired, watched paths, packages. Exit code 2 when a patched
symbol changed or is missing, 1 when anything else is behind, 0 when clean.

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
