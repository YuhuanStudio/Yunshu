# Historical release notes proposal

`historical-release-notes.patch` proposes editorial rewrites of 0.1.1 and 0.1.2
without changing their entries in CHANGELOG.md. This patch is deliberately in a
separate commit so the proposal can be dropped independently. It does not update
published GitHub releases.

Review with `git apply --check docs/proposals/historical-release-notes.patch`.
The patch reorganizes published changes into highlights, migration notes,
performance and short categorized entries. Historical numbers retain the original
measurement limitations; published notes did not identify every checkpoint or
repeat count, so the proposal does not invent those details. Review the published
release bodies and dated benchmark sources before adopting or republishing it.
