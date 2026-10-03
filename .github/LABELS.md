# Triage labels

This is a proposed, small vocabulary for maintainers. It does not create or sync
GitHub labels; repository administrators apply it separately.

| Label | Meaning |
|---|---|
| bug | Reproducible incorrect behavior |
| enhancement | New capability or improved behavior |
| documentation | Guides, reference or examples |
| compatibility | SDK, client or API contract |
| performance | Decode, cold/cached TTFT or prefix reuse |
| hardware | Chip generation, memory or macOS-specific behavior |
| security | Public remediation after coordinated disclosure; never a private report |
| needs-reproduction | Missing minimal reproduction or environment |
| needs-evidence | Missing correctness or benchmark evidence |
| good first issue | Bounded work with an explicit starting point |
| help wanted | Maintainer welcomes implementation or reproduction help |
| breaking-change | Requires an upgrade note and migration review |

Use a type plus at most a few useful topic/status labels. Do not use labels as a
substitute for a clear issue description. Close resolved needs-* statuses;
priority belongs in the maintainer's explanation, not a promise of delivery.
