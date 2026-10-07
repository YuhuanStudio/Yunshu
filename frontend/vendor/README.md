# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `c3eb7ca` on YunUI branch `status-pill` (unreleased), on top of `52f8a991dda770df35f27b6521dafee71645aa8c` (analytics primitives); adds `StatusPill` / `StatusPillBar`.
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-status-pill.tgz`
- SHA-256: `9f5570bb9c5a2c1095d2da45448b8e5e7ab4696793f698cc75450400796e6742`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
