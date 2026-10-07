# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `84d5817` on YunUI branch `console-integration` (unreleased): `yundesign-ports` (SegmentMeter, ScrollFade, DisclosureCard, DetailRow/DetailList/Figure, WorkspaceLayout/GroupLabel, BoundedText, HoverRow, StatusPill) merged with `console-feedback` (Progress indeterminate, GenerationStats ttft/cached/prompt, SegmentedBar marks, `yunui icons`, TraceTimeline, MetricHero), on top of `52f8a991dda770df35f27b6521dafee71645aa8c`.
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-console-integration.tgz`
- SHA-256: `afc0f1a1e9d4623e69df88d8bfbd60c23f05453c442afa8f0f9f024a6cfd3bcb`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
