# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `64cd596` on YunUI branch `arc` (unreleased), on top of `charts` `7d2f027`: agent interaction (`ToolCall`, `ToolCallGroup`, `useStreamReveal`, soft-follow `ChatMessageList` with jump pill, `ThinkingBlock` `doneLabel`, `AgentRunStatus` activity roll and `elapsed`, send/stop morph with optional progress ring, spring easing tokens).
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-arc.tgz`
- SHA-256: `fa60f336ad535ecc3ab05f32a9a6b505b1096e1e1547b1ba758c0d97084c2970`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
