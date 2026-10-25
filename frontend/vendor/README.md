# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `a8ad8da` on YunUI branch `arc` (unreleased), on top of `charts` `7d2f027`: agent interaction (`ToolCall`, `ToolCallGroup`, `useStreamReveal`, soft-follow `ChatMessageList` with jump pill, `ThinkingBlock` `doneLabel`, `AgentRunStatus` activity roll and `elapsed`, send/stop morph with optional progress ring, spring easing tokens).
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-arc.tgz`
- SHA-256: `db4951774501c4466b5f8e7386b5bc6bcc01b0bc3092d6e58498feb51fcdc361`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
