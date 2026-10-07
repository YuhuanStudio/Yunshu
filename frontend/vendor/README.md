# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `62990d3` on YunUI branch `global-ab` (unreleased): the `console-integration` content (YunDesign ports, console-feedback) plus the default look from YunDesign (window tier, flat cards, solid switch, press step, thin Progress, Sidebar on the window tier).
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-global-ab.tgz`
- SHA-256: `45e61e413c23332369b65ab68655a68587cd0970173ccbdc469369fc4be87f2f`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
