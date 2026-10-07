# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `16396a9` (opt-in compact app density `data-yunui-density="compact"`, calm `AnimatedNumber`, tabular stat values, `StatusPill` `valueMinCh` on top of `e70ce34`) on YunUI branch `global-ab` (unreleased): the `console-integration` content (YunDesign ports, console-feedback) plus the default look from YunDesign (window tier, flat cards, solid switch, press step, thin Progress, Sidebar on the window tier), plus integer-only Y ticks in TimeSeriesChart, `DonutChart monochrome` ink ramp Sidebar `footerItems`, x-axis tick thinning by pixel spacing in TimeSeriesChart, and the chart curve work (zero-based value domain, monotone cubic paths, unique Y ticks, `minSamples` collecting state, subtler Sparkline area).
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-global-ab.tgz`
- SHA-256: `a030840157ecd5c14903565aa4470a796358128f4a24c4b9945b791097d3f4c0`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
