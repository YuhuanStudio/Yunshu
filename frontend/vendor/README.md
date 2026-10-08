# YunUI dependency snapshot

This is an installed package artifact, not copied component source. The console
imports public `@yuhuanowo/yunui` exports and composes their props/slots.

- Upstream repository: https://github.com/YuhuanStudio/YunUI
- Source checkout commit: `8ddb29b` (BarChart fits narrow containers, TimeSeriesChart `showReadout`; on top of `d3de49a` (`SettingRow divider`; `SegmentedSelect variant="tray"`, code-only entry `@yuhuanowo/yunui/code`; on top of `c3c3190` Alert neutral, `d0a8a6c` StatusPill flat, `ca489ca` (`SettingRow` stacks its control under the title below `sm`, `stack={false}` keeps a small control inline; on top of `856c201`: toast action slot, `CodeBlock` `defaultExpanded`, on top of `20535d1`: Sidebar renders one `nav` landmark, chart collecting caption and heatmap readout are not live regions, on top of opt-in compact app density `data-yunui-density="compact"`, calm `AnimatedNumber`, tabular stat values, `StatusPill` `valueMinCh` on top of `e70ce34`) on YunUI branch `global-ab` (unreleased): the `console-integration` content (YunDesign ports, console-feedback) plus the default look from YunDesign (window tier, flat cards, solid switch, press step, thin Progress, Sidebar on the window tier), plus integer-only Y ticks in TimeSeriesChart, `DonutChart monochrome` ink ramp Sidebar `footerItems`, x-axis tick thinning by pixel spacing in TimeSeriesChart, and the chart curve work (zero-based value domain, monotone cubic paths, unique Y ticks, `minSamples` collecting state, subtler Sparkline area).
- Package version: `0.2.18` (publication remains on hold).
- Artifact: `yuhuanowo-yunui-0.2.18-global-ab.tgz`
- SHA-256: `558eb0f9fd9a6ddd6987135a3610fbfeede7f4d8aac069f06b8b86635fc5d351`
- License: Apache-2.0; package metadata and upstream notices are included in the archive.

The pinned archive keeps the frontend reproducible without requiring a sibling
YunUI checkout or publishing an unreleased package. To update intentionally, build
YunUI, run `pnpm pack --pack-destination <Yunshu>/frontend/vendor`, update this
provenance, then regenerate the lockfile and run the console checks. Do not copy
individual controls into the application.
