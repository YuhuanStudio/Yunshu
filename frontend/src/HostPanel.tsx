import { useState } from "react";
import {
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@yuhuanowo/yunui";
import {
  DetailList,
  DetailRow,
  SectionRow,
  StatCard,
} from "@yuhuanowo/yunui/patterns";
import { Activity, ChevronDown, Cpu, Thermometer, Zap } from "lucide-react";
import { ChartCard, SeriesChart } from "./AnalyticsPanels";
import type { HostTelemetry, HostSystem } from "./host-api";
import type { HostState } from "./host-hook";
import { has, t, tr, useLocale } from "./i18n/index.ts";
import { clock, fixed, number } from "./ui";

/** A sample older than this is stale: the values stay (dimmed) and the age says so. */
export const HOST_STALE_S = 6;

const dash = "—";
const unit = (u: string) => (
  <span className="ml-1 text-xs font-normal text-muted-foreground">{u}</span>
);
const reasonOf = (r: Record<string, string>, ...keys: string[]) =>
  keys.map((k) => r[k]).find(Boolean) ?? undefined;

/** Value + unit, or an em dash that carries the engine's reason as its tooltip. */
function Reading({
  v,
  u,
  digits = 0,
  reason,
}: {
  v: number | null;
  u: string;
  digits?: number;
  reason?: string;
}) {
  if (v == null)
    return <span title={reason ?? t("overview.host.unknownTip")}>{dash}</span>;
  return (
    <>
      {digits > 0 ? fixed(v, digits) : number(v, 0)}
      {unit(u)}
    </>
  );
}

const pressureWord = (s: HostSystem["pressure"]["state"]) =>
  s === "unknown"
    ? dash
    : tr(`shell.footer.pressure.${s === "warning" ? "warn" : s}`);

/**
 * Host power, GPU clock and activity, die temperature, OS thermal limit and memory pressure from
 * the engine's own 1 Hz sampler. Every number carries its source and age; nothing the engine did
 * not report is drawn as 0. An engine without the telemetry section shows nothing at all.
 */
export function HostPanel({
  state,
  now,
}: {
  state: HostState;
  /** Injected so the age label is testable. */
  now: number;
}) {
  useLocale();
  const [open, setOpen] = useState(false);
  if (!state.host || state.host === "unsupported") return null;
  const { telemetry, system } = state.host;
  if (!telemetry) return null;
  const tm: HostTelemetry = telemetry;
  const ageS =
    tm.sampledAt == null
      ? null
      : Math.max(0, (now - tm.sampledAt * 1000) / 1000);
  const stale = ageS != null && ageS > HOST_STALE_S;
  const unknown = tm.state === "unknown";
  const r = tm.reasons;
  const rows = state.history.map((p) => ({
    x: p.at * 1000,
    values: { gpu: p.gpuW, package: p.packageW } as Record<
      string,
      number | null
    >,
  }));
  return (
    <section
      className="space-y-3"
      data-testid="host-panel"
      data-state={tm.state}
      data-stale={stale ? "true" : undefined}
    >
      <SectionRow
        title={t("overview.host.title")}
        action={
          <span
            className="text-xs tabular-nums text-muted-foreground"
            title={t("overview.host.sourceTip")}
          >
            {ageS == null
              ? t("overview.host.noSample")
              : stale
                ? t("overview.host.stale", { n: number(ageS, 0) })
                : t("overview.host.age", { n: number(ageS, 0) })}
          </span>
        }
      />
      {unknown ? (
        <p
          className="rounded-lg bg-(--bg-elevated) px-4 py-3 text-sm text-muted-foreground"
          data-testid="host-unavailable"
        >
          {t("overview.host.unavailable")}
          {tm.reason && (
            <span className="ml-2 text-xs">
              {t("overview.host.reasonPrefix")}
              <span className="font-mono">{tm.reason}</span>
            </span>
          )}
        </p>
      ) : (
        <>
          <div
            className={`grid grid-cols-2 gap-2 sm:gap-3 xl:grid-cols-4 ${stale ? "opacity-60" : ""}`}
            data-stat-grid=""
            data-testid="host-stats"
          >
            <StatCard
              compact
              valueFirst
              icon={Zap}
              label={t("overview.host.gpuPower")}
              value={
                <Reading
                  v={tm.watts.gpu}
                  u="W"
                  digits={1}
                  reason={reasonOf(r, "watts.gpu", "watts")}
                />
              }
              subtext={
                tm.watts.package != null
                  ? t("overview.host.packageSub", {
                      w: number(tm.watts.package, 1),
                    })
                  : t("overview.host.packageUnknown")
              }
            />
            <StatCard
              compact
              valueFirst
              icon={Activity}
              label={t("overview.host.gpuClock")}
              value={
                <Reading
                  v={tm.gpu.frequencyMhz}
                  u="MHz"
                  reason={reasonOf(r, "gpu.frequency_mhz", "gpu")}
                />
              }
              subtext={
                tm.gpu.activeRatio != null
                  ? t("overview.host.activeSub", {
                      pct: number(tm.gpu.activeRatio * 100, 0),
                    })
                  : t("overview.host.activeUnknown")
              }
            />
            <StatCard
              compact
              valueFirst
              icon={Thermometer}
              label={t("overview.host.die")}
              value={
                <Reading
                  v={tm.temperature.dieMaxC}
                  u="°C"
                  reason={reasonOf(r, "temperature.die_max_c", "temperature")}
                />
              }
              subtext={
                tm.temperature.dieMeanC != null
                  ? t("overview.host.dieMean", {
                      c: number(tm.temperature.dieMeanC, 0),
                    })
                  : t("overview.host.dieMax")
              }
            />
            <StatCard
              compact
              valueFirst
              icon={Cpu}
              label={t("overview.host.thermal")}
              value={
                system.thermal.state === "unknown"
                  ? dash
                  : system.thermal.state === "normal"
                    ? t("overview.host.thermalNormal")
                    : system.thermal.speedLimitPercent != null
                      ? t("overview.host.thermalLimited", {
                          pct: number(system.thermal.speedLimitPercent, 0),
                        })
                      : t("overview.host.thermalLimitedPlain")
              }
              subtext={t("overview.host.pressureSub", {
                level: pressureWord(system.pressure.state),
              })}
            />
          </div>
          {rows.length >= 1 && (
            <ChartCard
              title={t("overview.host.chartTitle")}
              description={t("overview.host.chartDesc")}
              data-testid="host-chart"
            >
              <SeriesChart
                busy={false}
                data={rows}
                series={[
                  {
                    key: "gpu",
                    label: t("overview.host.seriesGpu"),
                    tone: "accent" as const,
                  },
                  {
                    key: "package",
                    label: t("overview.host.seriesPackage"),
                    tone: "neutral" as const,
                    dashed: true,
                  },
                ]}
                height={120}
                ariaLabel={t("overview.host.chartAria")}
                formatX={clock}
                formatY={(v) => `${number(v, 0)} W`}
                maxGap={6000}
              />
            </ChartCard>
          )}
        </>
      )}
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="-ml-2">
            {t("overview.host.details")}
            <ChevronDown
              size={14}
              className={`transition-transform duration-150 ${open ? "rotate-180" : ""}`}
            />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div data-testid="host-details">
            <DetailList className="mt-2">
              <DetailRow
                label={t("overview.host.cpuPower")}
                value={
                  <Reading
                    v={tm.watts.cpu}
                    u="W"
                    digits={1}
                    reason={reasonOf(r, "watts.cpu")}
                  />
                }
              />
              <DetailRow
                label={t("overview.host.anePower")}
                value={
                  <Reading
                    v={tm.watts.ane}
                    u="W"
                    digits={1}
                    reason={reasonOf(r, "watts.ane")}
                  />
                }
              />
              <DetailRow
                label={t("overview.host.dramPower")}
                value={
                  <Reading
                    v={tm.watts.dram}
                    u="W"
                    digits={1}
                    reason={reasonOf(r, "watts.dram")}
                  />
                }
              />
              <DetailRow
                label={t("overview.host.powerSource")}
                value={
                  system.power.source === "unknown"
                    ? dash
                    : system.power.source === "ac"
                      ? t("overview.host.sourceAc")
                      : system.power.batteryPercent != null
                        ? t("overview.host.sourceBatteryPct", {
                            pct: number(system.power.batteryPercent, 0),
                          })
                        : t("overview.host.sourceBattery")
                }
              />
              <DetailRow
                label={t("overview.host.interval")}
                value={
                  tm.intervalS != null ? `${number(tm.intervalS, 1)} s` : dash
                }
              />
              <DetailRow
                label={t("overview.host.sampledAt")}
                value={tm.sampledAt != null ? clock(tm.sampledAt * 1000) : dash}
              />
            </DetailList>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {t("overview.host.honesty")}
          </p>
          {Object.keys(r).length > 0 && (
            <ul
              className="mt-2 space-y-0.5 text-xs text-muted-foreground"
              data-testid="host-reasons"
            >
              {Object.entries(r).map(([k, v]) => (
                <li key={k}>
                  <span className="font-mono">{k}</span>
                  {" · "}
                  {has(`overview.host.reason.${v}`)
                    ? tr(`overview.host.reason.${v}`)
                    : v}
                </li>
              ))}
            </ul>
          )}
        </CollapsibleContent>
      </Collapsible>
    </section>
  );
}
