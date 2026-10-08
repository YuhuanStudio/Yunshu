import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { useMinWidth } from "./ui";
import {
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  SegmentMeter,
  Sparkline,
  StatusIndicator,
} from "@yuhuanowo/yunui";
import {
  StatusIslandCard,
  StatusIslandHeader,
  StatusIslandMetric,
  StatusIslandNavGroup,
  StatusIslandNavRow,
} from "@yuhuanowo/yunui/patterns";
import type { HostState } from "./host-hook";
import type { MemoryLedgerData } from "./memory-api";
import { activity, phaseLabels } from "./engineView";
import {
  MEMORY_DANGER,
  MEMORY_WARN,
  pressureText,
  uptimeText,
} from "./footer-status";
import { offlineCause } from "./errors";
import { t } from "./i18n/index.ts";
import { fixed, number } from "./i18n/format.ts";
import { decodeSparkline, modelKind, shortModelName } from "./status-island";
import { routeHref } from "./route";
import type { Engine } from "./ui";

/** A host sample older than this is stale: values stay, dimmed, and the age says so. */
const HOST_STALE_S = 6;
const DASH = "—";
const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

export type IslandDot = "online" | "away" | "offline" | "neutral";

/**
 * The status island's content: header, live, memory, machine (only when the engine reports host
 * telemetry) and navigation. Every card with nothing to say is not rendered.
 */
export function StatusIslandContent({
  engine,
  ledger,
  host,
  dot,
  now,
  onNavigate,
}: {
  engine: Engine;
  ledger: MemoryLedgerData | null;
  host: HostState;
  dot: IslandDot;
  /** Injected so the sample age is testable. */
  now: number;
  onNavigate: () => void;
}) {
  const status = engine.status;
  const online = engine.phase === "online" && status != null;
  const go = (page: Parameters<typeof routeHref>[0]) => () => {
    globalThis.location.hash = routeHref(page).slice(1);
    onNavigate();
  };
  if (!online) {
    const cause = offlineCause(engine.phase, engine.errorStatus);
    const word =
      engine.phase === "connecting"
        ? t("shell.footer.engine.connecting")
        : engine.phase === "unauthorized"
          ? t("shell.footer.engine.unauthorized")
          : t("shell.footer.engine.offline");
    return (
      <>
        <StatusIslandHeader
          indicator={<StatusIndicator status={dot} />}
          title={word}
          subtitle={engine.phase === "offline" ? cause.short : undefined}
        />
        <Nav online={false} active={0} loaded={0} />
      </>
    );
  }

  const loaded = status.models.filter((m) => m.loaded);
  const first = loaded[0];
  const a = activity(status);
  const spark = decodeSparkline(engine.series);
  const mem = status.memory;
  const total = finite(mem.total_gb) ? mem.total_gb : ledger?.total_gb;
  const active = finite(mem.active_gb) ? mem.active_gb : ledger?.mlx.active_gb;
  const usage =
    finite(active) && finite(total) && total > 0 ? active / total : null;
  const reserve = finite(mem.cache_gb) ? mem.cache_gb : ledger?.mlx.cache_gb;
  const level = ledger?.host.pressure_level ?? null;
  const swap = ledger?.host.swap_used_gb;
  const kind = modelKind(first);

  return (
    <>
      <StatusIslandHeader
        indicator={<StatusIndicator status={dot} />}
        title={first ? shortModelName(first.id) : t("shell.island.noModel")}
        tag={
          first && loaded.length > 1
            ? `+${loaded.length - 1}`
            : (kind ?? undefined) // i18n-ignore
        }
        subtitle={t("shell.island.version", {
          version: status.version,
          uptime: uptimeText(status.uptime_s),
        })}
        action={
          <Button
            variant="outline"
            size="sm"
            className="rounded-full"
            onClick={go("overview")}
          >
            {t("shell.island.overview")}
          </Button>
        }
      />

      <StatusIslandCard
        title={t("shell.island.live.title")}
        note={phaseLabels[a.phase]}
        data-testid="island-live"
      >
        {a.decodeNow != null && (
          <div className="flex items-end justify-between gap-4">
            <div
              className="min-w-[7ch] text-[22px] font-semibold leading-none tabular-nums text-foreground"
              data-testid="island-tps"
            >
              {number(a.decodeNow, 0)}
              <span className="text-xs font-normal text-muted-foreground">
                tok/s
              </span>
            </div>
            {spark.length > 0 && (
              <div
                className="min-w-0 flex-1"
                title={t("shell.island.live.sparkNote")}
              >
                <Sparkline
                  data={spark}
                  width={160}
                  height={32}
                  tone="neutral"
                  area
                  label={t("shell.island.live.spark")}
                  className="block h-8 w-full"
                />
              </div>
            )}
          </div>
        )}
        <div
          className={`${a.decodeNow != null ? "mt-2.5 " : ""}text-[13px] tabular-nums text-muted-foreground`}
        >
          {t("shell.island.live.counts", {
            active: a.counts.active,
            queued: a.counts.queued,
          })}
        </div>
        {a.phase === "prefill" && (
          <div className="mt-1.5" data-testid="island-prefill">
            <SegmentMeter
              value={(a.progress ?? 0) / 100}
              label={t("shell.island.live.prefillPlain")}
              valueText={
                a.progress != null ? `${number(a.progress, 0)}%` : undefined
              }
              segments={32}
              warnAt={null}
              dangerAt={null}
              tone="neutral"
              trackColor="color-mix(in srgb, var(--text-primary) 14%, transparent)"
            />
            <div className="mt-1 text-xs tabular-nums text-muted-foreground">
              {a.progress != null
                ? t("shell.island.live.prefill", { pct: number(a.progress, 0) })
                : t("shell.island.live.prefillPlain")}
            </div>
          </div>
        )}
      </StatusIslandCard>

      {usage != null && finite(active) && finite(total) && (
        <StatusIslandCard
          title={t("shell.island.mem.title")}
          note={
            <span>
              {fixed(active, 1)}/{number(total, 0)}
              <span className="text-xs">GB</span>
            </span>
          }
          data-testid="island-memory"
        >
          <SegmentMeter
            value={usage}
            label={t("shell.island.mem.meter")}
            valueText={`${fixed(active, 1)} / ${number(total, 0)} GB`}
            segments={32}
            warnAt={MEMORY_WARN}
            dangerAt={MEMORY_DANGER}
            tone="neutral"
            trackColor="color-mix(in srgb, var(--text-primary) 14%, transparent)"
          />
          <div className="mt-2">
            {level && (
              <StatusIslandMetric
                label={t("shell.island.mem.pressure")}
                value={pressureText(level)}
              />
            )}
            {finite(swap) && (
              <StatusIslandMetric
                label={t("shell.island.mem.swap")}
                value={fixed(swap, 1)}
                unit="GB"
              />
            )}
            {finite(reserve) && (
              <StatusIslandMetric
                label={t("shell.island.mem.reserve")}
                title={t("shell.island.mem.reserveTip")}
                value={fixed(reserve, 1)}
                unit="GB"
              />
            )}
          </div>
        </StatusIslandCard>
      )}

      <MachineCard host={host} now={now} />
      <Nav
        online
        active={a.counts.active}
        loaded={loaded.length}
        onNavigate={onNavigate}
      />
    </>
  );
}

function MachineCard({ host, now }: { host: HostState; now: number }) {
  const wide = useMinWidth(640);
  const [open, setOpen] = useState(false);
  const snap = host.host;
  if (!snap || snap === "unsupported" || !snap.telemetry) return null;
  const tm = snap.telemetry;
  if (tm.state === "unknown") return null;
  const { system } = snap;
  const ageS =
    tm.sampledAt == null
      ? null
      : Math.max(0, (now - tm.sampledAt * 1000) / 1000);
  const stale = ageS != null && ageS > HOST_STALE_S;
  const r = tm.reasons;
  const why = (...k: string[]) =>
    k.map((x) => r[x]).find(Boolean) ?? t("overview.host.unknownTip");
  const unit = (v: number | null, u: string, d = 0) =>
    v == null ? undefined : u;
  const num = (v: number | null, d = 0) =>
    v == null ? DASH : d > 0 ? fixed(v, d) : number(v, 0);
  const busy = tm.gpu.activeRatio == null ? null : tm.gpu.activeRatio * 100;
  const thermal =
    system.thermal.state === "unknown"
      ? null
      : system.thermal.state === "normal"
        ? t("overview.host.thermalNormal")
        : system.thermal.speedLimitPercent != null
          ? t("overview.host.thermalLimited", {
              pct: number(system.thermal.speedLimitPercent, 0),
            })
          : t("overview.host.thermalLimitedPlain");
  const summary = [
    busy != null ? `GPU ${num(busy)}%` : null,
    tm.temperature.dieMaxC != null ? `${num(tm.temperature.dieMaxC)}°C` : null,
    tm.watts.gpu != null ? `${fixed(tm.watts.gpu, 1)}W` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const rows = (
    <>
      <StatusIslandMetric
        label={t("shell.island.host.busy")}
        value={num(busy)}
        unit={unit(busy, "%")}
        title={busy == null ? why("gpu.active_ratio", "gpu") : undefined}
      />
      <StatusIslandMetric
        label={t("shell.island.host.power")}
        value={
          tm.watts.gpu == null
            ? DASH
            : tm.watts.package == null
              ? fixed(tm.watts.gpu, 1)
              : `${fixed(tm.watts.gpu, 1)} / ${fixed(tm.watts.package, 1)}`
        }
        unit={tm.watts.gpu == null ? undefined : "W"}
        title={tm.watts.gpu == null ? why("watts.gpu", "watts") : undefined}
      />
      <StatusIslandMetric
        label={t("overview.host.gpuClock")}
        value={num(tm.gpu.frequencyMhz)}
        unit={unit(tm.gpu.frequencyMhz, "MHz")}
        title={
          tm.gpu.frequencyMhz == null
            ? why("gpu.frequency_mhz", "gpu")
            : undefined
        }
      />
      <StatusIslandMetric
        label={t("overview.host.die")}
        value={num(tm.temperature.dieMaxC)}
        unit={unit(tm.temperature.dieMaxC, "°C")}
        title={
          tm.temperature.dieMaxC == null
            ? why("temperature.die_max_c", "temperature")
            : undefined
        }
      />
      <StatusIslandMetric
        label={t("overview.host.thermal")}
        value={thermal ?? DASH}
        title={
          thermal == null
            ? (system.thermal.reason ?? t("overview.host.unknownTip"))
            : undefined
        }
      />
    </>
  );
  return (
    <StatusIslandCard
      title={t("overview.host.title")}
      note={
        <span title={t("overview.host.sourceTip")}>
          {ageS == null
            ? t("overview.host.noSample")
            : stale
              ? t("overview.host.stale", { n: number(ageS, 0) })
              : t("overview.host.age", { n: number(ageS, 0) })}
        </span>
      }
      className={stale ? "opacity-60" : undefined}
      data-testid="island-machine"
    >
      {wide ? (
        rows
      ) : (
        <Collapsible open={open} onOpenChange={setOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              data-testid="island-machine-toggle"
              className="h-auto w-full justify-between gap-2 border-transparent bg-transparent p-0 text-left text-[13px] font-normal tabular-nums text-foreground hover:bg-transparent"
            >
              <span className="truncate">{summary || DASH}</span>
              <ChevronDown
                aria-hidden
                size={14}
                className={`shrink-0 text-muted-foreground transition-transform duration-[180ms] ease-out motion-reduce:transition-none ${open ? "" : "-rotate-90"}`}
              />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="pt-1.5">{rows}</div>
          </CollapsibleContent>
        </Collapsible>
      )}
    </StatusIslandCard>
  );
}

function Nav({
  online,
  active,
  loaded,
  onNavigate,
}: {
  online: boolean;
  active: number;
  loaded: number;
  onNavigate?: () => void;
}) {
  return (
    <StatusIslandNavGroup>
      <StatusIslandNavRow
        label={t("shell.island.nav.requests")}
        detail={
          online && active > 0
            ? t("shell.island.nav.requestsActive", { n: active })
            : undefined
        }
        href={routeHref("requests")}
        onClick={onNavigate}
      />
      <StatusIslandNavRow
        label={t("shell.island.nav.models")}
        detail={
          online && loaded > 0
            ? t("shell.island.nav.modelsLoaded", { n: loaded })
            : undefined
        }
        href={routeHref("models")}
        onClick={onNavigate}
      />
      <StatusIslandNavRow
        label={t("shell.island.nav.diagnostics")}
        href={routeHref("diagnostics")}
        onClick={onNavigate}
      />
      <StatusIslandNavRow
        label={t("shell.island.nav.logs")}
        href={routeHref("logs")}
        onClick={onNavigate}
      />
    </StatusIslandNavGroup>
  );
}
