import { useRecentRequests } from "./recentRequests";
import { prefixHit, prefixHitWindow } from "./request-kpi";
import { gbTotalText } from "./byte-format";
import { LiveNumber } from "./LiveNumber";
import { SkeletonSwap } from "./motion/SkeletonSwap.tsx";
import { rangeSeconds, sinceStartId, useRangeHistory } from "./useRangeHistory";
import { MetricChartGroup } from "@yuhuanowo/yunui";
import type {
  MetricBand,
  MetricMarker,
  MetricSeries,
} from "@yuhuanowo/yunui/charts";
import { PrefillBar } from "./PrefillBar";
import { SegmentedTray } from "./SegmentedTray";
import { useFlip } from "./motion/flip";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  AnimatedNumber,
  Button,
  Card,
  EmptyState,
  ScrollFade,
  SegmentedBar,
  Skeleton,
  Sparkline,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import {
  ArrowRight,
  Check,
  Copy,
  Download,
  Gauge,
  HardDrive,
  Link2,
  Pause,
  Play,
  RefreshCw,
  Server,
  Stethoscope,
  Timer,
  Zap,
} from "lucide-react";
import {
  CodeBlock,
  DashboardPage,
  DetailList,
  DetailRow,
  HoverRow,
  PageHeader,
  SectionRow,
  StatCard,
  StatGrid,
} from "@yuhuanowo/yunui/patterns";
import { ChartCard, SeriesChart } from "./AnalyticsPanels";
import { has, t, tr, useLocale } from "./i18n/index.ts";
import { observationCsv } from "./analytics";
import { chartRows, windowPoints } from "./series";
import { totalsFrom } from "./engineView";
import {
  FoldSection,
  HealthLine,
  SpeedPair,
  StateStrip,
  TotalsLine,
} from "./OverviewParts";
import { useSignals } from "./signals";
import type { Connection, RequestRow } from "./api";
import { buildIntegrations, serviceRoot } from "./integrations";
import {
  clock,
  elapsed,
  fixed,
  modelLabel,
  number,
  phaseDot,
  Slot,
  sizeGb,
  StaleStamp,
  supportsChat,
  type Engine,
  StatValue,
} from "./ui";
import { formatMs } from "./RequestTimeline";
import { phaseDurations } from "./request-insight";
import { HostPanel } from "./HostPanel";
import { useHostTelemetry } from "./host-hook";
// Color follows the entity: decode is slot 1 and prefill slot 2 on every chart of the page; context
// series (the sum, the reserve pool) are the de-emphasis gray.
const memorySeries = (): MetricSeries[] => [
  { key: "active", label: t("overview.series.active") },
  { key: "cache", label: t("overview.series.pool"), muted: true },
];
const requestSeries = (): MetricSeries[] => [
  {
    key: "decodeRequests",
    label: t("overview.series.decode"),
    color: "var(--chart-1)",
    raw: true,
  },
  {
    key: "prefillRequests",
    label: t("overview.series.prefill"),
    color: "var(--chart-2)",
    raw: true,
  },
  {
    key: "queued",
    label: t("overview.series.queued"),
    color: "var(--chart-3)",
    raw: true,
  },
  {
    key: "requests",
    label: t("overview.series.all"),
    muted: true,
    raw: true,
  },
];
const rateSeries = () => {
  const decode: MetricSeries = {
    key: "decode",
    label: t("overview.series.decodeLive"),
    color: "var(--chart-1)",
    band: { lo: "decodeLo", hi: "decodeHi" },
  };
  const prefill: MetricSeries = {
    key: "prefill",
    label: t("overview.series.prefillLive"),
    color: "var(--chart-2)",
    band: { lo: "prefillLo", hi: "prefillHi" },
  };
  return { decode: [decode], prefill: [prefill], both: [decode, prefill] };
};
const formatNumber = (value: number) => number(value);
const formatCount = (value: number) => number(value, 0);
const LANES_SHOWN = 5;
/** Five lane rows of 4 rem each (+1 px divider). The room is always there. */
const LANE_AREA_PX = LANES_SHOWN * 65;

/**
 * The lane list in start order with room for `n`: a request joins at the end, an ended one fades in
 * place for `ms` and is then removed, and the rows below glide up (FLIP, see useFlip). The area is
 * always `n` rows tall, so nothing around it moves, and a lone request is always the first row.
 */
type LaneSlot = { row: RequestRow; until: number };
function useLaneSlots(
  rows: readonly RequestRow[],
  n: number,
  ms = 260,
): ({ row: RequestRow; leaving: boolean } | null)[] {
  const [, bump] = useState(0);
  const list = useRef<LaneSlot[]>([]);
  const now = Date.now();
  const byId = new Map(rows.map((r) => [r.request_id, r]));
  list.current = list.current.flatMap((cur) => {
    const fresh = byId.get(cur.row.request_id);
    if (fresh) return [{ row: fresh, until: 0 }];
    if (cur.until === 0) {
      setTimeout(() => bump((x) => x + 1), ms + 20);
      return [{ row: cur.row, until: now + ms }];
    }
    return cur.until > now ? [cur] : [];
  });
  const placed = new Set(list.current.map((x) => x.row.request_id));
  const waiting = rows
    .filter((r) => !placed.has(r.request_id))
    .sort((a, b) => b.elapsed_s - a.elapsed_s);
  for (const r of waiting) list.current.push({ row: r, until: 0 });
  const shown = list.current
    .slice(0, n)
    .map((x) => ({ row: x.row, leaving: x.until > 0 }));
  return [...shown, ...Array.from({ length: n - shown.length }, () => null)];
}

function RequestLane({ row, leaving }: { row: RequestRow; leaving?: boolean }) {
  useLocale();
  const phase = String(row.phase);
  const prompt = row.prompt_tokens ?? 0,
    cached = row.cached_tokens ?? 0;
  const spent = phase === "decode" ? phaseDurations(row as never) : null;
  const progress =
    phase === "prefill" || phase === "starting"
      ? (row.percent ??
        (prompt > 0 && row.processed_tokens != null
          ? (row.processed_tokens / prompt) * 100
          : null))
      : null;
  return (
    <li
      className="live-slot h-16"
      data-flip={row.request_id}
      data-leaving={leaving ? "" : undefined}
      aria-hidden={leaving ? true : undefined}
    >
      <div className="live-row-inner">
        <div className="live-appear grid h-16 grid-cols-[minmax(0,1fr)_auto] grid-rows-[1.25rem_1rem] content-center items-center gap-x-4 gap-y-1">
          <div className="flex min-w-0 items-center gap-2.5">
            <StatusIndicator className="shrink-0" status={phaseDot(phase)} />
            <Slot ch={7} className="shrink-0 text-sm font-medium">
              {/* i18n-keys: overview.phase. */}
              {has(`overview.phase.${phase}`)
                ? tr(`overview.phase.${phase}`)
                : phase}
            </Slot>
            <span
              title={row.request_id}
              className="min-w-0 max-w-[9rem] shrink truncate font-mono text-xs text-muted-foreground"
            >
              {row.request_id}
            </span>
            {row.model && (
              <span
                title={row.model}
                className="hidden min-w-0 flex-1 truncate text-xs text-muted-foreground sm:inline"
              >
                · {modelLabel(row.model)}
              </span>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-4 text-xs tabular-nums text-muted-foreground">
            <Slot ch={9} align="right">
              {phase === "decode" ? (
                <LiveNumber
                  value={row.completion_tokens ?? null}
                  jumpKey={row.request_id}
                />
              ) : (
                number(prompt, 0)
              )}{" "}
              tok
            </Slot>
            <Slot ch={12} align="right" className="text-foreground">
              <LiveNumber
                value={row.tokens_per_second ?? null}
                glide
                format={(v) => fixed(v)}
                jumpKey={row.request_id + phase}
              />
              {row.tokens_per_second == null ? "" : " tok/s"}
            </Slot>
            <Slot ch={7} align="right">
              {elapsed(row.elapsed_s)}
            </Slot>
          </div>
          {progress != null ? (
            <div className="col-span-2 flex h-4 items-center gap-3">
              <PrefillBar row={row} className="h-1 min-w-0 flex-1" />
              {cached > 0 && (
                <span className="shrink-0 whitespace-nowrap text-xs text-muted-foreground">
                  {t("overview.prefill.cached", { n: number(cached, 0) })}
                </span>
              )}
            </div>
          ) : phase === "queued" && row.queue_position != null ? (
            <p className="col-span-2 truncate text-xs text-muted-foreground">
              {t("overview.lane.waiting", {
                ahead: number(row.queue_position, 0),
              })}
            </p>
          ) : spent ? (
            <p className="col-span-2 truncate text-xs tabular-nums text-muted-foreground">
              {t("overview.lane.phases", {
                queue: formatMs(spent.queueMs),
                prefill: formatMs(spent.prefillMs),
                cached:
                  spent.cachedPct == null
                    ? "—"
                    : `${number(spent.cachedPct, 0)}%`,
              })}
            </p>
          ) : phase !== "prefill" && cached > 0 && prompt > 0 ? (
            <p className="col-span-2 truncate text-xs text-muted-foreground">
              {t("overview.lane.prefixHit", {
                cached: number(cached, 0),
                prompt: number(prompt, 0),
              })}
            </p>
          ) : (
            <div className="col-span-2 h-4" />
          )}
        </div>
      </div>
    </li>
  );
}

function savedBaseUrl() {
  try {
    return localStorage.getItem("yunshu.console.url") || location.origin;
  } catch {
    return location.origin;
  }
}

function QuickAction({
  icon,
  title,
  caption,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  caption: string;
  onClick: () => void;
}) {
  return (
    // One outline: the row fills the card (no inner rounded fill inside the card's own border), the
    // focus ring is the card's, and a press nudges the whole card.
    <Card className="min-w-0 overflow-hidden p-0 transition-[box-shadow,transform] duration-(--dur-fast) ease-out has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-(--border-strong) active:scale-[0.99] motion-reduce:transition-none motion-reduce:active:scale-100">
      <HoverRow
        onClick={onClick}
        className="flex w-full items-center gap-3 rounded-none px-4 py-4 text-left focus-visible:ring-0"
      >
        <span className="text-muted-foreground">{icon}</span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{title}</span>
          <span className="block truncate text-xs text-muted-foreground">
            {caption}
          </span>
        </span>
        <ArrowRight size={14} className="shrink-0 text-muted-foreground/60" />
      </HoverRow>
    </Card>
  );
}

export function Dashboard({
  engine,
  connection,
  navigate,
}: {
  engine: Engine;
  connection: Connection;
  navigate: (page: string) => void;
}) {
  useLocale();
  const [range, setRange] = useState("15m"),
    [metric, setMetric] = useState("decode"),
    [heroMetric, setHeroMetric] = useState("decode"),
    [heroWindow, setHeroWindow] = useState("60"),
    [table, setTable] = useState(false),
    [copied, setCopied] = useState<string | null>(null);
  const status = engine.status,
    online = engine.phase === "online";
  const { verdict } = useSignals();
  // Offline with data on screen: the numbers stop moving, so they are dimmed and stamped.
  const stale = engine.phase === "offline" && status != null;
  const dim = stale ? "opacity-60" : "";
  // "Since start" is a window as long as the engine has been up; everything else is a preset.
  const rangeId = range === "start" ? sinceStartId(status?.uptime_s) : range;
  const end = engine.updatedAt ?? Date.now(),
    start = end - rangeSeconds(rangeId) * 1000;
  // Charts read the slim series (engine history first, then live polls). The
  // window is a binary-search slice and the chart gets at most 300 rows.
  // Up to an hour the live series (backfilled from the console's history on open) is the source;
  // 6 h to 30 d come straight from the recorded history, gaps included.
  const longRange = useRangeHistory(
    connection,
    rangeId,
    engine.status?.console_process === true,
  );
  const livePoints = useMemo(
    () => windowPoints(engine.series, start, end),
    [engine.series, start, end],
  );
  const points = longRange.active ? longRange.points : livePoints;
  const rangeMs = rangeSeconds(rangeId) * 1000;
  // A line breaks where samples are further apart than a few of this range's own steps.
  const rangeGapMs = Math.max(12_000, (longRange.resolutionS ?? 0) * 3_000);
  const data = useMemo(() => chartRows(points), [points]);
  const heroSpanMs = Number(heroWindow) * 1000;
  const heroNarrow = useMemo(
    () => windowPoints(engine.series, end - Math.max(300_000, heroSpanMs), end),
    [engine.series, end, heroSpanMs],
  );
  // Right after a restart the window can hold no sample at all; the console's record is still there,
  // so show the last 15 minutes (and say so) instead of an empty chart.
  const heroWiden =
    heroNarrow.filter((p) => !p.gap).length < 2 &&
    windowPoints(engine.series, end - 900_000, end).filter((p) => !p.gap)
      .length >= 2;
  const heroPoints = useMemo(
    () =>
      heroWiden ? windowPoints(engine.series, end - 900_000, end) : heroNarrow,
    [heroWiden, heroNarrow, engine.series, end],
  );
  const heroMark = engine.restarts
    .filter((m) => m.at >= end - (heroWiden ? 900_000 : heroSpanMs))
    .at(-1);
  const rates = rateSeries();
  const throughputSeries = rates[metric as keyof typeof rates];
  const busy = (engine.status?.requests.active ?? 0) > 0;
  const last = status?.last;
  // Engine boundaries inside a window become hairlines on the charts: a restart is drawn, never bridged.
  const marksIn = (from: number, to: number): MetricMarker[] =>
    engine.restarts
      .filter((m) => m.at >= from && m.at <= to)
      .map((m) => ({
        x: m.at,
        kind:
          m.kind === "restarted"
            ? "restart"
            : m.kind === "away"
              ? "away"
              : "switch",
        label: t(
          m.kind === "restarted"
            ? "overview.chart.markRestarted"
            : m.kind === "away"
              ? "overview.chart.markAway"
              : "overview.chart.markSwitched",
        ),
      }));
  const perfMarks = marksIn(start, end);
  const memTotal = status?.memory.total_gb ?? null;
  // Thresholds are quiet background bands, shown only where the axis reaches them.
  const memBands: MetricBand[] =
    memTotal != null && memTotal > 0
      ? [
          {
            from: memTotal * 0.8,
            to: memTotal * 0.92,
            tone: "warning",
            label: t("overview.memory.warn"),
          },
          {
            from: memTotal * 0.92,
            to: memTotal,
            tone: "error",
            label: t("overview.memory.critical"),
          },
        ]
      : [];
  const items = status?.requests.items ?? [];
  const lanes = useLaneSlots(items, LANES_SHOWN);
  const laneRef = useRef<HTMLUListElement>(null);
  useFlip(laneRef);
  const hostState = useHostTelemetry(connection, online);
  const hostShown =
    !!hostState.host &&
    hostState.host !== "unsupported" &&
    !!hostState.host.telemetry;
  const memory = status?.memory;
  const loaded = (status?.models ?? []).filter((m) => m.loaded);
  // Trends compare the later half of this window's samples with the earlier half.
  const observed = useMemo(
    () => engine.finished.filter((r) => r.firstObservedAt >= start),
    [engine.finished, start],
  );
  const totals = useMemo(() => totalsFrom(engine.finished), [engine.finished]);
  // Prefix reuse: the same definition and window as the Requests page (request-kpi.ts), over the
  // finished requests the console holds, so the two pages can never show different numbers.
  const recentRequests = useRecentRequests(
    connection,
    status?.last?.request_id,
    status?.console_process === true ||
      (status == null &&
        engine.failures > 0 &&
        engine.phase !== "unauthorized"),
  );
  const hit = useMemo(
    () =>
      prefixHit(
        [...recentRequests.rows].sort((a, b) => (a.t ?? 0) - (b.t ?? 0)),
      ),
    [recentRequests.rows],
  );
  const heroData = useMemo(() => chartRows(heroPoints), [heroPoints]);
  // Idle sparkline only when the last five minutes actually carried decode traffic.
  const recent = heroData.some((p) => {
    const v = p.values.decode;
    return typeof v === "number" && Number.isFinite(v) && v !== 0;
  });
  const baseUrl = serviceRoot(savedBaseUrl());
  const quickModel = (loaded.find(supportsChat) ?? loaded[0])?.id ?? "";
  const curl =
    buildIntegrations(baseUrl, quickModel).find((i) => i.id === "curl")?.code ??
    "";
  const copy = (id: string, text: string) => {
    void navigator.clipboard?.writeText(text).then(
      () => {
        setCopied(id);
        setTimeout(() => setCopied((c) => (c === id ? null : c)), 2000);
      },
      () => undefined,
    );
  };
  const chooseRange = (next: string) => setRange(next);
  function exportData() {
    const rows = engine.history.filter((s) => s.at >= start && s.at <= end);
    const url = URL.createObjectURL(
        new Blob([observationCsv(rows)], { type: "text/csv;charset=utf-8" }),
      ),
      a = document.createElement("a");
    a.href = url;
    a.download = `yunshu-observations-${range}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  // First-run onboarding only when nothing was ever configured or the token is
  // refused. A restart (502, refused, timeout) keeps the skeleton and the banner.
  // A refused token on a console that was configured keeps the page: the shell
  // shows the banner with a token prompt in place (App.tsx).
  const neverConfigured = (() => {
    try {
      return localStorage.getItem("yunshu.console.url") === null;
    } catch {
      return true;
    }
  })();
  // A console following engines (--engine auto) is always configured: its own address is the connection.
  if (
    !status &&
    neverConfigured &&
    engine.outage.kind === "none" &&
    (engine.phase === "unauthorized" || engine.phase === "offline")
  )
    return (
      <DashboardPage width="7xl" data-testid="overview">
        <PageHeader
          title={t("overview.page.title")}
          description={t("overview.page.descConnect")}
        />
        <Card className="p-4">
          <EmptyState
            size="inline"
            icon={<Server size={22} strokeWidth={1.5} />}
            title={t("overview.connect.title")}
            description={t("overview.connect.desc")}
            action={
              <Button onClick={() => navigate("settings")}>
                {t("overview.connect.action")}
                <ArrowRight size={14} />
              </Button>
            }
          />
          <ol className="mt-4 grid gap-4 sm:grid-cols-3">
            {(["1", "2", "3"] as const).map((n) => (
              <li key={n}>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {`0${n}`}
                </span>
                <h2 className="yunui-section-title mt-2 text-base font-semibold">
                  {tr(`overview.connect.step${n}.title`)}
                </h2>
                <p className="mt-1 text-caption">
                  {tr(`overview.connect.step${n}.desc`)}
                </p>
              </li>
            ))}
          </ol>
        </Card>
        <SectionRow title={t("overview.connect.curl")} />
        <CodeBlock code={curl} language="bash" filename={baseUrl} />
      </DashboardPage>
    );
  return (
    <DashboardPage width="7xl" data-testid="overview">
      <PageHeader
        title={t("overview.page.title")}
        description={t("overview.page.desc")}
        actions={
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => engine.setPolling(!engine.polling)}
            >
              {engine.polling ? <Pause size={13} /> : <Play size={13} />}
              {engine.polling
                ? t("overview.actions.pause")
                : t("overview.actions.resume")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void engine.refresh()}
            >
              <RefreshCw size={13} />
              {t("overview.actions.refresh")}
            </Button>
            <Button
              size="sm"
              disabled={
                !online ||
                !status?.models.some(
                  (model) => model.loaded && supportsChat(model),
                )
              }
              onClick={() => navigate("playground")}
            >
              <Play size={13} />
              {t("overview.actions.test")}
            </Button>
          </div>
        }
      />
      <StaleStamp engine={engine} />
      {/* One status block: verdict and version on top, the phase pipeline under it. */}
      <div className={dim} data-stale={stale ? "true" : undefined}>
        <StateStrip
          status={status ?? null}
          header={
            <div className="flex flex-wrap items-center justify-between gap-x-5 gap-y-1">
              <HealthLine
                verdict={verdict}
                checking={!status}
                navigate={navigate}
              />
              <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs tabular-nums text-muted-foreground">
                {(!online || (status && status.state !== "running")) && (
                  <StatusIndicator status={online ? "away" : "offline"}>
                    <span className="text-foreground">
                      {online
                        ? (status?.state ?? t("overview.status.connected"))
                        : t("overview.status.offline")}
                    </span>
                  </StatusIndicator>
                )}
                <span>Yunshu {status?.version ?? "—"}</span>
                <span>
                  {t("overview.status.uptime", {
                    t: elapsed(status?.uptime_s),
                  })}
                </span>
                <span>
                  {engine.updatedAt
                    ? t("overview.status.updated", {
                        t: clock(engine.updatedAt),
                      })
                    : t("overview.status.waiting")}
                </span>
                {!engine.polling && <span>{t("overview.status.paused")}</span>}
                {status?.load_error && (
                  <span className="text-error">{status.load_error}</span>
                )}
              </p>
            </div>
          }
        />
      </div>

      {/* Offline or refused: the banner says why; four empty placeholder cards would only add blank space.
          While connecting, the skeleton cross-fades into the cards in the same grid cell (no pop, no shift). */}
      {status || engine.phase === "connecting" ? (
        <SkeletonSwap
          ready={status != null}
          skeleton={
            <StatGrid data-stat-grid="">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-[104px] w-full rounded-lg" />
              ))}
            </StatGrid>
          }
        >
          {status && (
            <div
              className={`grid grid-cols-2 gap-2 sm:gap-3 xl:grid-cols-5 max-sm:[&>:last-child]:col-span-2 ${dim}`}
              data-testid="overview-stats"
              data-stat-grid=""
            >
              <SpeedPair status={status} />
              <StatCard
                compact
                valueFirst
                icon={Timer}
                label={t("overview.stats.ttft")}
                value={
                  <span title={t("overview.tip.ttft")}>
                    <StatValue
                      text={
                        last?.ttft_ms == null ? "—" : formatMs(last.ttft_ms)
                      }
                    />
                  </span>
                }
                subtext={
                  last
                    ? t("overview.stats.latestAge", {
                        age: elapsed(Math.max(0, Date.now() / 1000 - last.t)),
                      })
                    : t("overview.stats.noFinished")
                }
              />
              <StatCard
                compact
                valueFirst
                icon={Gauge}
                label={t("overview.stats.prefixRate")}
                value={
                  <span title={t("overview.tip.hit")}>
                    {hit.pct == null ? "—" : `${number(hit.pct, 0)}%`}
                  </span>
                }
                subtext={prefixHitWindow(
                  hit,
                  recentRequests.supported === true,
                )}
              />
              <StatCard
                compact
                valueFirst
                icon={HardDrive}
                label={t("overview.stats.metal")}
                value={
                  <span title={t("overview.tip.metal")}>
                    <span className="tabular-nums">
                      <LiveNumber
                        value={memory?.active_gb ?? null}
                        digits={1}
                      />
                    </span>
                    <span className="ml-1 text-xs font-normal text-muted-foreground">
                      GB
                    </span>
                  </span>
                }
                subtext={t("overview.stats.metalSub", {
                  total: number(memory?.total_gb),
                  peak: number(memory?.peak_gb),
                })}
              />
            </div>
          )}
        </SkeletonSwap>
      ) : null}

      {/* Live: what the engine is doing right now. Idle collapses to one compact card. */}
      <Card
        className={`grid min-w-0 overflow-hidden p-0 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] ${dim}`}
        data-testid="live-panel"
      >
        <div className="flex min-w-0 flex-col justify-between gap-5 p-4 max-lg:order-2">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs text-muted-foreground">
                {heroMetric === "decode"
                  ? t("overview.hero.decode")
                  : t("overview.hero.prefill")}
              </p>
              <div className="flex items-center gap-2">
                <SegmentedTray
                  aria-label={t("overview.hero.window")}
                  value={heroWindow}
                  onChange={setHeroWindow}
                  options={[
                    { value: "60", label: t("overview.hero.window60") },
                    { value: "300", label: t("overview.hero.window300") },
                  ]}
                />
                <SegmentedTray
                  aria-label={t("overview.hero.metric")}
                  value={heroMetric}
                  onChange={setHeroMetric}
                  options={[
                    { value: "decode", label: t("overview.series.decode") },
                    { value: "prefill", label: t("overview.series.prefill") },
                  ]}
                />
              </div>
            </div>
            <SeriesChart
              busy={busy}
              className="mt-3"
              data={heroData}
              series={rates[heroMetric as "decode" | "prefill"]}
              height={150}
              idleLabel={t("overview.chart.quiet")}
              ariaLabel={
                heroMetric === "decode"
                  ? t("overview.hero.ariaDecode")
                  : t("overview.hero.ariaPrefill")
              }
              formatX={clock}
              formatY={formatNumber}
              maxGap={12000}
              liveWindowMs={heroWiden ? 900_000 : heroSpanMs}
              smoothed
              tools={{ smooth: true }}
              markers={marksIn(end - (heroWiden ? 900_000 : heroSpanMs), end)}
            />
            {(heroMark || heroWiden) && (
              <p
                className="mt-1 text-xs text-muted-foreground"
                data-testid="hero-boundary"
              >
                {[
                  heroWiden ? t("overview.hero.widened") : "",
                  heroMark
                    ? t(
                        heroMark.kind === "switched"
                          ? "overview.hero.switched"
                          : heroMark.kind === "away"
                            ? "overview.hero.away"
                            : "overview.hero.restarted",
                        { time: clock(heroMark.at) },
                      )
                    : "",
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
          </div>
        </div>
        <div className="flex min-w-0 flex-col gap-3 p-4 max-lg:order-1">
          <div className="flex items-center justify-between gap-3">
            <h2 className="yunui-section-title text-base font-semibold">
              {t("overview.active.title")}
              <span className="ml-2 text-muted-foreground tabular-nums">
                {number(status?.requests.active, 0)}
              </span>
            </h2>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <Slot ch={14} align="right">
                {t("overview.active.window", {
                  s: number(status?.throughput.window_s ?? 60, 0),
                  n: number(status?.throughput.requests, 0),
                })}
              </Slot>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigate("requests")}
              >
                {t("overview.active.all")}
                <ArrowRight size={13} />
              </Button>
            </div>
          </div>
          {/* Five rows of reserved space: requests come and go inside it, nothing below moves. */}
          <div
            className="relative flex-1"
            style={{ minHeight: online ? LANE_AREA_PX : 0 }}
          >
            <ul
              className="divide-y divide-border/60 [&>li:has(+li[aria-hidden])]:border-b-0 [&>li[aria-hidden]]:border-0"
              data-live-list=""
              ref={laneRef}
            >
              {lanes.map((lane, i) =>
                !online && lanes.every((l) => !l) ? null : lane ? (
                  <RequestLane
                    key={`slot-${i}`}
                    row={lane.row}
                    leaving={lane.leaving}
                  />
                ) : (
                  <li key={`slot-${i}`} className="h-16" aria-hidden />
                ),
              )}
            </ul>
            {lanes.every((l) => !l) && (
              <p className="absolute inset-x-0 top-0 py-3 text-xs text-muted-foreground">
                {memory?.cache_gb
                  ? t("overview.active.idleDescPool", {
                      gb: number(memory.cache_gb),
                    })
                  : t("overview.active.idleDesc")}
              </p>
            )}
          </div>
          <TotalsLine totals={totals} />
        </div>
      </Card>

      {hostShown && (
        <FoldSection title={t("overview.host.title")}>
          <HostPanel state={hostState} now={Date.now()} />
        </FoldSection>
      )}

      <FoldSection title={t("overview.quick.title")}>
        <SectionRow title={t("overview.quick.title")} />
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <QuickAction
            icon={<Play size={18} strokeWidth={1.5} />}
            title={t("overview.quick.prompt.title")}
            caption={t("overview.quick.prompt.caption")}
            onClick={() => navigate("playground")}
          />
          <QuickAction
            icon={<HardDrive size={18} strokeWidth={1.5} />}
            title={t("overview.quick.load.title")}
            caption={t("overview.quick.load.caption")}
            onClick={() => navigate("models")}
          />
          <QuickAction
            icon={
              copied === "url" ? (
                <Check size={18} strokeWidth={1.5} />
              ) : (
                <Link2 size={18} strokeWidth={1.5} />
              )
            }
            title={
              copied === "url"
                ? t("overview.quick.url.copied")
                : t("overview.quick.url.title")
            }
            caption={`${baseUrl}/v1`}
            onClick={() => copy("url", `${baseUrl}/v1`)}
          />
          <QuickAction
            icon={<Stethoscope size={18} strokeWidth={1.5} />}
            title={t("overview.quick.diag.title")}
            caption={t("overview.quick.diag.caption")}
            onClick={() => navigate("diagnostics")}
          />
        </div>
        <CodeBlock code={curl} language="bash" filename="curl" />
      </FoldSection>

      <FoldSection title={t("overview.perf.title")}>
        {/* One crosshair, one zoom window and one time range for every chart below (hover a chart, all read that moment). */}
        <MetricChartGroup>
          <div className={`space-y-6 ${dim}`}>
            <SectionRow
              title={t("overview.perf.title")}
              action={
                <div className="flex flex-wrap items-center gap-2">
                  <SegmentedTray
                    aria-label={t("overview.perf.range")}
                    value={range}
                    onChange={chooseRange}
                    options={[
                      { value: "15m", label: t("overview.perf.range15m") },
                      { value: "1h", label: t("overview.perf.range1h") },
                      { value: "6h", label: t("overview.perf.range6h") },
                      { value: "24h", label: t("overview.perf.range24h") },
                      { value: "7d", label: t("overview.perf.range7d") },
                      { value: "30d", label: t("overview.perf.range30d") },
                      { value: "start", label: t("overview.perf.rangeStart") },
                    ]}
                  />
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={!points.length}
                    onClick={exportData}
                  >
                    <Download size={13} />
                    {t("overview.perf.export")}
                  </Button>
                </div>
              }
            />
            <div className="-mt-3 space-y-0.5 text-xs text-muted-foreground">
              <p>
                {engine.historyFrom != null
                  ? t("overview.perf.noteEngine", {
                      t: clock(engine.historyFrom),
                      n: points.length,
                    })
                  : t("overview.perf.noteLocal", { n: points.length })}
              </p>
              <p>{t("overview.chart.scrub")}</p>
            </div>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
              <ChartCard
                data-testid="throughput-panel"
                title={t("overview.throughput.title")}
                description={t("overview.throughput.desc")}
                action={
                  <SegmentedTray
                    aria-label={t("overview.throughput.metric")}
                    value={metric}
                    onChange={setMetric}
                    options={[
                      { value: "decode", label: t("overview.series.decode") },
                      { value: "prefill", label: t("overview.series.prefill") },
                      {
                        value: "both",
                        label: t("overview.throughput.compare"),
                      },
                    ]}
                  />
                }
              >
                <SeriesChart
                  busy={busy}
                  data={data}
                  series={throughputSeries}
                  height={220}
                  ariaLabel={t("overview.throughput.aria")}
                  formatX={clock}
                  formatY={formatNumber}
                  maxGap={rangeGapMs}
                  liveWindowMs={longRange.active ? undefined : rangeMs}
                  refreshing={longRange.loading}
                  markers={perfMarks}
                  smoothed
                  tools={{ smooth: true, log: true, expand: true }}
                  expandTitle={t("overview.throughput.title")}
                />
                <Button
                  className="mt-2"
                  size="sm"
                  variant="ghost"
                  aria-expanded={table}
                  onClick={() => setTable(!table)}
                >
                  {table
                    ? t("overview.throughput.hideTable")
                    : t("overview.throughput.showTable")}
                </Button>
                {table && (
                  <ScrollFade className="mt-3 max-h-60 overflow-auto">
                    <Table scrollLabel={t("overview.throughput.tableLabel")}>
                      <Thead>
                        <Tr>
                          <Th>{t("overview.throughput.colTime")}</Th>
                          <Th>{t("overview.series.decode")}</Th>
                          <Th>{t("overview.series.prefill")}</Th>
                          <Th>Metal GB</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {points.slice(-200).map((p) => (
                          <Tr key={p.at}>
                            <Td>{clock(p.at)}</Td>
                            <Td>{number(p.decode)}</Td>
                            <Td>{number(p.prefill)}</Td>
                            <Td>{number(p.memActive)}</Td>
                          </Tr>
                        ))}
                      </Tbody>
                    </Table>
                  </ScrollFade>
                )}
              </ChartCard>
              <ChartCard
                data-testid="memory-panel"
                title={t("overview.stats.metal")}
                description={t("overview.memory.source")}
                action={
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {t("overview.memory.physical", {
                      gb: number(memory?.total_gb),
                    })}
                  </span>
                }
              >
                <p className="text-2xl font-semibold tabular-nums">
                  <span className="tabular-nums">
                    {fixed(memory?.active_gb)}
                  </span>
                  <span className="ml-1.5 text-sm font-normal text-muted-foreground">
                    {t("overview.memory.activeOf", {
                      gb:
                        memory?.total_gb != null
                          ? gbTotalText(memory.total_gb)
                          : fixed(memory?.total_gb),
                    })}
                  </span>
                </p>
                {/* the legend wraps to a second line when the numbers get wider: both lines are reserved */}
                <div className="mt-4 min-h-[54px]">
                  <SegmentedBar
                    height={10}
                    total={memory?.total_gb ?? undefined}
                    label={t("overview.memory.barLabel", {
                      active: number(memory?.active_gb),
                      pool: number(memory?.cache_gb),
                    })}
                    segments={[
                      {
                        value: memory?.active_gb ?? 0,
                        tone: "accent",
                        label: t("overview.series.active"),
                      },
                      {
                        value: memory?.cache_gb ?? 0,
                        tone: "neutral",
                        label: t("overview.series.pool"),
                      },
                    ]}
                    marks={
                      memory?.peak_gb
                        ? [
                            {
                              value: memory.peak_gb,
                              label: t("overview.memory.peakMark"),
                            },
                          ]
                        : undefined
                    }
                    legend
                    formatValue={(v) => `${number(v)} GB`}
                  />
                </div>
                <DetailList className="mt-4">
                  <DetailRow
                    label={t("overview.memory.peak")}
                    value={`${number(memory?.peak_gb)} GB`}
                  />
                  <DetailRow
                    label={t("overview.memory.weights")}
                    value={
                      loaded.length
                        ? [
                            loaded.reduce((s, m) => s + (m.size_gb ?? 0), 0) > 0
                              ? sizeGb(
                                  loaded.reduce(
                                    (s, m) => s + (m.size_gb ?? 0),
                                    0,
                                  ),
                                )
                              : null,
                            t("overview.memory.weightsCount", {
                              n: loaded.length,
                            }),
                          ]
                            .filter(Boolean)
                            .join(" · ")
                        : "—"
                    }
                  />
                </DetailList>
                <SeriesChart
                  busy={busy}
                  className="mt-4"
                  data={data}
                  series={memorySeries()}
                  height={120}
                  ariaLabel={t("overview.memory.aria")}
                  formatX={clock}
                  formatY={formatNumber}
                  maxGap={rangeGapMs}
                  liveWindowMs={longRange.active ? undefined : rangeMs}
                  refreshing={longRange.loading}
                  markers={perfMarks}
                  bands={memBands}
                  tools={{ expand: true }}
                  expandTitle={t("overview.stats.metal")}
                />
              </ChartCard>
            </div>

            <ChartCard
              data-testid="concurrency-panel"
              title={t("overview.concurrency.title")}
              description={t("overview.concurrency.desc")}
            >
              <SeriesChart
                busy={busy}
                data={data}
                series={requestSeries()}
                height={180}
                ariaLabel={t("overview.concurrency.aria")}
                formatX={clock}
                formatY={formatCount}
                maxGap={rangeGapMs}
                liveWindowMs={longRange.active ? undefined : rangeMs}
                refreshing={longRange.loading}
                markers={perfMarks}
                tools={{ expand: true }}
                expandTitle={t("overview.concurrency.title")}
              />
            </ChartCard>
          </div>
        </MetricChartGroup>
      </FoldSection>
      <FoldSection title={t("overview.models.title")}>
        <SectionRow
          title={t("overview.models.title")}
          action={
            <Button
              size="sm"
              variant="ghost"
              onClick={() => navigate("models")}
            >
              {t("overview.models.library")}
              <ArrowRight size={13} />
            </Button>
          }
        />
        <Card className="min-w-0 p-0">
          <ul className="divide-y divide-border/60 py-1">
            {(status?.models ?? []).slice(0, 6).map((model) => (
              <li key={model.id} className="px-3 py-0.5 sm:px-4">
                <HoverRow
                  onClick={() => navigate("models")}
                  aria-label={t("overview.models.open", {
                    name: modelLabel(model.id),
                  })}
                  className="flex items-center gap-3 px-2 py-2.5"
                >
                  <StatusIndicator
                    status={
                      model.loading
                        ? "away"
                        : model.loaded
                          ? "online"
                          : "neutral"
                    }
                  />
                  <span className="min-w-0 flex-1 truncate text-sm font-medium">
                    {modelLabel(model.id)}
                  </span>
                  <span className="hidden text-xs text-muted-foreground sm:inline">
                    {model.type}
                  </span>
                  <span className="w-20 text-right text-xs tabular-nums text-muted-foreground">
                    {sizeGb(model.size_gb)}
                  </span>
                  <span className="w-24 text-right text-xs text-muted-foreground">
                    {model.loading
                      ? t("overview.models.loading")
                      : model.loaded
                        ? model.expires_in_s != null
                          ? t("overview.models.unloadIn", {
                              t: elapsed(model.expires_in_s),
                            })
                          : model.pinned
                            ? t("overview.models.pinned")
                            : t("overview.models.loaded")
                        : t("overview.models.notLoaded")}
                  </span>
                </HoverRow>
              </li>
            ))}
            {!status?.models.length && (
              <li className="px-5 py-5 text-sm text-muted-foreground sm:px-6">
                {status
                  ? t("overview.models.none")
                  : t("overview.models.connectFirst")}
              </li>
            )}
          </ul>
        </Card>
      </FoldSection>
    </DashboardPage>
  );
}
