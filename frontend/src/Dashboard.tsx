import { gbTotalText } from "./byte-format";
import { LiveNumber } from "./LiveNumber";
import { rangeSeconds, useRangeHistory } from "./useRangeHistory";
import { useLinger } from "./motion/linger";
import { PrefillBar } from "./PrefillBar";
import { SegmentedTray } from "./SegmentedTray";
import { useEffect, useMemo, useState, type ReactNode } from "react";
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
import {
  ActivityPanel,
  ChartCard,
  LatencyPanel,
  PhasePanel,
  SeriesChart,
} from "./AnalyticsPanels";
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
import { HostPanel } from "./HostPanel";
import { useHostTelemetry } from "./host-hook";
const memorySeries = () => [
  {
    key: "active",
    label: t("overview.series.active"),
    tone: "accent" as const,
  },
  {
    key: "cache",
    label: t("overview.series.pool"),
    tone: "neutral" as const,
    dashed: true,
  },
];
const requestSeries = () => [
  { key: "requests", label: t("overview.series.all"), tone: "accent" as const },
  {
    key: "queued",
    label: t("overview.series.queued"),
    tone: "neutral" as const,
    dashed: true,
  },
  {
    key: "prefillRequests",
    label: t("overview.series.prefill"),
    tone: "neutral" as const,
  },
  {
    key: "decodeRequests",
    label: t("overview.series.decode"),
    tone: "accent" as const,
    dashed: true,
  },
];
const rateSeries = () => {
  const decode = {
    key: "decode",
    label: t("overview.series.decodeLive"),
    tone: "accent" as const,
  };
  const prefill = {
    key: "prefill",
    label: t("overview.series.prefillLive"),
    tone: "neutral" as const,
  };
  return { decode: [decode], prefill: [prefill], both: [decode, prefill] };
};
const formatNumber = (value: number) => number(value);
const formatCount = (value: number) => number(value, 0);
function RequestLane({ row, leaving }: { row: RequestRow; leaving?: boolean }) {
  useLocale();
  const phase = String(row.phase);
  const prompt = row.prompt_tokens ?? 0,
    cached = row.cached_tokens ?? 0;
  const progress =
    phase === "prefill" || phase === "starting"
      ? (row.percent ??
        (prompt > 0 && row.processed_tokens != null
          ? (row.processed_tokens / prompt) * 100
          : null))
      : null;
  return (
    <li
      className="live-row"
      data-leaving={leaving ? "" : undefined}
      aria-hidden={leaving ? true : undefined}
    >
      <div className="live-row-inner">
        <div className="live-appear grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 py-3">
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
            <div className="col-span-2 flex h-4 items-center">
              <PrefillBar row={row} className="h-1" caption />
            </div>
          ) : phase !== "prefill" && cached > 0 && prompt > 0 ? (
            <p className="col-span-2 truncate text-xs text-muted-foreground">
              {t("overview.lane.prefixHit", {
                cached: number(cached, 0),
                prompt: number(prompt, 0),
              })}
            </p>
          ) : null}
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
    <Card className="min-w-0 overflow-hidden p-0 transition-[box-shadow,transform] duration-[150ms] ease-out has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-(--border-strong) active:scale-[0.99] motion-reduce:transition-none motion-reduce:active:scale-100">
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
    [table, setTable] = useState(false),
    [activeX, setActiveX] = useState<number | null>(null),
    [copied, setCopied] = useState<string | null>(null);
  const status = engine.status,
    online = engine.phase === "online";
  const { verdict } = useSignals();
  // Offline with data on screen: the numbers stop moving, so they are dimmed and stamped.
  const stale = engine.phase === "offline" && status != null;
  const dim = stale ? "opacity-60" : "";
  const end = engine.updatedAt ?? Date.now(),
    start = end - rangeSeconds(range) * 1000;
  // Charts read the slim series (engine history first, then live polls). The
  // window is a binary-search slice and the chart gets at most 300 rows.
  // Up to an hour the live series (backfilled from the console's history on open) is the source;
  // 6 h to 30 d come straight from the recorded history, gaps included.
  const longRange = useRangeHistory(
    connection,
    range,
    engine.status?.console_process === true,
  );
  const livePoints = useMemo(
    () => windowPoints(engine.series, start, end),
    [engine.series, start, end],
  );
  const points = longRange.active ? longRange.points : livePoints;
  const rangeMs = rangeSeconds(range) * 1000;
  // A line breaks where samples are further apart than a few of this range's own steps.
  const rangeGapMs = Math.max(12_000, (longRange.resolutionS ?? 0) * 3_000);
  const data = useMemo(() => chartRows(points), [points]);
  const heroPoints = useMemo(
    () => windowPoints(engine.series, end - 300_000, end),
    [engine.series, end],
  );
  const rates = rateSeries();
  const throughputSeries = rates[metric as keyof typeof rates];
  const busy = (engine.status?.requests.active ?? 0) > 0;
  useEffect(() => {
    if (activeX !== null && !points.some((point) => point.at === activeX))
      setActiveX(null);
  }, [points, activeX]);
  const last = status?.last;
  const activePoint =
    activeX == null
      ? null
      : (points.find((sample) => sample.at === activeX) ?? null);
  const items = status?.requests.items ?? [];
  const lanes = useLinger(items, (r) => r.request_id);
  const few = items.length <= 2;
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
  // Prefix reuse is weighted over the observed finished requests (cached /
  // prompt tokens); the latest request is a secondary line, as on Requests.
  const promptSum = observed.reduce((n, r) => n + r.prompt_tokens, 0),
    cachedSum = observed.reduce((n, r) => n + r.cached_tokens, 0);
  const cache = promptSum > 0 ? (cachedSum / promptSum) * 100 : null;
  const lastHit =
    last && last.prompt_tokens > 0
      ? (last.cached_tokens / last.prompt_tokens) * 100
      : null;
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
  const chooseRange = (next: string) => {
    setRange(next);
    setActiveX(null);
  };
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
  if (
    !status &&
    neverConfigured &&
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
                checking={!status && engine.phase === "connecting"}
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

      {!status ? (
        // Offline or refused: the banner says why; four empty placeholder cards would only add blank space.
        engine.phase === "connecting" ? (
          <StatGrid data-stat-grid="">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-[104px] w-full rounded-lg" />
            ))}
          </StatGrid>
        ) : null
      ) : (
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
              <StatValue
                text={last?.ttft_ms == null ? "—" : formatMs(last.ttft_ms)}
              />
            }
            subtext={
              last
                ? t("overview.stats.latestAt", { t: clock(last.t * 1000) })
                : t("overview.stats.noFinished")
            }
          />
          <StatCard
            compact
            valueFirst
            icon={Gauge}
            label={t("overview.stats.prefixRate")}
            value={cache == null ? "—" : `${number(cache, 0)}%`}
            subtext={
              cache == null
                ? t("overview.stats.noFinished")
                : lastHit == null
                  ? t("overview.stats.observed", {
                      n: number(observed.length, 0),
                    })
                  : t("overview.stats.observedLast", {
                      n: number(observed.length, 0),
                      pct: number(lastHit, 0),
                    })
            }
          />
          <StatCard
            compact
            valueFirst
            icon={HardDrive}
            label={t("overview.stats.metal")}
            value={
              <>
                <LiveNumber value={memory?.active_gb ?? null} digits={1} />
                <span className="ml-1 text-xs font-normal text-muted-foreground">
                  GB
                </span>
              </>
            }
            subtext={t("overview.stats.metalSub", {
              total: number(memory?.total_gb),
              peak: number(memory?.peak_gb),
            })}
          />
        </div>
      )}

      {/* Live: what the engine is doing right now. Idle collapses to one compact card. */}
      {items.length === 0 && !busy ? (
        <Card
          className={`flex min-w-0 flex-col gap-3 p-4 ${dim}`}
          data-testid="live-panel"
          data-idle="true"
        >
          <div className="flex items-center justify-between gap-3">
            <h2 className="yunui-section-title text-base font-semibold">
              {t("overview.active.title")}
              <span className="ml-2 text-muted-foreground tabular-nums">
                {number(status?.requests.active, 0)}
              </span>
            </h2>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span className="max-sm:hidden">
                {t("overview.active.window", {
                  s: number(status?.throughput.window_s ?? 60, 0),
                  n: number(status?.throughput.requests, 0),
                })}
              </span>
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
          <p className="text-xs text-muted-foreground">
            {memory?.cache_gb
              ? t("overview.active.idleDescPool", {
                  gb: number(memory.cache_gb),
                })
              : t("overview.active.idleDesc")}
          </p>
          {recent && (
            <div className="min-w-0" data-testid="live-spark">
              <p className="text-xs text-muted-foreground">
                {t("overview.hero.decode")}
              </p>
              <SeriesChart
                busy={busy}
                className="mt-2"
                data={heroData}
                series={rates.decode}
                height={72}
                ariaLabel={t("overview.hero.ariaDecode")}
                formatX={clock}
                formatY={formatNumber}
                maxGap={12000}
              />
            </div>
          )}
          <TotalsLine totals={totals} />
        </Card>
      ) : (
        <Card
          className={`grid min-w-0 overflow-hidden p-0 ${few ? "" : "lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]"} ${dim}`}
          data-testid="live-panel"
          data-layout={few ? "stacked" : "split"}
        >
          <div
            className={`flex min-w-0 flex-col justify-between gap-5 p-4 ${few ? "order-2 pt-0" : "max-lg:order-2"}`}
          >
            <div className="min-w-0">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs text-muted-foreground">
                  {heroMetric === "decode"
                    ? t("overview.hero.decode")
                    : t("overview.hero.prefill")}
                </p>
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
              <SeriesChart
                busy={busy}
                className="mt-3"
                data={heroData}
                series={rates[heroMetric as "decode" | "prefill"]}
                height={150}
                ariaLabel={
                  heroMetric === "decode"
                    ? t("overview.hero.ariaDecode")
                    : t("overview.hero.ariaPrefill")
                }
                formatX={clock}
                formatY={formatNumber}
                maxGap={12000}
                liveWindowMs={300_000}
              />
            </div>
          </div>
          <div
            className={`flex min-w-0 flex-col gap-3 p-4 ${few ? "order-1" : "max-lg:order-1"}`}
          >
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
            <ul className={`divide-y divide-border/60 ${few ? "" : "flex-1"}`}>
              {lanes.slice(0, 5).map(({ row, leaving }) => (
                <RequestLane key={row.request_id} row={row} leaving={leaving} />
              ))}
            </ul>
            <TotalsLine totals={totals} />
          </div>
        </Card>
      )}

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
          <p className="-mt-3 text-xs text-muted-foreground">
            {engine.historyFrom != null
              ? t("overview.perf.noteEngine", {
                  t: clock(engine.historyFrom),
                  n: points.length,
                })
              : t("overview.perf.noteLocal", { n: points.length })}
          </p>
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
                    { value: "both", label: t("overview.throughput.compare") },
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
                activeX={activeX}
                onActiveXChange={setActiveX}
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
                <Slot ch={5}>{fixed(memory?.active_gb)}</Slot>
                <span className="ml-1.5 text-sm font-normal text-muted-foreground">
                  {t("overview.memory.activeOf", {
                    gb:
                      memory?.total_gb != null
                        ? gbTotalText(memory.total_gb)
                        : fixed(memory?.total_gb),
                  })}
                </span>
              </p>
              <SegmentedBar
                className="mt-4"
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
                activeX={activeX}
                onActiveXChange={setActiveX}
              />
            </ChartCard>
          </div>

          {activePoint && (
            <div
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/60 px-4 py-2 text-xs text-muted-foreground"
              role="status"
            >
              <span>
                {t("overview.selection.summary", {
                  t: clock(activePoint.at),
                  n: number(activePoint.active, 0),
                  gb: number(activePoint.memActive),
                })}
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setActiveX(null)}
              >
                {t("overview.selection.clear")}
              </Button>
            </div>
          )}
          <div className="grid min-w-0 gap-5 xl:grid-cols-2">
            <LatencyPanel records={observed} />
            <PhasePanel engine={engine} navigate={navigate} />
          </div>
          <ActivityPanel
            history={points}
            start={start}
            end={end}
            onSelectTime={setActiveX}
          />
          <ChartCard
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
              maxGap={12000}
              activeX={activeX}
              onActiveXChange={setActiveX}
            />
          </ChartCard>
        </div>
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
