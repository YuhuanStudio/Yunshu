import { FRESH_ROW, useFreshIds } from "./fresh-rows";
import { ErrorNote } from "./error-note";
import { RequestWaterfall } from "./RequestWaterfall";
import { LatencyDistribution } from "./LatencyDistribution";
import { SpeculationPanel } from "./SpeculationPanel";
import { useRequestArchive } from "./useRequestArchive";
import { ArchiveControls } from "./ArchiveControls";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  CustomSelect,
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  EmptyState,
  Input,
  Sheet,
  Sparkline,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
  Tr,
} from "@yuhuanowo/yunui";
import {
  DashboardPage,
  GroupLabel,
  PageHeader,
  StatCard,
  StatGrid,
  TableState,
  WorkspaceLayout,
} from "@yuhuanowo/yunui/patterns";
import { Activity, Copy, Gauge, Timer, Zap } from "lucide-react";
import { t } from "./i18n/index.ts";
import { rollingMedian, trendDelta } from "./analytics";
import { ArrowDown, ArrowUp, Download, ScrollText, Search } from "lucide-react";
import { ApiError, cancelRequest, requestJson, type Connection } from "./api";
import {
  causeOf,
  PrefillMeter,
  StageRail,
  TokenTrace,
  finishedFromHistory,
  isLive,
  median,
  phaseLabel,
  prefillPercent,
  speculativeText,
  type Row,
} from "./RequestTrace";
import {
  clock,
  dateTime,
  elapsed,
  isOnline,
  modelLabel,
  number,
  phaseDot,
  Readout,
  relative,
  useMinWidth,
  type Engine,
  StatValue,
} from "./ui";
import type { Perform } from "./Models";
import { outcomeLabel, useRecentRequests } from "./recentRequests";
import {
  RequestBreakdown,
  RequestTimeline,
  durationUnit,
  durationValue,
  formatMs,
} from "./RequestTimeline";
import {
  SLOW_TOTAL_MS,
  SLOW_TTFT_MS,
  isSlow,
  slowRule,
  sortRows,
  stages,
  type SortKey,
  type SortState,
} from "./request-insight";
import { logsRangeHash } from "./Logs";

import { SegmentedTray } from "./SegmentedTray";
const PAGE = 50;
/** Long ids keep both ends so ids that differ only at the tail stay distinguishable. */
const shortId = (id: string) =>
  id.length > 20 ? `${id.slice(0, 9)}…${id.slice(-8)}` : id;
/** Error is the only red dot; a cancel is the user's own choice, so it stays neutral (amber means in progress). */
const outcomeDot = (o?: string): "busy" | "neutral" =>
  o === "error" ? "busy" : "neutral";
function csv(rows: Row[]) {
  const keys: (keyof Row)[] = [
    "id",
    "model",
    "phase",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "ttft_ms",
    "elapsed_s",
  ];
  const quote = (v: unknown) =>
    '"' +
    String(v ?? "")
      .replace(/^[=+\-@]/, "'$&")
      .replaceAll('"', '""') +
    '"';
  const blob = new Blob(
    [
      "\uFEFF" +
        [
          keys.join(","),
          ...rows.map((row) => keys.map((k) => quote(row[k])).join(",")),
        ].join("\r\n"),
    ],
    { type: "text/csv;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "yunshu-observed-requests.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
/** Relative text against the last engine poll, so rows never need their own timer. */
function relativeTime(thenMs: number, nowMs: number) {
  const s = Math.max(0, Math.round((nowMs - thenMs) / 1000));
  return s < 5 ? t("requests.time.justNow") : relative(s);
}

/** Arrival time of a row in ms (server `t0_wall`, else finish time), null when unknown. */
const rowMs = (row: Row) => {
  const v = row.t0_wall ?? row.t;
  return v != null && Number.isFinite(v) ? v * 1000 : null;
};

/** Header cell that sorts: first click descending (slowest / newest first), second click reverses. */
function SortTh({
  k,
  sort,
  setSort,
  className,
  children,
}: {
  k: SortKey;
  sort: SortState;
  setSort: (s: SortState) => void;
  className?: string;
  children: ReactNode;
}) {
  const on = sort.key === k;
  return (
    <Th
      className={className}
      aria-sort={
        on ? (sort.dir === "asc" ? "ascending" : "descending") : "none"
      }
    >
      <Button
        variant="ghost"
        size="sm"
        title={t("requests.list.sortHint")}
        className="h-auto gap-1 whitespace-nowrap p-0 font-[inherit] text-[length:inherit] hover:bg-transparent hover:text-foreground"
        onClick={() =>
          setSort({ key: k, dir: on && sort.dir === "desc" ? "asc" : "desc" })
        }
      >
        {children}
        {on &&
          (sort.dir === "asc" ? (
            <ArrowUp size={12} />
          ) : (
            <ArrowDown size={12} />
          ))}
      </Button>
    </Th>
  );
}

const hasTrend = (data: number[]) => data.length >= 5 && new Set(data).size > 1;

export function Requests({
  engine,
  connection,
  perform,
  busy,
}: {
  engine: Engine;
  connection: Connection;
  perform: Perform;
  busy: string | null;
}) {
  const [filterChoice, setFilterChoice] = useState<string | null>(null),
    [outcome, setOutcome] = useState("all"),
    [speed, setSpeed] = useState("all"),
    [sort, setSort] = useState<SortState>({ key: "time", dir: "desc" }),
    [limit, setLimit] = useState(PAGE),
    [copied, setCopied] = useState<string | null>(null),
    [query, setQuery] = useState(""),
    [detail, setDetail] = useState<Row | null>(null),
    [cancel, setCancel] = useState<Row | null>(null),
    opener = useRef<HTMLButtonElement | null>(null),
    detailOpener = useRef<HTMLButtonElement | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null),
    [detailUpdated, setDetailUpdated] = useState<number | null>(null);
  useEffect(() => {
    setDetailError(null);
    setDetailUpdated(null);
    if (!detail || detail.phase === "complete") return;
    const id = detail.id,
      controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined,
      ended = false;
    const poll = async () => {
      try {
        const value = await requestJson<Record<string, unknown>>(
          connection,
          `/requests/${encodeURIComponent(id)}`,
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        if (value.request_id !== id)
          throw Error(t("requests.detail.idMismatch"));
        setDetail((current) =>
          current?.id === id ? ({ ...current, ...value, id } as Row) : current,
        );
        setDetailError(null);
        setDetailUpdated(Date.now());
      } catch (e) {
        if (controller.signal.aborted) return;
        ended = e instanceof ApiError && e.status === 404;
        setDetailError(
          ended
            ? t("requests.detail.ended")
            : e instanceof Error
              ? e.message
              : t("requests.detail.loadFailed"),
        );
      } finally {
        if (!controller.signal.aborted && !ended)
          timer = setTimeout(() => void poll(), 1500);
      }
    };
    void poll();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [detail?.id, connection.baseUrl, connection.token]);
  const recent = useRecentRequests(
    connection,
    engine.status?.last?.request_id,
    engine.status?.console_process === true,
  );
  const sampled = useMemo(
    () =>
      finishedFromHistory(engine.history).map(
        (row) => ({ ...row, source: "sampled" as const }) satisfies Row,
      ),
    [engine.history],
  );
  // The server ring when it exists; otherwise only what this page sampled from status polls.
  const archive = useRequestArchive(connection.baseUrl, recent.rows);
  const finished = useMemo(
    () => (recent.supported ? [...archive.rows].reverse() : sampled),
    [recent.supported, archive.rows, sampled],
  );
  // The first tab is decided once from what is running, so it never flips under the reader:
  // 進行中 when something runs, otherwise the finished requests (the thing to investigate).
  useEffect(() => {
    if (filterChoice == null && engine.status)
      setFilterChoice(
        engine.status.requests.items.length > 0 ? "active" : "complete",
      );
  }, [filterChoice, engine.status]);
  const filter = filterChoice ?? "complete",
    setFilter = setFilterChoice;
  useEffect(() => setLimit(PAGE), [filter, outcome, query, speed, sort]);
  const rows = useMemo(() => {
    const observed = new Map<string, Row>();
    for (const row of finished) observed.set(row.id, row);
    for (const row of engine.status?.requests.items ?? [])
      observed.set(row.request_id, {
        ...row,
        id: row.request_id,
        t0_wall: (engine.updatedAt ?? Date.now()) / 1000 - (row.elapsed_s ?? 0),
      } as Row);
    return [...observed.values()].reverse();
  }, [finished, engine.status, engine.updatedAt]);
  const scoped = rows.filter(
    (row) =>
      (filter === "all" ||
        (filter === "active"
          ? row.phase !== "complete"
          : row.phase === "complete")) &&
      (outcome === "all" ||
        (row.phase === "complete" && row.outcome === outcome)) &&
      `${row.id} ${row.model ?? ""}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()),
  );
  // Slow thresholds come from the finished requests of the current view (p90 from n >= 20).
  const rule = useMemo(
    () => slowRule(scoped.filter((r) => r.phase === "complete")),
    [scoped],
  );
  const bySpeed =
    speed === "all"
      ? scoped
      : scoped.filter((row) => {
          if (row.phase !== "complete") return false;
          const s = stages(row);
          return speed === "slow"
            ? isSlow(row, rule)
            : speed === "ttft"
              ? s.ttft != null && s.ttft > SLOW_TTFT_MS
              : s.total != null && s.total > SLOW_TOTAL_MS;
        });
  const matched = useMemo(() => sortRows(bySpeed, sort), [bySpeed, sort]);
  const shown = matched.slice(0, limit);
  const fresh = useFreshIds(shown.map((r) => r.id));
  const restore = (e: Event) => {
    if (opener.current?.isConnected) {
      e.preventDefault();
      opener.current.focus();
    }
  };
  const status = engine.status,
    activeNow = status?.requests.active ?? 0;
  const activeSeries = engine.history.map((p) => p.status.requests.active);
  const decodeSeries = engine.history.flatMap((p) => {
    const v =
      p.status.throughput.live_decode_tps ??
      p.status.throughput.mean_decode_tps;
    return v == null ? [] : [v];
  });
  const ttfts = finished.flatMap((r) => (r.ttft_ms == null ? [] : [r.ttft_ms]));
  const ttftMedian = ttfts.length ? median(ttfts) : null;
  const hits = finished.flatMap((r) =>
    (r.prompt_tokens ?? 0) > 0
      ? [((r.cached_tokens ?? 0) / (r.prompt_tokens ?? 1)) * 100]
      : [],
  );
  const promptSum = finished.reduce((n, r) => n + (r.prompt_tokens ?? 0), 0),
    cachedSum = finished.reduce((n, r) => n + (r.cached_tokens ?? 0), 0);
  const retention = {
    since: Math.min(
      ...finished.map((r) => r.t ?? Infinity).concat(Date.now() / 1000),
    ),
  };
  const liveDecode =
    status?.throughput.live_decode_tps ?? status?.throughput.mean_decode_tps;
  const tiles = [
    {
      label: t("requests.tile.activeLabel"),
      value: status ? number(activeNow, 0) : "—",
      hint: status
        ? t("requests.tile.activeHint", {
            queued: number(status.requests.queued, 0),
            prefill: number(status.requests.prefill, 0),
            decode: number(status.requests.decode, 0),
          })
        : undefined,
      data: activeSeries,
      tone: "accent" as const,
      name: t("requests.tile.activeName"),
      icon: Activity,
      trend: null,
    },
    {
      label: t("requests.tile.ttftLabel"),
      value: ttftMedian != null ? durationValue(ttftMedian) : "—",
      unit: ttftMedian != null ? durationUnit(ttftMedian) : undefined,
      hint: ttfts.length
        ? t(
            recent.supported
              ? "requests.tile.ttftServer"
              : "requests.tile.ttftPage",
            {
              count: number(ttfts.length, 0),
              last: formatMs(ttfts.at(-1) ?? 0),
            },
          )
        : t("requests.tile.noFinished"),
      data: rollingMedian(ttfts),
      tone: "accent" as const,
      name: t("requests.tile.ttftName"),
      icon: Timer,
      trend: null,
    },
    {
      label: t("requests.tile.decodeLabel"),
      value: number(liveDecode),
      unit: liveDecode == null ? undefined : "tok/s",
      hint: t("requests.tile.decodeHint"),
      data: decodeSeries,
      tone: "accent" as const,
      name: t("requests.tile.decodeName"),
      icon: Zap,
      trend: trendDelta(decodeSeries),
    },
    {
      label: t("requests.tile.hitLabel"),
      value: promptSum > 0 ? number((cachedSum / promptSum) * 100, 0) : "—",
      unit: promptSum > 0 ? "%" : undefined,
      hint:
        promptSum > 0
          ? t(
              recent.supported
                ? "requests.tile.hitServer"
                : "requests.tile.hitPage",
              { count: number(finished.length, 0) },
            ) +
            (hits.length
              ? t("requests.tile.hitLatest", { value: number(hits.at(-1), 0) })
              : "")
          : t("requests.tile.noFinished"),
      data: rollingMedian(hits),
      tone: "accent" as const,
      name: t("requests.tile.hitName"),
      icon: Gauge,
      trend: null,
    },
  ];
  const xl = useMinWidth(1280);
  // The inspector column is not a dialog, so it handles Escape itself and gives
  // focus back to the button that opened it.
  useEffect(() => {
    if (!xl || !detail) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setDetail(null);
      detailOpener.current?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [xl, detail]);
  const view =
    (detail &&
      finished.find(
        (r) =>
          r.id === detail.id &&
          (r.source === "ring" ||
            r.source === "archive" ||
            r.source === "history"),
      )) ||
    detail;
  const done = view?.phase === "complete";
  const copyId = (id: string) => {
    void navigator.clipboard?.writeText(id).then(() => {
      setCopied(id);
      setTimeout(() => setCopied((c) => (c === id ? null : c)), 1500);
    });
  };
  const detailBody = view ? (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <StatusIndicator status={phaseDot(view.phase)}>
            <span className="text-sm font-medium">
              {phaseLabel(view.phase)}
            </span>
          </StatusIndicator>
        </div>
        {done && view.outcome && (
          <div className="flex min-h-6 flex-wrap items-center gap-2">
            <StatusIndicator
              className="gap-1.5 text-xs text-muted-foreground"
              status={outcomeDot(view.outcome)}
            >
              {outcomeLabel(view.outcome)}
            </StatusIndicator>
            {view.finish_reason && (
              <span className="font-mono text-xs text-muted-foreground">
                {view.finish_reason}
              </span>
            )}
          </div>
        )}
        <div className="flex items-start gap-2">
          <p className="min-w-0 flex-1 break-all font-mono text-xs">
            {view.id}
          </p>
          <Button
            size="sm"
            variant="ghost"
            aria-label={t("requests.detail.copyId")}
            onClick={() => copyId(view.id)}
          >
            <Copy size={14} />
            {copied === view.id
              ? t("requests.detail.copied")
              : t("requests.detail.copy")}
          </Button>
        </div>
        {detailError && (
          <p role="status" className="text-xs text-warning">
            {detailError}
          </p>
        )}
        {detailUpdated && (
          <p className="text-xs text-muted-foreground">
            {t("requests.detail.liveUpdated", { time: clock(detailUpdated) })}
          </p>
        )}
      </div>
      {done && (
        <div className="space-y-1.5" data-testid="request-cause">
          <p className="text-xs text-muted-foreground">
            {t("requests.detail.cause")}
          </p>
          <p className="text-sm">{causeOf(view)}</p>
          {rowMs(view) != null && (
            <Button size="sm" variant="ghost" asChild className="-ml-2">
              <a
                href={logsRangeHash(
                  rowMs(view)! / 1000 - 30,
                  (view.t ?? rowMs(view)! / 1000) + 30,
                )}
              >
                <ScrollText size={14} />
                {t("requests.detail.logs")}
              </a>
            </Button>
          )}
        </div>
      )}
      {done && (
        <section
          className="space-y-4"
          aria-label={t("requests.detail.breakdownLabel")}
        >
          <RequestBreakdown row={view} />
          <RequestTimeline row={view} />
          <RequestWaterfall row={view} />
          <p className="text-xs text-muted-foreground">
            {t("requests.detail.timelineNote")}
          </p>
        </section>
      )}
      <section
        className="space-y-3"
        aria-label={t("requests.detail.stagesLabel")}
      >
        {!done && <StageRail phase={view.phase} />}
        {view.phase === "queued" && (
          <div className="grid grid-cols-2 gap-5">
            <Readout
              label={t("requests.detail.queuePosition")}
              value={number(view.queue_position, 0)}
            />
            <Readout
              label={t("requests.detail.queueWait")}
              value={number(view.queue_est_wait_ms, 0)}
              unit="ms"
            />
          </div>
        )}
        <PrefillMeter row={view} />
        <TokenTrace row={view} />
      </section>
      <div className="grid grid-cols-2 gap-5">
        <Readout
          label={t("requests.detail.model")}
          value={
            view.model
              ? modelLabel(view.model)
              : t("requests.detail.notReported")
          }
        />
        {!done && (
          <>
            <Readout
              label={t("requests.detail.elapsed")}
              value={elapsed(view.elapsed_s)}
            />
            <Readout
              label={t("requests.detail.inputTokens")}
              value={number(view.prompt_tokens, 0)}
            />
            <Readout
              label={t("requests.detail.cachedTokens")}
              value={number(view.cached_tokens, 0)}
            />
            <Readout
              label={t("requests.detail.outputTokens")}
              value={number(view.completion_tokens, 0)}
            />
            <Readout
              label={t("requests.detail.ttft")}
              value={number(view.ttft_ms)}
              unit="ms"
            />
          </>
        )}
        <Readout
          label={t("requests.detail.decode")}
          value={number(
            view.decode_tps ??
              (view.phase === "decode" ? view.tokens_per_second : null),
          )}
          unit="tok/s"
        />
        <Readout
          label={t("requests.detail.prefill")}
          value={number(
            view.prefill_tps ??
              (view.phase === "prefill" ? view.tokens_per_second : null),
          )}
          unit="tok/s"
        />
      </div>
      {view.speculative && (
        <div className="grid grid-cols-3 gap-5 pt-1">
          <Readout
            label={t("requests.detail.speculative")}
            value={view.speculative.mode ?? "—"}
          />
          <Readout
            label={t("requests.detail.acceptance")}
            value={number(
              view.speculative.acceptance_rate == null
                ? null
                : view.speculative.acceptance_rate * 100,
              0,
            )}
            unit="%"
          />
          <Readout
            label={t("requests.detail.rounds")}
            value={number(view.speculative.rounds, 0)}
          />
        </div>
      )}
    </div>
  ) : null;
  // The trend slot is reserved only once some tile has a trend to draw; before that the cards stay compact.
  const anySpark = tiles.some((tile) => hasTrend(tile.data));
  return (
    <DashboardPage width="7xl" data-testid="requests">
      <PageHeader
        title={t("requests.page.title")}
        description={t("requests.page.description")}
        actions={
          <Button
            size="sm"
            variant="secondary"
            disabled={!shown.length}
            onClick={() => csv(shown)}
          >
            <Download size={14} />
            {t("requests.page.exportCsv")}
          </Button>
        }
      />
      <StatGrid data-stat-grid="" data-testid="request-stats">
        {tiles.map((tile) => {
          // A trend line needs a few samples that actually vary; a flat run of zeros is not data.
          const spark = hasTrend(tile.data);
          return (
            <StatCard
              key={tile.label}
              compact
              valueFirst
              icon={tile.icon}
              label={tile.label}
              value={<StatValue text={tile.value} unit={tile.unit} />}
              trend={tile.trend ?? undefined}
              subtext={
                <span className="block min-w-0 space-y-1 sm:space-y-2">
                  <span className="block whitespace-normal">{tile.hint}</span>
                  {/* The trend slot keeps its height before and after samples arrive; only the line waits for data. */}
                  <span
                    className={`hidden sm:block ${anySpark ? "h-7" : "sm:hidden"}`}
                  >
                    {spark && (
                      <Sparkline
                        data={tile.data.slice(-60)}
                        tone={tile.tone}
                        area
                        height={28}
                        className="h-7 w-full"
                        label={tile.name}
                      />
                    )}
                  </span>
                </span>
              }
            />
          );
        })}
      </StatGrid>
      {recent.supported && <LatencyDistribution rows={finished} />}
      {recent.supported && (
        <SpeculationPanel rows={finished} connection={connection} />
      )}
      <div className="flex flex-wrap justify-between gap-3">
        <div className="w-full sm:w-auto sm:max-w-xs">
          <Input
            className="w-full"
            aria-label={t("requests.list.search")}
            icon={<Search size={14} />}
            placeholder={t("requests.list.searchPlaceholder")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="grid w-full grid-cols-3 items-center gap-2 sm:flex sm:w-auto sm:flex-wrap">
          <SegmentedTray
            fillOnPhone
            className="col-span-3 sm:col-auto"
            aria-label={t("requests.list.scope")}
            value={filter}
            onChange={(v) => {
              setFilter(v);
              if (v === "active") setOutcome("all");
            }}
            options={[
              { value: "active", label: t("requests.list.scopeActive") },
              {
                value: "complete",
                label: t("requests.list.scopeComplete"),
              },
              { value: "all", label: t("requests.list.scopeAll") },
            ]}
          />
          <div
            className={`min-w-0 sm:col-auto sm:flex-none ${filter === "active" ? "col-span-3" : ""}`}
          >
            <CustomSelect
              className="w-full sm:w-40 [&_button]:h-8 [&_button]:text-xs"
              aria-label={t("requests.list.sortBy")}
              value={sort.key}
              onChange={(v) => setSort({ key: v as SortKey, dir: "desc" })}
              options={[
                { value: "time", label: t("requests.list.sortTime") },
                { value: "ttft", label: t("requests.list.sortTtft") },
                { value: "total", label: t("requests.list.sortTotal") },
                { value: "tps", label: t("requests.list.sortTps") },
              ]}
            />
          </div>
          {filter !== "active" && (
            <>
              <CustomSelect
                className="w-full sm:w-36 [&_button]:h-8 [&_button]:text-xs"
                value={outcome}
                onChange={setOutcome}
                options={[
                  { value: "all", label: t("requests.list.outcomeAll") },
                  {
                    value: "completed",
                    label: t("requests.list.outcomeCompleted"),
                  },
                  {
                    value: "cancelled",
                    label: t("requests.list.outcomeCancelled"),
                  },
                  {
                    value: "error",
                    label: t("requests.list.outcomeError"),
                  },
                ]}
              />
              <CustomSelect
                className="w-full sm:w-44 [&_button]:h-8 [&_button]:text-xs"
                aria-label={t("requests.list.speedAria")}
                value={speed}
                onChange={setSpeed}
                options={[
                  { value: "all", label: t("requests.list.speedAll") },
                  { value: "slow", label: t("requests.list.speedSlow") },
                  { value: "ttft", label: t("requests.list.speedTtft") },
                  {
                    value: "total",
                    label: t("requests.list.speedTotal"),
                  },
                ]}
              />
            </>
          )}
        </div>
      </div>
      <WorkspaceLayout
        detailLabel={t("requests.detail.title")}
        list={
          <div className="space-y-5">
            <Card className="overflow-hidden">
              <div className="min-h-24">
                <TooltipProvider delayDuration={200}>
                  <Table scrollLabel={t("requests.list.tableLabel")}>
                    <Thead>
                      <Tr>
                        <SortTh
                          k="time"
                          sort={sort}
                          setSort={setSort}
                          className="w-24 max-sm:hidden"
                        >
                          {t("requests.list.colTime")}
                        </SortTh>
                        <Th>{t("requests.list.colRequest")}</Th>
                        <Th className="hidden w-36 lg:table-cell">
                          {t("requests.list.colUsage")}
                        </Th>
                        <SortTh
                          k="ttft"
                          sort={sort}
                          setSort={setSort}
                          className="hidden w-24 md:table-cell"
                        >
                          {t("requests.list.colTtft")}
                        </SortTh>
                        <SortTh
                          k="total"
                          sort={sort}
                          setSort={setSort}
                          className="hidden w-24 md:table-cell"
                        >
                          {t("requests.list.colTotal")}
                        </SortTh>
                        <SortTh
                          k="tps"
                          sort={sort}
                          setSort={setSort}
                          className="hidden w-24 md:table-cell"
                        >
                          tok/s
                        </SortTh>
                        <Th className="hidden w-24 min-[1800px]:table-cell">
                          {t("requests.list.colSpec")}
                        </Th>
                        <Th className="w-28">
                          <span className="sr-only">
                            {t("requests.list.colActions")}
                          </span>
                        </Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {shown.map((row) => {
                        const percent =
                            row.phase === "prefill"
                              ? prefillPercent(row)
                              : null,
                          speedValue =
                            row.phase === "complete"
                              ? row.decode_tps
                              : row.tokens_per_second,
                          st = stages(row),
                          slow = row.phase === "complete" && isSlow(row, rule);
                        const when = rowMs(row),
                          now = engine.updatedAt ?? Date.now();
                        const ttftText =
                            st.ttft != null ? formatMs(st.ttft) : "—",
                          totalText =
                            st.total != null ? formatMs(st.total) : "—";
                        return (
                          <Tr
                            key={row.id}
                            className={
                              [
                                detail?.id === row.id ? "bg-accent-subtle" : "",
                                fresh(row.id) ? FRESH_ROW : "",
                              ]
                                .filter(Boolean)
                                .join(" ") || undefined
                            }
                          >
                            <Td className="whitespace-nowrap tabular-nums max-sm:hidden">
                              {when == null ? (
                                "—"
                              ) : (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span tabIndex={0}>{clock(when)}</span>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    {t("requests.list.clockTitle", {
                                      relative: relativeTime(when, now),
                                      date: dateTime(when),
                                    })}
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </Td>
                            <Td
                              className="min-w-44"
                              data-testid="request-identity"
                            >
                              <div className="flex min-w-0 items-center gap-2">
                                {(row.phase !== "complete" ||
                                  (row.outcome &&
                                    row.outcome !== "completed")) && (
                                  <StatusIndicator
                                    className="shrink-0 gap-1.5 whitespace-nowrap font-sans text-xs text-muted-foreground"
                                    status={
                                      row.phase === "complete"
                                        ? outcomeDot(row.outcome)
                                        : phaseDot(row.phase)
                                    }
                                  >
                                    {row.phase === "complete"
                                      ? row.outcome
                                        ? outcomeLabel(row.outcome)
                                        : ""
                                      : phaseLabel(row.phase)}
                                  </StatusIndicator>
                                )}
                                <span
                                  title={row.id}
                                  className="min-w-0 flex-1 truncate font-mono text-xs"
                                >
                                  {shortId(row.id)}
                                </span>
                              </div>
                              <p
                                title={row.model ?? undefined}
                                className="mt-0.5 truncate text-xs text-muted-foreground"
                              >
                                {row.model
                                  ? modelLabel(row.model)
                                  : t("requests.list.modelUnknown")}
                              </p>
                              <p className="mt-0.5 truncate text-xs tabular-nums text-muted-foreground md:hidden">
                                {/* The time column is gone on phones; the time rides on this line. */}
                                {when != null && (
                                  <span className="sm:hidden">
                                    {clock(when)}
                                    {" · "}
                                  </span>
                                )}
                                {row.phase === "complete"
                                  ? t("requests.list.mobileLine", {
                                      ttft: ttftText,
                                      total: totalText,
                                      tps: number(speedValue),
                                    })
                                  : percent != null
                                    ? t("requests.list.prefillPercent", {
                                        percent: number(percent, 0),
                                      })
                                    : `${elapsed(row.elapsed_s)} · ${number(speedValue)} tok/s`}
                              </p>
                            </Td>
                            <Td className="hidden overflow-hidden tabular-nums lg:table-cell">
                              <span className="block truncate">
                                {number(row.prompt_tokens, 0)} /{" "}
                                {number(row.completion_tokens, 0)}
                              </span>
                              <span className="block truncate text-xs text-muted-foreground">
                                {t("requests.list.cachedOf", {
                                  cached: number(row.cached_tokens, 0),
                                  prompt: number(row.prompt_tokens, 0),
                                })}
                              </span>
                            </Td>
                            <Td
                              className={`hidden whitespace-nowrap tabular-nums md:table-cell ${slow ? "text-warning" : ""}`}
                              title={
                                slow ? t("requests.list.slowMark") : undefined
                              }
                            >
                              {percent != null ? (
                                <span>
                                  {t("requests.list.prefillPercent", {
                                    percent: number(percent, 0),
                                  })}
                                </span>
                              ) : row.phase === "complete" ? (
                                ttftText
                              ) : (
                                elapsed(row.elapsed_s)
                              )}
                            </Td>
                            <Td
                              className={`hidden whitespace-nowrap tabular-nums md:table-cell ${slow ? "text-warning" : ""}`}
                            >
                              {row.phase === "complete" ? totalText : "—"}
                            </Td>
                            <Td className="hidden whitespace-nowrap tabular-nums md:table-cell">
                              {number(speedValue)}
                            </Td>
                            <Td className="hidden min-[1800px]:table-cell">
                              {speculativeText(row) ?? (
                                <span className="text-muted-foreground">—</span>
                              )}
                            </Td>
                            <Td>
                              <div className="flex justify-end gap-1 whitespace-nowrap">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={(e) => {
                                    e.currentTarget.focus();
                                    detailOpener.current = e.currentTarget;
                                    setDetail(row);
                                  }}
                                >
                                  {t("requests.list.details")}
                                </Button>
                                {row.phase !== "complete" && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    disabled={!isOnline(engine) || !!busy}
                                    onClick={(e) => {
                                      opener.current = e.currentTarget;
                                      setCancel(row);
                                    }}
                                  >
                                    {t("requests.list.cancel")}
                                  </Button>
                                )}
                              </div>
                            </Td>
                          </Tr>
                        );
                      })}
                    </Tbody>
                  </Table>
                </TooltipProvider>
                {!shown.length && (
                  <TableState loading={!engine.status}>
                    {engine.status
                      ? query
                        ? t("requests.list.noMatch")
                        : filter === "active"
                          ? t("requests.list.noActive")
                          : t("requests.list.noneFiltered")
                      : t("requests.list.waiting")}
                  </TableState>
                )}
              </div>
              {matched.length > shown.length && (
                <div className="flex justify-center py-2">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setLimit((n) => n + PAGE)}
                  >
                    {t("requests.list.showMore", {
                      count: matched.length - shown.length,
                    })}
                  </Button>
                </div>
              )}
              <div
                className="min-h-[3.25rem] space-y-1 bg-muted/30 px-4 py-2.5 text-xs text-muted-foreground"
                data-testid="requests-footer"
              >
                <p className="tabular-nums">
                  {t("requests.footer.counts", {
                    shown: shown.length,
                    matched: matched.length,
                    total: rows.length,
                  })}
                  {recent.supported && finished.length > 0
                    ? ` · ${t("requests.footer.since", {
                        time: clock(retention.since * 1000).slice(0, 5),
                        count: number(finished.length, 0),
                        prompt: number(promptSum, 0),
                        cached: number(cachedSum, 0),
                      })}`
                    : ""}
                </p>
                {recent.supported === false ? (
                  <ErrorNote
                    tone="muted"
                    className="min-w-0 max-w-3xl"
                    message={t("requests.footer.unsupported")}
                    detail={t("requests.footer.unsupportedDetail")}
                  />
                ) : (
                  <p className="min-w-0 max-w-3xl">
                    {recent.supported
                      ? t("requests.footer.ring", {
                          capacity: number(recent.capacity, 0),
                        })
                      : t("requests.footer.loading")}
                  </p>
                )}
                {recent.supported && (
                  <ArchiveControls
                    archive={archive}
                    capacity={recent.capacity}
                    shown={matched}
                    baseUrl={connection.baseUrl}
                  />
                )}
                {speed !== "all" && (
                  <p data-testid="slow-rule">
                    {t(
                      rule.basis === "p90"
                        ? "requests.list.slowRuleP90"
                        : "requests.list.slowRuleFixed",
                      {
                        ttft: formatMs(rule.ttftMs),
                        total: formatMs(rule.totalMs),
                      },
                    )}
                  </p>
                )}
                {recent.error && (
                  <p role="status" className="text-warning">
                    {t("requests.footer.error", { error: recent.error })}
                  </p>
                )}
              </div>
            </Card>
          </div>
        }
        detail={
          xl ? (
            <Card className="sticky top-0 max-h-[calc(100dvh-8rem)] overflow-y-auto p-4">
              <GroupLabel
                className="px-0"
                title={t("requests.detail.title")}
                action={
                  detail ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setDetail(null);
                        detailOpener.current?.focus();
                      }}
                    >
                      {t("requests.detail.close")}
                    </Button>
                  ) : undefined
                }
              />
              {detailBody ?? (
                <EmptyState
                  size="inline"
                  title={t("requests.detail.emptyTitle")}
                  description={t("requests.detail.emptyBody")}
                />
              )}
            </Card>
          ) : null
        }
      />
      {!xl && (
        <Sheet
          open={!!detail}
          onClose={() => setDetail(null)}
          title={t("requests.detail.title")}
          closeLabel={t("requests.detail.closeSheet")}
        >
          {detailBody}
        </Sheet>
      )}
      <Dialog
        open={!!cancel}
        onOpenChange={(open) => {
          if (!open) setCancel(null);
        }}
      >
        <DialogContent
          closeLabel={t("requests.cancel.closeLabel")}
          onCloseAutoFocus={restore}
        >
          <DialogTitle>{t("requests.cancel.title")}</DialogTitle>
          <DialogDescription>
            {t("requests.cancel.description", { id: cancel?.id ?? "" })}
          </DialogDescription>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setCancel(null)}>
              {t("requests.cancel.keep")}
            </Button>
            <Button
              onClick={() => {
                const r = cancel;
                setCancel(null);
                if (r)
                  void perform(`cancel:${r.id}`, () =>
                    cancelRequest(connection, r.id),
                  );
              }}
            >
              {t("requests.cancel.confirm")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </DashboardPage>
  );
}
