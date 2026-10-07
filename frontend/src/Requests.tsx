import { useEffect, useMemo, useRef, useState } from "react";
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
import { Download, Search } from "lucide-react";
import { ApiError, cancelRequest, requestJson, type Connection } from "./api";
import {
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
} from "./ui";
import type { Perform } from "./Models";
import { outcomeLabel, useRecentRequests } from "./recentRequests";
import { RequestBreakdown, RequestTimeline } from "./RequestTimeline";

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
  const [filter, setFilter] = useState("active"),
    [outcome, setOutcome] = useState("all"),
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
  const recent = useRecentRequests(connection, engine.status?.last?.request_id);
  const sampled = useMemo(
    () =>
      finishedFromHistory(engine.history).map(
        (row) => ({ ...row, source: "sampled" as const }) satisfies Row,
      ),
    [engine.history],
  );
  // The server ring when it exists; otherwise only what this page sampled from status polls.
  const finished = useMemo(
    () => (recent.supported ? [...recent.rows].reverse() : sampled),
    [recent.supported, recent.rows, sampled],
  );
  useEffect(() => setLimit(PAGE), [filter, outcome, query]);
  const rows = useMemo(() => {
    const observed = new Map<string, Row>();
    for (const row of finished) observed.set(row.id, row);
    for (const row of engine.status?.requests.items ?? [])
      observed.set(row.request_id, { ...row, id: row.request_id } as Row);
    return [...observed.values()].reverse();
  }, [finished, engine.status]);
  const matched = rows.filter(
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
  const shown = matched.slice(0, limit);
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
      value: ttfts.length ? number(median(ttfts), 0) : "—",
      unit: ttfts.length ? "ms" : undefined,
      hint: ttfts.length
        ? t(
            recent.supported
              ? "requests.tile.ttftServer"
              : "requests.tile.ttftPage",
            {
              count: number(ttfts.length, 0),
              last: number(ttfts.at(-1), 0),
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
      finished.find((r) => r.id === detail.id && r.source === "ring")) ||
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
        <section
          className="space-y-4"
          aria-label={t("requests.detail.breakdownLabel")}
        >
          <RequestBreakdown row={view} />
          <RequestTimeline row={view} />
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
        <div className="grid grid-cols-3 gap-5 border-t border-border/60 pt-5">
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
      <StatGrid data-testid="request-stats">
        {tiles.map((tile) => (
          <StatCard
            key={tile.label}
            compact
            valueFirst
            icon={tile.icon}
            label={tile.label}
            value={
              tile.unit && tile.value !== "—"
                ? `${tile.value} ${tile.unit}`
                : tile.value
            }
            trend={tile.trend ?? undefined}
            subtext={
              <span className="block min-w-0 space-y-2">
                <span className="block truncate">{tile.hint}</span>
                <span className="block h-7">
                  {tile.data.length > 1 && (
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
        ))}
      </StatGrid>
      <WorkspaceLayout
        detailLabel={t("requests.detail.title")}
        list={
          <div className="space-y-5">
            <div className="flex flex-wrap justify-between gap-3">
              <Input
                className="sm:max-w-xs"
                aria-label={t("requests.list.search")}
                icon={<Search size={14} />}
                placeholder={t("requests.list.searchPlaceholder")}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <div className="flex flex-wrap items-center gap-2">
                <SegmentedTray
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
                {filter !== "active" && (
                  <CustomSelect
                    className="w-36 [&_button]:h-8 [&_button]:text-xs"
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
                )}
              </div>
            </div>
            <Card className="overflow-hidden">
              <div className="min-h-[17rem]">
                <TooltipProvider delayDuration={200}>
                  <Table
                    scrollLabel={t("requests.list.tableLabel")}
                    className="table-fixed"
                  >
                    <Thead>
                      <Tr>
                        <Th>{t("requests.list.colRequest")}</Th>
                        <Th className="hidden w-48 2xl:table-cell">
                          {t("requests.list.colModel")}
                        </Th>
                        <Th className="w-44">{t("requests.list.colUsage")}</Th>
                        <Th className="w-20">tok/s</Th>
                        <Th className="hidden w-32 md:table-cell">
                          {t("requests.list.colProgress")}
                        </Th>
                        <Th className="hidden w-24 min-[1800px]:table-cell">
                          {t("requests.list.colSpec")}
                        </Th>
                        <Th className="hidden w-24 2xl:table-cell">
                          {t("requests.list.colTime")}
                        </Th>
                        <Th className="w-36">
                          {t("requests.list.colActions")}
                        </Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {shown.map((row) => {
                        const percent =
                            row.phase === "prefill"
                              ? prefillPercent(row)
                              : null,
                          speed =
                            row.phase === "complete"
                              ? row.decode_tps
                              : row.tokens_per_second;
                        const when = row.t ? row.t * 1000 : null,
                          now = engine.updatedAt ?? Date.now();
                        return (
                          <Tr
                            key={row.id}
                            className={
                              detail?.id === row.id
                                ? "bg-accent-subtle"
                                : undefined
                            }
                          >
                            <Td>
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
                                  className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground"
                                >
                                  {shortId(row.id)}
                                </span>
                              </div>
                              <p
                                title={row.model ?? undefined}
                                className="mt-1 truncate text-xs text-muted-foreground 2xl:hidden"
                              >
                                {row.model
                                  ? modelLabel(row.model)
                                  : t("requests.list.modelUnknown")}
                              </p>
                            </Td>
                            <Td className="hidden 2xl:table-cell">
                              <span
                                title={row.model ?? undefined}
                                className="block truncate text-xs text-muted-foreground"
                              >
                                {row.model ? (
                                  modelLabel(row.model)
                                ) : (
                                  <span className="text-muted-foreground">
                                    {t("requests.detail.notReported")}
                                  </span>
                                )}
                              </span>
                            </Td>
                            <Td className="overflow-hidden tabular-nums">
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
                            <Td className="whitespace-nowrap tabular-nums">
                              {number(speed)}
                            </Td>
                            <Td className="hidden tabular-nums md:table-cell">
                              {percent != null ? (
                                <span>
                                  {t("requests.list.prefillPercent", {
                                    percent: number(percent, 0),
                                  })}
                                  <span className="ml-2 text-muted-foreground">
                                    {elapsed(row.elapsed_s)}
                                  </span>
                                </span>
                              ) : row.phase === "complete" ? (
                                t("requests.list.ttftMs", {
                                  ms: number(row.ttft_ms, 0),
                                })
                              ) : (
                                elapsed(row.elapsed_s)
                              )}
                            </Td>
                            <Td className="hidden min-[1800px]:table-cell">
                              {speculativeText(row) ?? (
                                <span className="text-muted-foreground">—</span>
                              )}
                            </Td>
                            <Td className="hidden whitespace-nowrap text-muted-foreground 2xl:table-cell">
                              {when == null ? (
                                "—"
                              ) : (
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <span tabIndex={0}>
                                      {relativeTime(when, now)}
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    {dateTime(when)}
                                  </TooltipContent>
                                </Tooltip>
                              )}
                            </Td>
                            <Td>
                              <div className="flex gap-1 whitespace-nowrap">
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
                <div className="flex justify-center border-t border-border/60 py-2">
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
                className="min-h-[3.25rem] space-y-1 border-t border-border/60 bg-muted/30 px-4 py-2.5 text-xs text-muted-foreground"
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
                <p className="min-w-0 max-w-3xl">
                  {recent.supported
                    ? t("requests.footer.ring", {
                        capacity: number(recent.capacity, 0),
                      })
                    : recent.supported === false
                      ? t("requests.footer.unsupported")
                      : t("requests.footer.loading")}
                </p>
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
