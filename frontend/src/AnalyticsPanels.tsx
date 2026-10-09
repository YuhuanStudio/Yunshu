import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  BarChart,
  Button,
  Card,
  DonutChart,
  EmptyState,
  Heatmap,
  Sheet,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  TimeSeriesChart,
  Tr,
} from "@yuhuanowo/yunui";
import type { ComponentProps, CSSProperties } from "react";
import { ArrowRight } from "lucide-react";
import {
  MIN_PERCENTILE_SAMPLES,
  activityHeatmap,
  latencyDistribution,
  percentileWhenEnough,
  phaseDistribution,
  type LatencyBucket,
  type ObservedRequest,
} from "./analytics";
import { t, useLocale } from "./i18n/index.ts";
import type { SeriesPoint } from "./series";
import { clock, elapsed, number, Slot, type Engine } from "./ui";
import { initialAxis, nextAxis, timeTicks } from "./motion/axis-core";
import { LiveScroll } from "./motion/LiveScroll";
import { reducedMotion } from "./motion/ticker";

type ChartProps = ComponentProps<typeof TimeSeriesChart>;

/**
 * The console's time-series chart. The library draws a dashed box (min 180px)
 * when the visible series have no finite value yet, which is exactly what a
 * long prefill looks like. Here that state becomes one calm muted caption at
 * the chart's own height: no border, nothing shifting. "Collecting" while the
 * engine is busy (values are on their way), "idle" when nothing is running.
 */
/**
 * A number that follows `target` with an ease-out over `ms`, re-rendering only while it moves (the
 * y-axis of a live chart changing its maximum: a few frames, not one per sample).
 */
function useGlide(target: number, ms = 280): number {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    if (value === target) return;
    if (reducedMotion()) {
      from.current = target;
      setValue(target);
      return;
    }
    const start = performance.now();
    const begin = from.current;
    let id = 0;
    const step = (now: number) => {
      const k = Math.min(1, (now - start) / ms);
      const v = begin + (target - begin) * (1 - Math.pow(1 - k, 3));
      from.current = v;
      setValue(v);
      if (k < 1) id = requestAnimationFrame(step);
    };
    id = requestAnimationFrame(step);
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, ms]);
  return value;
}

export function SeriesChart({
  busy,
  height = 180,
  className,
  liveWindowMs,
  ...rest
}: Omit<
  ChartProps,
  | "emptyLabel"
  | "unavailableLabel"
  | "missingValueLabel"
  | "hiddenLabel"
  | "legendLabel"
  | "keyboardHint"
  | "collectingLabel"
  | "minSamples"
  | "height"
> & {
  busy: boolean;
  height?: number;
  /**
   * Makes the chart a live one: the x axis is the last `liveWindowMs` up to the newest sample (so a
   * point's position depends on its own time only and the series slides with the clock between
   * samples), the y axis holds a nice maximum with hysteresis, and ticks are absolute round times.
   */
  liveWindowMs?: number;
}) {
  useLocale();
  const newest = rest.data.length ? rest.data[rest.data.length - 1].x : 0;
  const dataMax = useMemo(() => {
    let m = 0;
    for (const p of rest.data)
      for (const d of rest.series) {
        const v = p.values[d.key];
        if (typeof v === "number" && Number.isFinite(v) && v > m) m = v;
      }
    return m;
  }, [rest.data, rest.series]);
  const axis = useRef(initialAxis(dataMax));
  axis.current = nextAxis(axis.current, dataMax, Date.now());
  const yMax = useGlide(axis.current.max);
  const live = liveWindowMs
    ? {
        xDomain: [newest - liveWindowMs, newest] as const,
        xTicks: timeTicks(newest - liveWindowMs, newest, 4),
        yMax,
      }
    : null;
  // No sample has a value yet: say so calmly, at a reduced height, instead of an empty plot with a readout row.
  const hasValues = rest.data.some((p) =>
    rest.series.some((d) => {
      const v = p.values[d.key];
      return typeof v === "number" && Number.isFinite(v) && v !== 0;
    }),
  );
  if (rest.data.length > 0 && !hasValues)
    return (
      <div
        role="status"
        aria-label={rest.ariaLabel}
        className={className}
        data-testid="series-idle"
      >
        <EmptyState
          size="inline"
          title={
            busy ? t("overview.chart.collecting") : t("overview.chart.idle")
          }
        />
      </div>
    );
  return (
    <div
      className={`[&_div[role=status]]:h-(--series-h) [&_div[role=status]]:min-h-0 [&_div[role=status]]:rounded-none [&_div[role=status]]:border-0 [&_div[role=status]]:text-xs ${className ?? ""}`}
      style={{ "--series-h": `${height}px` } as CSSProperties}
    >
      {live ? (
        <LiveScroll spanMs={liveWindowMs!} newestAt={newest}>
          <TimeSeriesChart
            {...rest}
            {...(live ?? {})}
            height={height}
            minSamples={5}
            showReadout={false}
            emptyLabel={t("overview.chart.empty")}
            unavailableLabel={
              busy ? t("overview.chart.collecting") : t("overview.chart.idle")
            }
            missingValueLabel={t("overview.chart.missing")}
            hiddenLabel={t("overview.chart.hidden")}
            legendLabel={t("overview.chart.legend")}
            keyboardHint={t("overview.chart.keyboardHint")}
            collectingLabel={(have, need) =>
              t("overview.chart.collectingSamples", { have, need })
            }
          />
        </LiveScroll>
      ) : (
        <TimeSeriesChart
          {...rest}
          {...(live ?? {})}
          height={height}
          minSamples={5}
          showReadout={false}
          emptyLabel={t("overview.chart.empty")}
          unavailableLabel={
            busy ? t("overview.chart.collecting") : t("overview.chart.idle")
          }
          missingValueLabel={t("overview.chart.missing")}
          hiddenLabel={t("overview.chart.hidden")}
          legendLabel={t("overview.chart.legend")}
          keyboardHint={t("overview.chart.keyboardHint")}
          collectingLabel={(have, need) =>
            t("overview.chart.collectingSamples", { have, need })
          }
        />
      )}
    </div>
  );
}

/** The one header every chart card shares: title and caption left, one control right. */
export function ChartCard({
  title,
  description,
  action,
  children,
  className,
  "data-testid": testId,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  "data-testid"?: string;
}) {
  return (
    <Card className={"min-w-0 p-4 " + (className ?? "")} data-testid={testId}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="heading-md">{title}</h2>
          {description && (
            <p className="mt-1 text-xs text-muted-foreground">{description}</p>
          )}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      {children}
    </Card>
  );
}

export function PhasePanel({
  engine,
  navigate,
}: {
  engine: Engine;
  navigate: (page: string) => void;
}) {
  useLocale();
  const [phase, setPhase] = useState<string | null>(null);
  const data = phaseDistribution(engine.status);
  const active = engine.status?.requests.items ?? [];
  const selected = phase ? active.filter((row) => row.phase === phase) : active;
  return (
    <ChartCard
      data-testid="phase-panel"
      title={t("overview.phasePanel.title")}
      description={t("overview.phasePanel.desc")}
      action={
        <span className="text-xs text-muted-foreground">
          {t("overview.phasePanel.live")}
        </span>
      }
    >
      <DonutChart
        monochrome
        data={data}
        size={154}
        ariaLabel={t("overview.phasePanel.aria")}
        emptyLabel={
          engine.status
            ? t("overview.phasePanel.empty")
            : t("overview.phasePanel.noStatus")
        }
        unavailableLabel={t("overview.chart.missing")}
        center={
          <div>
            <strong className="block text-2xl font-semibold tabular-nums">
              {number(engine.status?.requests.active, 0)}
            </strong>
            <span className="text-xs text-muted-foreground">
              {t("overview.phasePanel.center")}
            </span>
          </div>
        }
        onSelect={(datum) =>
          setPhase((value) => (value === datum.id ? null : datum.id))
        }
      />
      {phase && (
        <Button
          size="sm"
          variant="ghost"
          className="mt-3"
          onClick={() => setPhase(null)}
        >
          {t("overview.phasePanel.clear")}
        </Button>
      )}
      <div className="mt-4 min-h-[7.5rem] divide-y divide-border/60">
        {selected.slice(0, 3).map((row) => (
          <div
            key={row.request_id}
            className="flex justify-between gap-3 py-2.5 text-xs"
          >
            <span className="min-w-0 truncate font-mono">{row.request_id}</span>
            <Slot
              ch={7}
              align="right"
              className="shrink-0 text-muted-foreground"
            >
              {elapsed(row.elapsed_s)}
            </Slot>
          </div>
        ))}
        {!selected.length && (
          <p className="py-3 text-xs text-muted-foreground">
            {engine.status
              ? t("overview.phasePanel.noMatch")
              : t("overview.phasePanel.connectFirst")}
          </p>
        )}
      </div>
      <Button
        variant="ghost"
        size="sm"
        className="mt-2"
        onClick={() => navigate("requests")}
      >
        {t("overview.phasePanel.open")}
        <ArrowRight size={13} />
      </Button>
    </ChartCard>
  );
}

export function LatencyPanel({
  records,
}: {
  records: readonly ObservedRequest[];
}) {
  useLocale();
  const bins = useMemo(() => latencyDistribution(records), [records]);
  const [selection, setSelection] = useState<LatencyBucket | null>(null);
  const valid = records.filter(
    (row) =>
      row.ttft_ms != null && Number.isFinite(row.ttft_ms) && row.ttft_ms >= 0,
  );
  const ttfts = valid.map((row) => row.ttft_ms);
  // Percentiles from a handful of samples are noise: below the minimum, only
  // the latest request is shown, labelled as such.
  const p50 = percentileWhenEnough(ttfts, 0.5);
  const p95 = percentileWhenEnough(ttfts, 0.95);
  const latest = valid.length ? valid[valid.length - 1].ttft_ms : null;
  const selected = selection
    ? valid.filter(
        (row) => row.ttft_ms! >= selection.min && row.ttft_ms! < selection.max,
      )
    : [];
  return (
    <ChartCard
      data-testid="latency-panel"
      title={t("overview.latency.title")}
      description={t("overview.latency.desc")}
      action={
        <span className="text-xs tabular-nums text-muted-foreground">
          {t("overview.latency.observed", { n: valid.length })}
        </span>
      }
    >
      <div className="mb-4 flex gap-6" data-testid="latency-figures">
        {[
          [t("overview.latency.latest"), latest, "latency-last"],
          ["P50", p50, "latency-p50"],
          ["P95", p95, "latency-p95"],
        ].map(([label, value, id]) => (
          <div key={id as string} data-testid={id as string}>
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums">
              {number(value as number | null, 0)}{" "}
              <span className="text-xs font-normal">ms</span>
            </p>
          </div>
        ))}
      </div>
      {valid.length < MIN_PERCENTILE_SAMPLES && (
        <p className="-mt-2 mb-3 text-xs text-muted-foreground">
          {t("overview.latency.minNote", {
            n: valid.length,
            min: MIN_PERCENTILE_SAMPLES,
          })}
        </p>
      )}
      <BarChart
        data={bins.map((bin) => ({ ...bin, tone: "neutral" as const }))}
        height={185}
        ariaLabel={t("overview.latency.aria")}
        emptyLabel={t("overview.latency.empty")}
        unavailableLabel={t("overview.chart.missing")}
        formatValue={(v) => t("overview.latency.count", { n: v })}
        onSelect={(datum) =>
          setSelection(bins.find((bin) => bin.id === datum.id) ?? null)
        }
      />
      <p className="mt-4 text-xs leading-5 text-muted-foreground">
        {t("overview.latency.note")}
      </p>
      <Sheet
        open={!!selection}
        onClose={() => setSelection(null)}
        title={t("overview.latency.sheetTitle", {
          label: selection?.label ?? "",
        })}
        closeLabel={t("overview.latency.close")}
      >
        <p className="mb-4 text-xs text-muted-foreground">
          {t("overview.latency.sheetCount", { n: selected.length })}
        </p>
        {selected.length ? (
          <Table scrollLabel={t("overview.latency.tableLabel")}>
            <Thead>
              <Tr>
                <Th>{t("overview.latency.colRequest")}</Th>
                <Th>TTFT</Th>
                <Th>{t("overview.latency.colCached")}</Th>
              </Tr>
            </Thead>
            <Tbody>
              {selected.map((row) => (
                <Tr key={row.request_id}>
                  <Td>
                    <span className="block max-w-36 truncate font-mono text-xs">
                      {row.request_id}
                    </span>
                  </Td>
                  <Td>{number(row.ttft_ms, 0)} ms</Td>
                  <Td>{number(row.cached_tokens, 0)}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        ) : (
          <EmptyState
            size="inline"
            title={t("overview.latency.emptyTitle")}
            description={t("overview.latency.emptyDesc")}
          />
        )}
      </Sheet>
    </ChartCard>
  );
}

export function ActivityPanel({
  history,
  start,
  end,
  onSelectTime,
}: {
  history: readonly SeriesPoint[];
  start: number;
  end: number;
  onSelectTime: (at: number | null) => void;
}) {
  const locale = useLocale();
  const heat = useMemo(
    () => activityHeatmap(history, start, end),
    // locale: the row names are translated inside.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [history, start, end, locale],
  );
  const [selection, setSelection] = useState<{
    row: number;
    column: number;
  } | null>(null);
  const columns = heat.starts.map(clock);
  const select = (row: number, column: number) => {
    setSelection({ row, column });
    const candidates = history.filter(
      (sample) =>
        sample.at >= heat.starts[column] &&
        sample.at <=
          (column === heat.starts.length - 1
            ? heat.ends[column]
            : heat.ends[column] - 1),
    );
    const value = (sample: SeriesPoint) =>
      [
        sample.active,
        sample.queued,
        sample.prefillRequests,
        sample.decodeRequests,
      ][row] ?? -1;
    const peak = candidates.reduce<SeriesPoint | null>(
      (best, next) => (!best || value(next) > value(best) ? next : best),
      null,
    );
    onSelectTime(peak?.at ?? null);
  };
  return (
    <ChartCard
      data-testid="activity-panel"
      title={t("overview.heat.title")}
      description={t("overview.heat.desc")}
      action={
        <span className="text-xs text-muted-foreground">
          <Slot ch={10} align="right">
            {t("overview.heat.samples", { n: history.length })}
          </Slot>
        </span>
      }
    >
      <Heatmap
        rows={heat.rows}
        columns={columns}
        data={heat.data}
        ariaLabel={t("overview.heat.aria")}
        unavailableLabel="—"
        emptyLabel={t("overview.heat.empty")}
        tone="neutral"
        formatValue={(v) => String(v)}
        onSelect={select}
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <p>{t("overview.heat.legend")}</p>
        {selection && (
          <p role="status">
            {t("overview.heat.selection", {
              from: clock(heat.starts[selection.column]),
              to: clock(heat.ends[selection.column]),
              n: heat.coverage[selection.column],
              peak: number(heat.data[selection.row][selection.column], 0),
            })}
          </p>
        )}
      </div>
    </ChartCard>
  );
}
