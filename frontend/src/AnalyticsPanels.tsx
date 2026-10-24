import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Card, EmptyState } from "@yuhuanowo/yunui";
import type {
  ColumnChartProps,
  MetricChartProps,
  SparklineTipProps,
} from "@yuhuanowo/yunui/charts";
import { Sparkline } from "@yuhuanowo/yunui";
import { t, useLocale } from "./i18n/index.ts";
import { initialAxis, nextAxis } from "./motion/axis-core";
import { reducedMotion } from "./motion/ticker";

// The chart code (YunUI MetricChart, ColumnChart, SparklineTip) is its own chunk: no route's first load
// carries it, and a chart reserves its final height while the chunk arrives.
const ConsoleChart = lazy(() => import("./charts/ConsoleChart"));
const ConsoleColumns = lazy(() => import("./charts/ConsoleColumns"));
const ConsoleSparkTip = lazy(() => import("./charts/SparkTip"));

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

/** What the chart reserves for its legend / toolbar row (24 px + 6 px gap), so a chart never changes height when the chunk lands. */
const HEAD_PX = 30;
/** The time axis and its padding under the plot: the idle placeholder reserves them so the chart replaces it without a jump. */
const CHART_AXIS_PX = 30;

type ChartProps = Omit<
  MetricChartProps,
  "labels" | "height" | "yMax" | "live" | "xDomain"
>;

export function SeriesChart({
  busy,
  height = 180,
  className,
  liveWindowMs,
  idleLabel,
  refreshing = false,
  ...rest
}: ChartProps & {
  busy: boolean;
  /** Plot height; the legend/toolbar row and the time axis come on top of it. */
  height?: number;
  /** What the placeholder says when no sample has a value (defaults to the page's idle wording). */
  idleLabel?: string;
  /** Earlier data is still on screen while a new range loads: it stays, dimmed (no skeleton, no jump). */
  refreshing?: boolean;
  /**
   * Makes the chart a live one: the x axis is the last `liveWindowMs` up to the newest sample (so a
   * point's position depends on its own time only and the series slides with the clock between
   * samples), and the y axis holds a nice maximum with hysteresis.
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
  // No sample has a value yet: say so calmly, at the chart's own height, instead of an empty plot.
  const hasValues = rest.data.some((p) =>
    rest.series.some((d) => {
      const v = p.values[d.key];
      return typeof v === "number" && Number.isFinite(v) && v !== 0;
    }),
  );
  const head =
    rest.series.length > 1 || rest.tools?.smooth || rest.tools?.log
      ? HEAD_PX
      : 0;
  const frame = height + head + CHART_AXIS_PX;
  if (rest.data.length > 0 && !hasValues)
    return (
      <div
        role="status"
        aria-label={rest.ariaLabel}
        className={`flex items-center justify-center ${className ?? ""}`}
        style={{ height: frame }}
        data-testid="series-idle"
      >
        <EmptyState
          size="inline"
          title={
            busy
              ? t("overview.chart.collecting")
              : (idleLabel ?? t("overview.chart.idle"))
          }
        />
      </div>
    );
  return (
    <div
      className={`transition-opacity duration-(--dur-base) ${refreshing ? "opacity-60" : ""} ${className ?? ""}`}
      style={{ minHeight: frame }}
    >
      <Suspense fallback={<div style={{ height: frame }} aria-hidden />}>
        <ConsoleChart
          {...rest}
          height={height + CHART_AXIS_PX}
          minSamples={rest.minSamples ?? 5}
          yMax={liveWindowMs ? yMax : undefined}
          live={!!liveWindowMs}
          xDomain={liveWindowMs ? [newest - liveWindowMs, newest] : undefined}
        />
      </Suspense>
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

/** Thin columns (counts per bucket or day) with a tooltip; holds its height while the chart chunk loads. */
export function Columns(props: ColumnChartProps) {
  return (
    <Suspense
      fallback={<div style={{ height: props.height ?? 150 }} aria-hidden />}
    >
      <ConsoleColumns {...props} />
    </Suspense>
  );
}

/** A sparkline that reads out the sample under the pointer; a plain sparkline until the chart chunk arrives. */
export function SparkTip(props: SparklineTipProps) {
  const { readout, ...plain } = props;
  void readout;
  return (
    <Suspense fallback={<Sparkline {...plain} />}>
      <ConsoleSparkTip {...props} />
    </Suspense>
  );
}
