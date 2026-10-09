import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Card, EmptyState, TimeSeriesChart } from "@yuhuanowo/yunui";
import type { ComponentProps, CSSProperties } from "react";
import { t, useLocale } from "./i18n/index.ts";
import { clock, number } from "./ui";
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

/** The time axis and its padding under the plot: the idle placeholder reserves them so the chart replaces it without a jump. */
const CHART_AXIS_PX = 36;

export function SeriesChart({
  busy,
  height = 180,
  className,
  liveWindowMs,
  idleLabel,
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
  /** What the placeholder says when no sample has a value (defaults to the page's idle wording). */
  idleLabel?: string;
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
        // A label that would touch either edge is nudged inward by the chart (a jump of its own): it
        // appears once it is clear of the right edge and goes before it reaches the left one.
        xTicks: timeTicks(newest - liveWindowMs, newest, 3).filter(
          (t) =>
            t <= newest - liveWindowMs * 0.1 &&
            t >= newest - liveWindowMs * 0.9,
        ),
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
        className={`flex items-center justify-center ${className ?? ""}`}
        style={{ height: height + CHART_AXIS_PX }}
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
