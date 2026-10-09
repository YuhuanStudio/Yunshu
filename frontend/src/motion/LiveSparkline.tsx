import { useId, useLayoutEffect, useMemo, useRef } from "react";
import { sparkGeometry, slide, type TimedPoint } from "./spark-core.ts";
import { reducedMotion, ticker } from "./ticker.ts";

const STROKE: Record<string, string> = {
  neutral: "var(--color-muted-foreground)",
  accent: "var(--color-accent, var(--text-primary))",
  success: "var(--success)",
  info: "var(--info)",
  warning: "var(--warning)",
  error: "var(--error)",
};

/**
 * A sparkline whose time axis is wall-clock: between samples the whole line slides left by a
 * transform (one ticker, no React render per frame), a new sample extends it on the right and its
 * newest segment fades in for 200 ms. The line is rebuilt only when a sample arrives. Under
 * reduced motion it neither slides nor fades.
 */
export function LiveSparkline({
  points,
  windowMs = 60_000,
  width = 160,
  height = 32,
  tone = "neutral",
  area = false,
  max,
  label,
  className,
}: {
  points: readonly TimedPoint[];
  windowMs?: number;
  width?: number;
  height?: number;
  tone?: string;
  area?: boolean;
  max?: number;
  label: string;
  className?: string;
}) {
  const gid = useId();
  const g = useRef<SVGGElement>(null);
  const newest = points.length ? points[points.length - 1].t : 0;
  const geo = useMemo(
    () => sparkGeometry(points, windowMs, width, height, max),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [newest, points.length, windowMs, width, height, max],
  );
  const pxPerMs = geo?.pxPerMs ?? 0;

  useLayoutEffect(() => {
    if (!geo) return;
    const place = (shift: number) => {
      if (g.current)
        g.current.style.transform = `translateX(${(width + shift).toFixed(2)}px)`;
    };
    place(0);
    if (reducedMotion()) return;
    // the newest sample is `age` old when it renders; the slide continues from there
    const born = performance.now() - Math.max(0, Date.now() - newest);
    return ticker.subscribe((now) => place(slide(now - born, pxPerMs)));
  }, [newest, geo, width, pxPerMs]);

  if (!geo) return null;
  const color = STROKE[tone] ?? STROKE.neutral;
  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={className}
      data-yunui="live-sparkline"
    >
      {area && (
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={color} stopOpacity="0.28" />
            <stop offset="1" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
      )}
      <g ref={g}>
        {area && <path d={geo.area} fill={`url(#${gid})`} stroke="none" />}
        <path
          d={geo.line}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
        <path
          key={newest}
          d={geo.tail}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
          className="live-point-in"
        />
      </g>
    </svg>
  );
}
