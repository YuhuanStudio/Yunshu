import { useLayoutEffect, useRef } from "react";
import { BarModel } from "./bar-core.ts";
import { reducedMotion, ticker } from "./ticker.ts";

const TONE: Record<string, string> = {
  info: "var(--info)",
  success: "var(--success)",
  warning: "var(--warning)",
  error: "var(--error)",
  neutral: "var(--color-muted-foreground)",
};

/**
 * A thin progress bar whose last segment moves continuously. `fixed` segments (a cache hit) are
 * drawn as given; the live segment follows `live`, a fraction of the whole, with the motion model
 * in bar-core (it extrapolates from the observed rate between real samples, never runs backwards,
 * and is capped). Everything moves by `transform` (translateX/scaleX), so frames stay on the
 * compositor and cost no layout and no React render. Reduced motion draws each sample as it is.
 * `resetKey` (the request id) lowers the bar at once when it changes.
 */
export function LiveBar({
  fixed = [],
  live,
  tone = "neutral",
  resetKey,
  label,
  liveLabel,
  valueText,
  height = 4,
  className,
}: {
  fixed?: { fraction: number; tone: string; label?: string }[];
  live: number;
  tone?: string;
  resetKey?: string | number | null;
  label: string;
  /** What the moving segment is ("computed so far"); shown in the tooltip. */
  liveLabel?: string;
  valueText?: string;
  height?: number;
  className?: string;
}) {
  const fill = useRef<HTMLSpanElement>(null);
  const model = useRef<BarModel | null>(null);
  const key = useRef(resetKey);
  const off = useRef<(() => void) | null>(null);
  const offset = Math.min(
    1,
    fixed.reduce((a, f) => a + Math.max(0, f.fraction), 0),
  );
  const room = Math.max(0, 1 - offset);
  const target = Math.min(room, Math.max(0, live));

  useLayoutEffect(() => {
    const paint = (frac: number) => {
      if (fill.current)
        fill.current.style.transform = `translateX(${offset * 100}%) scaleX(${frac})`;
    };
    const stop = () => {
      off.current?.();
      off.current = null;
    };
    const now = performance.now();
    if (!model.current) model.current = new BarModel(target, now);
    const m = model.current;
    if (reducedMotion() || key.current !== resetKey) {
      key.current = resetKey;
      stop();
      m.reset(target, now);
      paint(target);
      return;
    }
    m.push(target, now);
    paint(m.frame(now));
    if (!off.current)
      off.current = ticker.subscribe((t) => {
        paint(Math.min(room, m.frame(t)));
        if (m.settled) stop();
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, offset, resetKey]);
  useLayoutEffect(
    () => () => {
      off.current?.();
      off.current = null;
    },
    [],
  );

  const filled = Math.round((offset + target) * 100);
  const tip = fixed
    .filter((f) => f.label && f.fraction > 0)
    .map((f) => f.label)
    .concat(liveLabel && target > 0 ? [liveLabel] : [])
    .join(" · ");
  let at = 0;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={filled}
      aria-valuetext={valueText}
      title={tip || valueText}
      data-yunui="live-bar"
      className={`relative w-full overflow-hidden rounded-full bg-(--bg-elevated) ${className ?? ""}`}
      style={{ height }}
    >
      {fixed.map((f, i) => {
        const w = Math.min(1 - at, Math.max(0, f.fraction));
        const left = at;
        at += w;
        return w > 0 ? (
          <span
            key={i}
            data-tone={f.tone}
            className="absolute inset-y-0 left-0 w-full origin-left"
            style={{
              backgroundColor: TONE[f.tone] ?? TONE.neutral,
              transform: `translateX(${left * 100}%) scaleX(${w})`,
            }}
          />
        ) : null;
      })}
      <span
        ref={fill}
        data-tone={tone}
        className="absolute inset-y-0 left-0 w-full origin-left will-change-transform"
        style={{
          backgroundColor: TONE[tone] ?? TONE.neutral,
          transform: "scaleX(0)",
        }}
      />
    </div>
  );
}
