import { useLayoutEffect, useRef, type ReactNode } from "react";
import { reducedMotion, ticker } from "./ticker.ts";

/**
 * Makes a stepwise time-series chart scroll continuously. The chart draws its x axis from the
 * first to the newest sample, so every new sample slides everything left by one step at once.
 * Between samples this wrapper slides the drawn series left by the same amount, in proportion to
 * the wall-clock time since the newest sample, so when the next sample lands (and the chart
 * re-bases by that step) the picture is already where the chart puts it: the motion is continuous
 * and the data is untouched. Only the series layer (the clipped group) moves; axes and grid stay.
 * `spanMs` is the chart's x extent (newest minus oldest sample), `newestAt` the newest sample's time.
 * The slide is a pure function of time (capped only at the plot width): a late sample re-bases by exactly
 * the distance already slid, so it never jumps.
 */
export function LiveScroll({
  spanMs,
  newestAt,
  maxMs = Infinity,
  className,
  children,
}: {
  spanMs: number;
  newestAt: number;
  maxMs?: number;
  className?: string;
  children: ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.setProperty("--live-shift", "0");
    if (reducedMotion() || !(spanMs > 0) || !newestAt) return;
    const born = performance.now() - Math.max(0, Date.now() - newestAt);
    return ticker.subscribe((now) => {
      const rect = el.querySelector("clipPath rect");
      const w = Number(rect?.getAttribute("width") ?? 0);
      const age = Math.min(maxMs, Math.max(0, now - born));
      el.style.setProperty(
        "--live-shift",
        Math.min(w, (age / spanMs) * w).toFixed(2),
      );
    });
  }, [spanMs, newestAt, maxMs]);

  return (
    <div ref={box} className={`live-scroll ${className ?? ""}`}>
      {children}
    </div>
  );
}
