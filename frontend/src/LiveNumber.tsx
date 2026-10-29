import { useLayoutEffect, useRef } from "react";
import { fixed, number } from "./i18n/format.ts";
import { useLocale } from "./i18n/index.ts";
import { reducedMotion } from "./motion/ticker.ts";
import { Readout } from "./motion/readout-core.ts";
import { HugWidth } from "./HugWidth";

/**
 * A live reading drawn with tabular digits, so its width does not move (and a new digit count moves it
 * by one digit only: the number holds its own width, see HugWidth), at a pace a person can
 * read: a new sample is drawn at once if the last draw is at least ~240 ms old (about 4 Hz), else
 * the newest sample is drawn when that interval ends. It never counts through values the engine
 * did not report (no per-frame glide: a rate that ticks every frame is unreadable), and a sample
 * is never invented. Writes go straight to the DOM, no React render. Whenever `jumpKey` changes
 * (another request, another phase), under reduced motion, and for a reading that goes away (null),
 * the new value is shown at once.
 */
export function LiveNumber({
  value,
  digits = 0,
  jumpKey,
  format,
  className,
}: {
  value: number | null;
  digits?: number;
  jumpKey?: string | number | null;
  format?: (v: number) => string;
  className?: string;
}) {
  const locale = useLocale();
  const el = useRef<HTMLSpanElement>(null);
  const readout = useRef(new Readout<number | null>());
  const key = useRef(jumpKey);
  const shown = useRef<number | null | undefined>(undefined);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const draw = (v: number | null) => {
    shown.current = v;
    if (el.current)
      el.current.textContent =
        v == null
          ? "—"
          : format
            ? format(v)
            : digits > 0
              ? fixed(v, digits)
              : number(v, 0);
  };
  const stop = () => {
    if (timer.current != null) clearTimeout(timer.current);
    timer.current = null;
  };

  useLayoutEffect(() => {
    const now = performance.now();
    if (
      value == null ||
      reducedMotion() ||
      key.current !== jumpKey ||
      value === shown.current
    ) {
      key.current = jumpKey;
      stop();
      draw(readout.current.jump(value, now));
      return;
    }
    const v = readout.current.offer(value, now);
    if (v != null) {
      stop();
      draw(v);
      return;
    }
    const fire = () => {
      timer.current = null;
      const t = performance.now();
      const next = readout.current.flush(t);
      if (next != null) draw(next);
      else {
        const wait = readout.current.dueIn(t);
        if (wait != null) timer.current = setTimeout(fire, wait);
      }
    };
    if (timer.current == null)
      timer.current = setTimeout(fire, readout.current.dueIn(now) ?? 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, jumpKey, digits, locale]);
  useLayoutEffect(() => stop, []);

  return (
    <HugWidth align="right" holdMs={8000}>
      <span ref={el} className={`tabular-nums ${className ?? ""}`} />
    </HugWidth>
  );
}
