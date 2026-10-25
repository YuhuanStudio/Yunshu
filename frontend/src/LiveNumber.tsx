import { useLayoutEffect, useRef } from "react";
import { fixed, number } from "./i18n/format.ts";
import { useLocale } from "./i18n/index.ts";
import { ticker, reducedMotion } from "./motion/ticker.ts";
import { Tween } from "./motion/tween-core.ts";

/**
 * A live reading drawn with tabular digits, so its width does not move. A count or any other fact
 * (the default) shows the sampled value at once: a glide would draw numbers the engine never
 * reported. A RATE (`glide`) eases to each new sample (ease-out, about one sample interval); it
 * only moves between two real samples and settles on the engine's value. Frames write the text straight into the
 * DOM from the shared ticker: no React render per frame. Under reduced motion, and whenever
 * `jumpKey` changes (another request, another phase), it shows the new value at once, because
 * gliding through values that never existed would be a lie.
 */
export function LiveNumber({
  value,
  digits = 0,
  ms = 220,
  glide = false,
  jumpKey,
  format,
  className,
}: {
  value: number | null;
  digits?: number;
  ms?: number;
  /** Smooth between samples; only for rates (tok/s). Counts and other facts snap. */
  glide?: boolean;
  jumpKey?: string | number | null;
  format?: (v: number) => string;
  className?: string;
}) {
  const locale = useLocale();
  const el = useRef<HTMLSpanElement>(null);
  const tween = useRef<Tween | null>(null);
  const key = useRef(jumpKey);
  const off = useRef<(() => void) | null>(null);
  const draw = (v: number | null) => {
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

  useLayoutEffect(() => {
    const stop = () => {
      off.current?.();
      off.current = null;
    };
    if (!tween.current) tween.current = new Tween(value, ms);
    const tw = tween.current;
    if (!glide || reducedMotion() || key.current !== jumpKey || value == null) {
      key.current = jumpKey;
      stop();
      tw.jump(value);
      draw(value);
      return;
    }
    const now = performance.now();
    tw.retarget(value, now);
    draw(tw.valueAt(now));
    if (!off.current && !tw.settled(now))
      off.current = ticker.subscribe((t) => {
        draw(tw.valueAt(t));
        if (tw.settled(t)) stop();
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, jumpKey, digits, locale, glide]);
  useLayoutEffect(
    () => () => {
      off.current?.();
      off.current = null;
    },
    [],
  );

  return <span ref={el} className={`tabular-nums ${className ?? ""}`} />;
}
