import { useEffect, useRef, useState } from "react";

const reduced = () =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * A number that glides to its next value instead of jumping, for live readings that arrive many times a
 * second (tok/s, prefill tokens). It eases out over `ms` with requestAnimationFrame, never overshoots, and
 * is the value itself under reduced motion. `null` clears at once: a reading that went away is not tweened
 * down to zero.
 */
export function useTween(target: number | null, ms = 250): number | null {
  const [shown, setShown] = useState<number | null>(target);
  const from = useRef<number | null>(target);
  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (raf.current != null) cancelAnimationFrame(raf.current);
    const start = from.current;
    if (target == null || start == null || reduced() || start === target) {
      from.current = target;
      setShown(target);
      return;
    }
    const t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / ms);
      const eased = 1 - Math.pow(1 - k, 3);
      const v = start + (target - start) * eased;
      from.current = v;
      setShown(v);
      if (k < 1) raf.current = requestAnimationFrame(step);
      else {
        from.current = target;
        raf.current = null;
      }
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
    };
  }, [target, ms]);
  return shown;
}
