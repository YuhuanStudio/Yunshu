import { useLayoutEffect, useRef, type ReactNode } from "react";
import { WidthRatchet } from "./motion/hug-core.ts";

/**
 * A box that hugs its content and keeps its width steady while the content changes. It is as wide
 * as the widest content of the last `holdMs` (see WidthRatchet): it grows at once when a digit or a
 * word needs room and gives the room back only after the content stayed narrower for the whole
 * hold, so a settled reading has no dead space and a flickering one does not resize per sample.
 * A step eases over the expand duration (styles.css `[data-hug]`, off under reduced motion) and the
 * text is revealed by the box, never drawn outside it.
 */
export function HugWidth({
  children,
  holdMs = 6000,
  className = "",
}: {
  children: ReactNode;
  holdMs?: number;
  className?: string;
}) {
  const outer = useRef<HTMLSpanElement>(null);
  const inner = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const box = outer.current;
    const content = inner.current;
    if (!box || !content || typeof ResizeObserver === "undefined") return;
    const ratchet = new WidthRatchet(holdMs);
    let timer: ReturnType<typeof setTimeout> | null = null;
    const apply = (w: number) => {
      box.style.width = `${w}px`;
    };
    const arm = (now: number) => {
      if (timer != null) clearTimeout(timer);
      timer = null;
      const due = ratchet.dueIn(now);
      if (due != null)
        timer = setTimeout(() => {
          const t = performance.now();
          apply(ratchet.settle(t));
          arm(t);
        }, due + 5);
    };
    const measure = () => {
      const now = performance.now();
      const natural = Math.ceil(content.getBoundingClientRect().width * 4) / 4;
      apply(ratchet.observe(natural, now));
      arm(now);
    };
    measure();
    const frame = requestAnimationFrame(() => {
      box.dataset.ready = "";
    });
    const observer = new ResizeObserver(measure);
    observer.observe(content);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      if (timer != null) clearTimeout(timer);
    };
  }, [holdMs]);
  return (
    <span ref={outer} data-hug="" className={`inline-block ${className}`}>
      <span
        ref={inner}
        data-hug-natural=""
        className="inline-block whitespace-nowrap"
      >
        {children}
      </span>
    </span>
  );
}
