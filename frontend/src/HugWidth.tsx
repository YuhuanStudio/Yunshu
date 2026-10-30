import {
  createContext,
  useContext,
  useLayoutEffect,
  useRef,
  type ReactNode,
} from "react";
import { WidthRatchet } from "./motion/hug-core.ts";

/**
 * A box that hugs its content and keeps its width steady while the content changes. It is as wide
 * as the widest content of the last `holdMs` (see WidthRatchet): it grows at once when a digit or a
 * word needs room and gives the room back only after the content stayed narrower for the whole
 * hold, so a settled reading has no dead space and a flickering one does not resize per sample.
 * The width is never animated: a step is one instant change, and steps are rare (a new digit
 * count, a new word). Use it for one NUMBER (see LiveNumber, HeldText) so the room it holds is a
 * digit wide; wrapped around a whole sentence it would hold the whole sentence's room.
 * Inside another HugWidth (a LiveNumber in a number + unit box) it is a plain box: the outer one holds
 * the room for both, and two nested observers would resize each other in one frame (WebKit then reports
 * "ResizeObserver loop completed with undelivered notifications").
 */
const Hugged = createContext(false);

export function HugWidth({
  children,
  holdMs = 6000,
  align,
  className = "",
}: {
  children: ReactNode;
  holdMs?: number;
  /** Where the content sits inside room it holds: a number is "right", so its digits stay put. */
  align?: "right";
  className?: string;
}) {
  const nested = useContext(Hugged);
  const outer = useRef<HTMLSpanElement>(null);
  const inner = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const box = outer.current;
    const content = inner.current;
    if (nested || !box || !content || typeof ResizeObserver === "undefined")
      return;
    const ratchet = new WidthRatchet(holdMs, 0.5, 1.5);
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
    const observer = new ResizeObserver(measure);
    observer.observe(content);
    return () => {
      observer.disconnect();
      if (timer != null) clearTimeout(timer);
    };
  }, [holdMs, nested]);
  return (
    <span
      ref={outer}
      data-hug=""
      className={`inline-block ${align === "right" ? "text-right" : ""} ${className}`}
    >
      <span
        ref={inner}
        data-hug-natural=""
        className="inline-block whitespace-nowrap"
      >
        <Hugged.Provider value={true}>{children}</Hugged.Provider>
      </span>
    </span>
  );
}
