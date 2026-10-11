import { useLayoutEffect, useRef, type RefObject } from "react";
import { reducedMotion } from "./ticker";

/**
 * FLIP for a live list: when rows are inserted or removed above others, the rows below are moved to
 * where they were (an inverse transform) in the same frame as the layout change, then glide to their
 * new place. The browser sees no layout shift and the eye sees the list make room.
 *
 * Mark each row with `data-flip="<stable id>"`; call with the list's container.
 */
export function useFlip(container: RefObject<HTMLElement | null>, ms = 200) {
  const tops = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const root = container.current;
    if (!root) return;
    const rows = [...root.querySelectorAll<HTMLElement>("[data-flip]")];
    const next = new Map<string, number>();
    for (const el of rows) {
      const key = el.dataset.flip!;
      const top = el.offsetTop;
      next.set(key, top);
      const before = tops.current.get(key);
      if (before == null || reducedMotion()) continue;
      // A row still gliding from the last update is drawn `residual` px off its layout position:
      // start from where it is drawn, not from where it was laid out.
      const residual = el.style.transform
        ? new DOMMatrixReadOnly(getComputedStyle(el).transform).m42
        : 0;
      const dy = before - top + residual;
      if (Math.abs(dy) < 1) continue;
      el.style.transition = "none";
      el.style.transform = `translateY(${dy}px)`;
      void el.offsetHeight; // commit the inverted position before animating
      el.style.transition = `transform ${ms}ms ease-out`;
      el.style.transform = "";
    }
    tops.current = next;
  });
}
