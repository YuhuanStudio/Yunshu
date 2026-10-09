// Pure analysis of what review-capture.mjs records in a page: layout-shift entries and, every 250 ms,
// the bounding boxes of the page's main cards, rows and numbers. Shared by the capture script, the
// replay spec and the node tests.

/** Sum of layout-shift values the browser reports, ignoring those right after user input. */
export function clsOf(entries) {
  return entries
    .filter((e) => !e.hadRecentInput)
    .reduce((n, e) => n + e.value, 0);
}

/**
 * Moves between two consecutive box samples that were not the page scrolling: an element still there
 * whose top moved or whose height or width changed by more than `tol` px. `frames` is
 * [{t, scrollTop, boxes: {id: {top, left, width, height, key}}}] in time order.
 */
export function boxDeltas(frames, tol = 1.5) {
  const out = [];
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1];
    const b = frames[i];
    const scrolled = b.scrollTop - a.scrollTop;
    for (const [id, now] of Object.entries(b.boxes)) {
      const before = a.boxes[id];
      if (!before) continue;
      const dTop = now.top - before.top + scrolled;
      const dH = now.height - before.height;
      const dW = now.width - before.width;
      if (Math.abs(dTop) > tol || Math.abs(dH) > tol || Math.abs(dW) > tol)
        out.push({
          t: b.t,
          id,
          key: now.key,
          dTop: round(dTop),
          dH: round(dH),
          dW: round(dW),
        });
    }
  }
  return out;
}

const round = (n) => Math.round(n * 10) / 10;

/** The elements that moved most often / most in total, for a short "who is jumping" list. */
export function topMovers(deltas, limit = 8) {
  const by = new Map();
  for (const d of deltas) {
    const cur = by.get(d.id) ?? { key: d.key, moves: 0, total: 0 };
    cur.moves += 1;
    cur.total += Math.abs(d.dTop) + Math.abs(d.dH) + Math.abs(d.dW);
    by.set(d.id, cur);
  }
  return [...by.values()]
    .sort((x, y) => y.total - x.total)
    .slice(0, limit)
    .map((m) => ({ ...m, total: round(m.total) }));
}

export function summarize(entries, frames) {
  const deltas = boxDeltas(frames);
  return {
    cls: Math.round(clsOf(entries) * 10000) / 10000,
    shifts: entries.length,
    moves: deltas.length,
    movers: topMovers(deltas),
  };
}

/**
 * The script that runs inside the page: starts a layout-shift observer and a 250 ms sampler of element
 * boxes, exposes `window.__review.stop()` returning {entries, frames}. Plain function body so Playwright
 * can inject it with `addInitScript`.
 */
export const PAGE_RECORDER = `
(() => {
  const entries = [];
  const frames = [];
  const ids = new WeakMap();
  let next = 1;
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) entries.push({ t: Math.round(e.startTime), value: e.value, hadRecentInput: e.hadRecentInput });
    }).observe({ type: "layout-shift", buffered: true });
  } catch {}
  const SEL = "main [data-testid], main section, main li, main tr, main [class*='rounded-'], main h1, main h2, main button";
  const keyOf = (el) => (el.getAttribute("data-testid") || el.tagName.toLowerCase() + "." + (el.className && el.className.toString().split(" ")[0])) + ":" + (el.textContent || "").trim().slice(0, 14);
  const sample = () => {
    const scroller = document.querySelector('[data-testid="page-scroll"]');
    const boxes = {};
    let n = 0;
    for (const el of document.querySelectorAll(SEL)) {
      const r = el.getBoundingClientRect();
      if (r.width < 24 || r.height < 12 || r.bottom < 0 || r.top > innerHeight) continue;
      let id = ids.get(el);
      if (!id) { id = next++; ids.set(el, id); }
      boxes[id] = { top: r.top, left: r.left, width: r.width, height: r.height, key: keyOf(el) };
      if (++n >= 220) break;
    }
    frames.push({ t: Math.round(performance.now()), scrollTop: scroller ? scroller.scrollTop : scrollY, boxes });
  };
  const timer = setInterval(sample, 250);
  window.__review = { stop() { clearInterval(timer); return { entries, frames }; }, reset() { entries.length = 0; frames.length = 0; } };
})();
`;
