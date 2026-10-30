// Layout stability of live readings. The page recorder samples the bounding box of every element in
// the header, main and footer every 50 ms and, for every status pill, how much empty room trails its
// text. These pure functions turn the samples into the three numbers the console is held to:
//
//   flips   an element whose width went A -> B -> A inside `windowMs`: a width that follows the value
//           instead of settling (a digit count flickering at a boundary, a phase label alternating).
//   steps   how often an element changed width at all.
//   shoves  an element that moved horizontally or vertically although its own size did not change,
//           because something before it changed size. Counted per sample, scroll discounted.
//   dead    trailing empty room inside a pill, in px (the dead space the user sees as a wide pill).
//
// A frame is {t, scrollTop, boxes: {id: [left, top, width, height]}, dead: {id: px}}; `keys` maps ids
// to a readable name.

const round = (n) => Math.round(n * 10) / 10;

/** Frames of `ids` only (all when omitted), so a spec can look at one region. */
export function select(frames, keys, match) {
  const ids = new Set(
    Object.entries(keys)
      .filter(([, k]) => match(k))
      .map(([id]) => id),
  );
  return frames.map((f) => ({
    ...f,
    boxes: Object.fromEntries(
      Object.entries(f.boxes).filter(([id]) => ids.has(id)),
    ),
    dead: Object.fromEntries(
      Object.entries(f.dead ?? {}).filter(([id]) => ids.has(id)),
    ),
  }));
}

export function widthSeries(frames, id) {
  const out = [];
  for (const f of frames) {
    const b = f.boxes[id];
    if (b) out.push({ t: f.t, w: b[2] });
  }
  return out;
}

/**
 * Width steps and A -> B -> A reversals of one element. Changes in consecutive frames (an eased step
 * spans a few 50 ms samples) are one step.
 */
export function widthEvents(
  series,
  { tol = 0.6, windowMs = 3000, joinMs = 110 } = {},
) {
  const raw = [];
  for (let i = 1; i < series.length; i++)
    if (Math.abs(series[i].w - series[i - 1].w) > tol)
      raw.push({ t: series[i].t, from: series[i - 1].w, to: series[i].w });
  const changes = [];
  for (const c of raw) {
    const last = changes.at(-1);
    if (last && c.t - last.end <= joinMs) {
      last.to = c.to;
      last.end = c.t;
    } else changes.push({ ...c, end: c.t });
  }
  let flips = 0;
  for (let i = 1; i < changes.length; i++)
    if (
      changes[i].t - changes[i - 1].end <= windowMs &&
      Math.abs(changes[i].to - changes[i - 1].from) <= tol
    )
      flips++;
  return { steps: changes.length, flips, changes };
}

/** Per element: its own width steps and flips; across the page: shoves (moved, not resized). */
export function analyze(
  frames,
  keys,
  { tol = 0.6, boxed = {}, flipMs = 1200 } = {},
) {
  const ids = new Set(frames.flatMap((f) => Object.keys(f.boxes)));
  const perElement = [];
  for (const id of ids) {
    const ev = widthEvents(widthSeries(frames, id), { tol, windowMs: flipMs });
    if (ev.steps)
      perElement.push({ id, key: keys[id], boxed: !!boxed[id], ...ev });
  }
  let shoves = 0;
  const shoved = new Map();
  let jitter = 0;
  let movedFrames = 0;
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1];
    const b = frames[i];
    const scrolled = b.scrollTop - a.scrollTop;
    const before = jitter;
    for (const [id, now] of Object.entries(b.boxes)) {
      const before = a.boxes[id];
      if (!before) continue;
      const dLeft = now[0] - before[0];
      const dTop = now[1] - before[1] + scrolled;
      const resized =
        Math.abs(now[2] - before[2]) > tol ||
        Math.abs(now[3] - before[3]) > tol;
      if (Math.abs(dLeft) > tol || Math.abs(dTop) > tol) {
        if (!resized) {
          shoves++;
          shoved.set(keys[id], (shoved.get(keys[id]) ?? 0) + 1);
        }
        jitter++;
      }
    }
    if (jitter > before) movedFrames++;
  }
  const dead = [];
  for (const f of frames)
    for (const [id, px] of Object.entries(f.dead ?? {}))
      dead.push({ id, key: keys[id], px: round(px), t: f.t });
  return {
    steps: perElement.reduce((n, e) => n + e.steps, 0),
    flips: perElement.reduce((n, e) => n + e.flips, 0),
    perElement: perElement.sort((x, y) => y.steps - x.steps),
    shoves,
    shoved: [...shoved.entries()].sort((x, y) => y[1] - x[1]).slice(0, 8),
    moved: jitter,
    movedFrames,
    dead,
  };
}

/**
 * The script that runs inside the page: every 50 ms the box of every element under header, main and
 * footer, and the trailing room of every status pill. `window.__stab.stop()` returns {frames, keys,
 * entries}. Plain function body so Playwright can inject it with `addInitScript`.
 */
export const STABILITY_RECORDER = `
(() => {
  const frames = [], keys = {}, entries = [], boxed = {};
  const ids = new WeakMap();
  let next = 1;
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) entries.push({ t: Math.round(e.startTime), value: e.value, hadRecentInput: e.hadRecentInput });
    }).observe({ type: "layout-shift", buffered: true });
  } catch {}
  const label = (el) => {
    const tid = el.closest("[data-testid]");
    const tag = el.tagName.toLowerCase();
    const cls = String(el.className && el.className.baseVal === undefined ? el.className : "").split(" ")[0];
    return (tid ? tid.getAttribute("data-testid") + " > " : "") + tag + "." + cls + ":" + (el.textContent || "").trim().slice(0, 12);
  };
  const trailing = (li) => {
    const r = li.getBoundingClientRect();
    const cs = getComputedStyle(li);
    const walker = document.createTreeWalker(li, NodeFilter.SHOW_TEXT);
    let right = -Infinity;
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!n.textContent.trim() || n.parentElement.closest(".sr-only")) continue;
      const range = document.createRange();
      range.selectNodeContents(n);
      const rr = range.getBoundingClientRect();
      if (rr.width) right = Math.max(right, rr.right);
    }
    return right === -Infinity ? 0 : r.right - parseFloat(cs.paddingRight) - right;
  };
  const sample = () => {
    const scroller = document.querySelector('[data-testid="page-scroll"]');
    const boxes = {}, dead = {};
    let n = 0;
    // a row that is fading in or sliding to its lane on purpose (keyframe, WAAPI or transform
    // animation) is not jitter; the eased width step of a hugging box (a width transition) stays measured
    const animated = new Set(document.getAnimations().filter((x) => !(self.CSSTransition && x instanceof CSSTransition && x.transitionProperty === "width")).map((x) => x.effect && x.effect.target).filter(Boolean));
    const memo = new WeakMap();
    const moving = (el) => { if (!el) return false; if (memo.has(el)) return memo.get(el); const r = animated.has(el) || moving(el.parentElement); memo.set(el, r); return r; };
    for (const el of document.querySelectorAll("header *, main *, footer *, [role=dialog] *, [aria-label] > li, li[data-tone]")) {
      if (el.closest(".sr-only, [data-hug-natural], svg")) continue;
      const cs0 = ids.has(el) ? null : getComputedStyle(el);
      if (cs0 && cs0.position === "absolute") { ids.set(el, -1); continue; }
      if (ids.get(el) === -1) continue;
      if (animated.size && moving(el)) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1 || r.bottom < 0 || r.top > innerHeight) continue;
      let id = ids.get(el);
      if (!id) {
        id = next++; ids.set(el, id); keys[id] = label(el);
        const cs = cs0 || getComputedStyle(el);
        boxed[id] = (parseFloat(cs.borderTopWidth) > 0 && cs.borderTopStyle !== "none") || !/rgba\\(0, 0, 0, 0\\)|transparent/.test(cs.backgroundColor);
      }
      boxes[id] = [r.left, r.top, r.width, r.height];
      if (el.matches("li[data-tone]")) dead[id] = trailing(el);
      if (++n >= 1500) break;
    }
    frames.push({ t: Math.round(performance.now()), scrollTop: scroller ? scroller.scrollTop : scrollY, boxes, dead });
  };
  const timer = setInterval(sample, 50);
  window.__stab = { stop() { clearInterval(timer); return { frames, keys, entries, boxed }; }, reset() { frames.length = 0; entries.length = 0; } };
})();
`;
