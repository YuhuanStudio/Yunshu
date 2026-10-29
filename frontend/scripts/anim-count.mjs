// How many animations a live feed starts. A monitoring screen under a bursty feed must not play a
// motion per update: routine ticks swap in place, only a meaningful change (a state change, a row
// after a quiet spell) may animate. The recorder counts every CSS animation, CSS transition and
// Web Animations call the page starts (`animationstart`, `transitionrun`, `Element.animate`), with
// the element and the property, so a spec can assert a low bound per second and name the culprit.
//
// Continuous time-driven motion (a chart sliding with the wall clock, a progress bar advancing at
// its observed rate) is not an animation per update and is not counted here; layout stability of
// those is held by the stability recorder.

/**
 * Events -> {total, gestures, perSecond, peak, byName}. A gesture is one visual motion: events of
 * the same kind within 150 ms of each other (a list making room moves four rows at once = one
 * gesture). perSecond and peak count gestures; `total` counts elements.
 */
export function summarize(events, seconds) {
  const sorted = [...events].sort((a, b) => a.t - b.t);
  const lastOf = new Map();
  const gestureTimes = [];
  for (const e of sorted) {
    const last = lastOf.get(e.name);
    if (last === undefined || e.t - last > 150) gestureTimes.push(e.t);
    lastOf.set(e.name, e.t);
  }
  return {
    ...count(events, gestureTimes, seconds),
  };
}

function count(events, gestureTimes, seconds) {
  const byName = new Map();
  for (const e of events) byName.set(e.name, (byName.get(e.name) ?? 0) + 1);
  const times = gestureTimes;
  let peak = 0;
  for (let i = 0, j = 0; i < times.length; i++) {
    while (times[i] - times[j] >= 1000) j++;
    peak = Math.max(peak, i - j + 1);
  }
  return {
    total: events.length,
    gestures: times.length,
    perSecond: times.length / Math.max(seconds, 1e-9),
    peak,
    byName: [...byName.entries()].sort((a, b) => b[1] - a[1]),
  };
}

export const ANIM_RECORDER = `
(() => {
  const events = [];
  const name = (el, what) => {
    const tid = el && el.closest ? el.closest("[data-testid]") : null;
    const tag = el && el.tagName ? el.tagName.toLowerCase() : "?";
    const cls = el && typeof el.className === "string" ? el.className.split(" ")[0] : "";
    return what + " " + (tid ? tid.getAttribute("data-testid") + " > " : "") + tag + (cls ? "." + cls : "");
  };
  const log = (el, what) => events.push({ t: Math.round(performance.now()), name: name(el, what) });
  document.addEventListener("animationstart", (e) => log(e.target, "css:" + e.animationName), true);
  document.addEventListener("transitionrun", (e) => log(e.target, "transition:" + e.propertyName), true);
  const animate = Element.prototype.animate;
  Element.prototype.animate = function (...args) { log(this, "waapi"); return animate.apply(this, args); };
  window.__anim = {
    reset() { events.length = 0; },
    stop() { return events.slice(); },
  };
})();
`;
