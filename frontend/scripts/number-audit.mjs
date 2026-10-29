// Every number that changes on screen under a live feed: is it drawn with tabular digits, and how
// often does its text change? The recorder watches text changes (MutationObserver) and, for every
// element whose text carries a digit and changed, keeps its change count and whether its computed
// style has tabular numerals (font-variant-numeric: tabular-nums, or a monospace face).

export const NUMBER_RECORDER = `
(() => {
  const changed = new Map();
  const note = (node) => {
    const el = node.nodeType === 3 ? node.parentElement : node;
    if (!el || !/\\d/.test(node.textContent || "")) return;
    changed.set(el, (changed.get(el) || 0) + 1);
  };
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === "characterData") note(m.target);
      else for (const n of m.addedNodes) if (n.nodeType === 3 || n.nodeType === 1) note(n);
    }
  }).observe(document, { subtree: true, characterData: true, childList: true });
  window.__num = {
    reset() { changed.clear(); },
    report() {
      const out = [];
      for (const [el, n] of changed) {
        if (!el.isConnected || el.closest("svg, .sr-only, title, head")) continue;
        const cs = getComputedStyle(el);
        const tid = el.closest("[data-testid]");
        out.push({
          key: (tid ? tid.getAttribute("data-testid") + " > " : "") + el.tagName.toLowerCase() + "." + String(el.className && el.className.baseVal === undefined ? el.className : "").split(" ")[0] + ":" + (el.textContent || "").trim().slice(0, 24),
          changes: n,
          tabular: /tabular-nums/.test(cs.fontVariantNumeric) || /mono/i.test(cs.fontFamily),
        });
      }
      return out;
    },
  };
})();
`;

/** Elements whose digits are not tabular. */
export const notTabular = (rows) => rows.filter((r) => !r.tabular);

/** Elements whose text changed more than `hz` times a second on average. */
export const tooFast = (rows, seconds, hz = 5) =>
  rows.filter((r) => r.changes / seconds > hz);
