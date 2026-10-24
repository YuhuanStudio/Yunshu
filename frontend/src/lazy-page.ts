import {
  createElement,
  lazy,
  useState,
  type ComponentProps,
  type ComponentType,
} from "react";

// The console is served by the engine itself, so a page chunk requested while the engine restarts
// fails to load, and browsers keep a failed module import: importing it again fails the same way,
// so the page stayed on its error card even after the engine was back. Wait until the console is
// reachable again, then reload once (the hash keeps the page); at most one automatic reload per
// RELOAD_GUARD_MS, after which the page boundary shows the error and its retry reloads.
const RELOAD_KEY = "yunshu.console.chunk-reload";
const RELOAD_GUARD_MS = 30_000;
const WAITS_MS = [250, 500, 1000, 2000, 4000, 8000, 8000, 8000];

async function reachable(): Promise<boolean> {
  for (const wait of WAITS_MS) {
    try {
      const r = await fetch(location.pathname, { cache: "no-store" });
      if (r.ok) return true;
    } catch {
      // still down
    }
    await new Promise((resolve) => setTimeout(resolve, wait));
  }
  return false;
}

function mayReload(): boolean {
  try {
    const last = Number(sessionStorage.getItem(RELOAD_KEY) ?? 0);
    if (Date.now() - last < RELOAD_GUARD_MS) return false;
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()));
    return true;
  } catch {
    return false; // no storage: never risk a reload loop
  }
}

export function reloadForChunk() {
  try {
    sessionStorage.removeItem(RELOAD_KEY);
  } catch {
    // ignore
  }
  location.reload();
}

async function recovering<T>(load: () => Promise<T>): Promise<T> {
  try {
    return await load();
  } catch (error) {
    if ((await reachable()) && mayReload()) {
      location.reload();
      return new Promise<T>(() => {}); // the reload replaces this page
    }
    throw error;
  }
}

const warmers: Array<() => void> = [];
/** Load every page chunk in the background (call when the browser is idle): a later route change then
 *  renders the page in the same commit, so the page transition never shows an empty hole. */
export function warmPages() {
  for (const warm of warmers) warm();
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- same constraint as React.lazy
export function lazyPage<T extends ComponentType<any>>(
  load: () => Promise<{ default: T }>,
): ComponentType<ComponentProps<T>> {
  const Inner = lazy(() => recovering(load));
  let loaded: T | null = null;
  warmers.push(() => {
    if (!loaded)
      load().then(
        (m) => {
          loaded = m.default;
        },
        () => {},
      );
  });
  return function LazyPage(props: ComponentProps<T>) {
    // Chosen once per mounted page: switching component type later would remount it.
    const [Page] = useState<ComponentType<any>>(() => loaded ?? Inner); // eslint-disable-line @typescript-eslint/no-explicit-any
    return createElement(Page, props);
  };
}
