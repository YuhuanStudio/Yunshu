import { useEffect, useRef } from "react";
import { parseRoute, withoutIntent, type Page } from "./route.ts";

/**
 * Lets a page answer a palette verb or a notification link such as
 * `#/models?action=load&model=<id>`. The handler runs once for each intent
 * (on mount and on every hash change while the page is open); the intent keys
 * are then removed from the address, so a reload or Back does not repeat it.
 *
 *   useRouteAction("cache", (action) => { if (action === "clear") openDialog(); });
 */
export function useRouteAction(
  page: Page,
  handler: (action: string, query: URLSearchParams) => void,
): void {
  const latest = useRef(handler);
  latest.current = handler;
  useEffect(() => {
    const run = () => {
      const route = parseRoute(location.hash);
      const action = route.query.get("action");
      if (route.page !== page || !action) return;
      history.replaceState(null, "", withoutIntent(location.hash));
      latest.current(action, route.query);
    };
    run();
    addEventListener("hashchange", run);
    return () => removeEventListener("hashchange", run);
  }, [page]);
}
