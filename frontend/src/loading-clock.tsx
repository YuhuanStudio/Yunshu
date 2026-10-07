import { useEffect, useState } from "react";
import { t } from "./i18n/index.ts";
import { elapsed } from "./i18n/format.ts";

/** When this console first saw (or started) each model's load, in ms. A load cannot be timed server-side yet. */
const since = new Map<string, number>();

export function markLoading(id: string, loading: boolean, at = Date.now()) {
  if (!loading) since.delete(id);
  else if (!since.has(id)) since.set(id, at);
}
export const loadingSeconds = (id: string, now = Date.now()): number | null => {
  const s = since.get(id);
  return s == null ? null : Math.max(0, Math.floor((now - s) / 1000));
};

/** "Loading 12s" for one model, ticking once a second; the width is fixed so nothing shifts. */
export function LoadingElapsed({ id }: { id: string }) {
  const [, tick] = useState(0);
  useEffect(() => {
    markLoading(id, true);
    const timer = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, [id]);
  const s = loadingSeconds(id);
  return (
    <span className="inline-block min-w-[9ch] tabular-nums text-muted-foreground">
      {t("models.loading.elapsed", { time: s == null ? "—" : elapsed(s) })}
    </span>
  );
}
