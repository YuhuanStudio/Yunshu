import { useEffect, useState } from "react";
import { Card } from "@yuhuanowo/yunui";
import type { Connection } from "./api";
import { bytesText } from "./byte-format";
import {
  parseLifecycle,
  requestHitRate,
  type LifecycleModel,
  type Stage,
} from "./cache-lifecycle";
import { t, tr, useLocale } from "./i18n/index.ts";
import { number, percent } from "./i18n/format";
import { requestServerJson } from "./management-api";
import { modelLabel } from "./ui";

const STAGES: Stage[] = ["admission", "movement", "removal", "hits"];
const POLL_MS = 8000;

function useLifecycle(connection: Connection, enabled: boolean) {
  const [data, setData] = useState<LifecycleModel[] | null | "unavailable">(
    null,
  );
  useEffect(() => {
    if (!enabled) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const c = new AbortController();
    const tick = async () => {
      try {
        const raw = await requestServerJson(connection, "/debug/kv-cache", {
          signal: c.signal,
        });
        if (!stop) setData(parseLifecycle(raw));
      } catch {
        if (!stop) setData((d) => (d === null ? "unavailable" : d));
      }
      if (!stop) timer = setTimeout(() => void tick(), POLL_MS);
    };
    void tick();
    return () => {
      stop = true;
      c.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, enabled]);
  return data;
}

/**
 * How the prefix cache has behaved since the engine started: counters grouped by what happened
 * (admitted, moved, removed, hit). Hit requests and cached tokens are separate lines, and the
 * warm tier's uncompressed size is marked as an estimate.
 */
export function CacheLifecyclePanel({
  connection,
  enabled,
}: {
  connection: Connection;
  enabled: boolean;
}) {
  useLocale();
  const data = useLifecycle(connection, enabled);
  if (!enabled || data === null) return null;
  if (data === "unavailable" || data.length === 0)
    return (
      <p
        className="text-xs text-muted-foreground"
        data-testid="cache-lifecycle-unavailable"
      >
        {t("cache.life.unavailable")}
      </p>
    );
  return (
    <div className="space-y-3" data-testid="cache-lifecycle">
      <div>
        <h2 className="heading-md">{t("cache.life.title")}</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          {t("cache.life.desc")}
        </p>
      </div>
      {data.map((m) => {
        const rate = requestHitRate(m);
        const bytes = [
          m.bytes.ram != null &&
            t("cache.life.bytes.ram", { size: bytesText(m.bytes.ram) }),
          m.bytes.warm != null &&
            t("cache.life.bytes.warm", { size: bytesText(m.bytes.warm) }) +
              (m.bytes.warmRatio != null && m.bytes.warmRatio > 1
                ? " " +
                  t("cache.life.bytes.warmRaw", {
                    size: bytesText(m.bytes.warm * m.bytes.warmRatio),
                  })
                : ""),
          m.bytes.ssd != null &&
            t("cache.life.bytes.ssd", { size: bytesText(m.bytes.ssd) }),
          m.entries != null &&
            t("cache.life.entries", { n: number(m.entries, 0) }),
        ].filter(Boolean);
        return (
          <Card
            key={m.model}
            className="min-w-0 space-y-3 p-4"
            data-model={m.model}
          >
            <div className="min-w-0">
              <h3 className="truncate text-sm font-semibold" title={m.model}>
                {t("cache.life.model", { model: modelLabel(m.model) })}
              </h3>
              {bytes.length > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {bytes.join(" · ")}
                </p>
              )}
            </div>
            {m.counters.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                {t("cache.life.empty")}
              </p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                {STAGES.map((stage) => {
                  const rows = m.counters.filter((c) => c.stage === stage);
                  if (rows.length === 0) return null;
                  return (
                    <section key={stage} data-stage={stage} className="min-w-0">
                      <h4 className="text-xs font-semibold text-muted-foreground">
                        {/* i18n-keys: cache.life.stage. */}
                        {tr(`cache.life.stage.${stage}`)}
                      </h4>
                      <dl className="mt-1 space-y-1 text-sm">
                        {rows.map((c) => (
                          <div
                            key={c.id}
                            className="flex items-baseline justify-between gap-3"
                            data-counter={c.id}
                          >
                            <dt className="min-w-0 text-muted-foreground">
                              {/* i18n-keys: cache.life.counter. */}
                              {tr(`cache.life.counter.${c.id}`)}
                            </dt>
                            <dd className="shrink-0 tabular-nums">
                              {number(c.value, 0)}
                            </dd>
                          </div>
                        ))}
                        {stage === "hits" && rate != null && (
                          <div className="text-xs text-muted-foreground">
                            {t("cache.life.hitRate", {
                              pct: percent(rate, 1),
                            })}
                          </div>
                        )}
                      </dl>
                    </section>
                  );
                })}
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
}
