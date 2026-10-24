import { useEffect, useMemo, useState } from "react";
import {
  Card,
  EmptyState,
  Switch,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { Bug } from "lucide-react";
import { t, useLocale } from "./i18n/index.ts";
import { ApiError, type Connection } from "./api";
import { requestServerJson } from "./management-api";
import type { Row } from "./RequestTrace";
import { Presence } from "./motion/Presence";
import {
  anyDiffers,
  compareRequest,
  parseInFlight,
  type DebugInFlight,
} from "./observe-compare";

const KEY = "yunshu.requests.debugData";
const POLL_MS = 2000;

function stored(): boolean | null {
  try {
    const v = window.localStorage.getItem(KEY);
    return v === null ? null : v === "1";
  } catch {
    return null;
  }
}

/**
 * Debug data: for each in-flight request, what the engine's /debug/requests says (phase, prefill progress,
 * generated tokens, admit / first-token times) next to what this page displays. On by default when the
 * engine reports observe mode; the lead uses it to supervise instead of trusting either side alone.
 */
export function ObservePanel({
  rows,
  connection,
  observe,
}: {
  rows: readonly Row[];
  connection: Connection;
  observe: boolean;
}) {
  useLocale();
  const [pref, setPref] = useState<boolean | null>(stored),
    open = pref ?? observe;
  const [debug, setDebug] = useState<DebugInFlight[] | null>(null),
    [error, setError] = useState<string | null>(null);
  const toggle = (value: boolean) => {
    setPref(value);
    try {
      window.localStorage.setItem(KEY, value ? "1" : "0");
    } catch {
      /* per-viewer convenience only */
    }
  };
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      if (document.visibilityState === "visible") {
        try {
          const body = await requestServerJson(connection, "/debug/requests", {
            signal: controller.signal,
          });
          if (controller.signal.aborted) return;
          setDebug(parseInFlight(body));
          setError(null);
        } catch (e) {
          if (controller.signal.aborted) return;
          setDebug(null);
          setError(
            e instanceof ApiError && e.status === 404
              ? t("requests.debug.off")
              : e instanceof ApiError
                ? `HTTP ${e.status}`
                : t("requests.debug.failed"),
          );
        }
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, POLL_MS);
    };
    void poll();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [open, connection.baseUrl, connection.token]);
  const live = useMemo(
    () => rows.filter((r) => r.phase !== "complete"),
    [rows],
  );
  const byId = useMemo(
    () => new Map((debug ?? []).map((d) => [d.request_id, d])),
    [debug],
  );
  const labels = {
    phase: t("requests.debug.col.phase"),
    prefill: t("requests.debug.col.prefill"),
    tokens: t("requests.debug.col.tokens"),
    admit: t("requests.debug.col.admit"),
    first: t("requests.debug.col.first"),
  } as const;
  return (
    <Card data-testid="observe-panel" className="space-y-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Bug size={16} aria-hidden />
          <h2 className="text-base font-semibold">
            {t("requests.debug.title")}
          </h2>
          {observe && (
            <span
              data-testid="observe-badge"
              className="rounded-full border px-2 py-0.5 text-xs text-muted-foreground"
            >
              {t("requests.debug.observe")}
            </span>
          )}
        </div>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={open}
            onCheckedChange={toggle}
            aria-label={t("requests.debug.toggle")}
            data-testid="observe-toggle"
          />
          {t("requests.debug.toggle")}
        </label>
      </div>
      <Presence show={open}>
        <p className="mb-2 text-xs text-muted-foreground">
          {t("requests.debug.help")}
        </p>
        {error ? (
          <p role="status" className="text-sm text-warning">
            {error}
          </p>
        ) : live.length === 0 ? (
          <EmptyState
            size="inline"
            title={t("requests.debug.emptyTitle")}
            description={t("requests.debug.emptyBody")}
          />
        ) : (
          <Table scrollLabel={t("requests.debug.title")}>
            <Thead>
              <Tr>
                <Th>{t("requests.debug.col.id")}</Th>
                {(Object.keys(labels) as (keyof typeof labels)[]).map((k) => (
                  <Th key={k}>{labels[k]}</Th>
                ))}
              </Tr>
            </Thead>
            <Tbody>
              {live.map((row) => {
                const cells = compareRequest(row, byId.get(row.id));
                const bad = anyDiffers(cells);
                return (
                  <Tr
                    key={row.id}
                    data-testid="observe-row"
                    data-differs={bad ? "true" : "false"}
                  >
                    <Td className="font-mono text-xs">{row.id.slice(-10)}</Td>
                    {cells.map((c) => (
                      <Td
                        key={c.key}
                        data-state={c.state}
                        className={`tabular-nums text-xs ${c.state === "differs" ? "text-destructive" : c.state === "missing" ? "text-muted-foreground" : ""}`}
                      >
                        <div>
                          {t("requests.debug.ui")}: {c.ui || "-"}
                        </div>
                        <div>
                          {t("requests.debug.engine")}: {c.debug}
                        </div>
                      </Td>
                    ))}
                  </Tr>
                );
              })}
            </Tbody>
          </Table>
        )}
      </Presence>
    </Card>
  );
}
