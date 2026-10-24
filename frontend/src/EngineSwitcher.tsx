import { useCallback, useEffect, useState } from "react";
import { Button } from "@yuhuanowo/yunui";
import { StatusIslandCard } from "@yuhuanowo/yunui/patterns";
import {
  fetchEngines,
  pinEngine,
  type Connection,
  type EngineList,
} from "./api";
import { t } from "./i18n/index.ts";
import { shortModelName } from "./status-island";
import { uptimeText } from "./footer-status";

/**
 * Polls the console process's engine list every 3 s while mounted (the island is open). `null`
 * until the first answer, and when the console follows one fixed engine (or there is no console
 * process): then no switcher is shown.
 */
export function useEngineList(connection: Connection): {
  list: EngineList | null;
  pin: (url: string | null) => Promise<void>;
} {
  const [list, setList] = useState<EngineList | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const next = await fetchEngines(connection, {
          signal: controller.signal,
          timeoutMs: 4000,
        });
        if (!controller.signal.aborted)
          setList(next.mode === "auto" ? next : null);
      } catch {
        if (!controller.signal.aborted) setList(null);
      }
      if (!controller.signal.aborted)
        timer = setTimeout(() => void tick(), 3000);
    };
    void tick();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection]);
  const pin = useCallback(
    async (url: string | null) => {
      try {
        const next = await pinEngine(connection, url);
        setList(next.mode === "auto" ? next : null);
      } catch {
        // The next poll shows the truth (the engine may have just gone away).
      }
    },
    [connection],
  );
  return { list, pin };
}

/** Compact engine switcher: the engine being followed, the others discovered, pin / follow newest. */
export function EngineSwitcher({
  list,
  pin,
}: {
  list: EngineList;
  pin: (url: string | null) => Promise<void>;
}) {
  if (list.waiting)
    return (
      <StatusIslandCard
        title={t("shell.engines.title")}
        data-testid="island-engines"
      >
        <div className="text-[13px] text-muted-foreground" role="status">
          {t("shell.engines.waiting")}
        </div>
      </StatusIslandCard>
    );
  return (
    <StatusIslandCard
      title={t("shell.engines.title")}
      note={
        list.pinned
          ? t("shell.engines.pinnedNote")
          : t("shell.engines.followingNote")
      }
      data-testid="island-engines"
    >
      <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
        {list.data.map((e) => (
          <li
            key={e.url}
            className="flex items-center justify-between gap-2 text-[13px]"
            aria-current={e.selected ? "true" : undefined}
          >
            <span className="min-w-0 truncate">
              <span className="tabular-nums">:{e.port ?? e.url}</span>{" "}
              <span className="text-muted-foreground">
                {e.model ? shortModelName(e.model) : t("shell.island.noModel")}
                {e.uptime_seconds != null
                  ? ` · ${uptimeText(e.uptime_seconds)}`
                  : ""}
              </span>
            </span>
            {e.selected ? (
              <span className="shrink-0 text-xs text-foreground">
                {e.pinned
                  ? t("shell.engines.pinned")
                  : t("shell.engines.current")}
              </span>
            ) : (
              <Button
                variant="outline"
                size="sm"
                className="shrink-0 rounded-full"
                onClick={() => void pin(e.url)}
              >
                {t("shell.engines.use")}
              </Button>
            )}
          </li>
        ))}
      </ul>
      {list.pinned && (
        <Button
          variant="ghost"
          size="sm"
          className="mt-1.5 rounded-full"
          onClick={() => void pin(null)}
        >
          {t("shell.engines.unpin")}
        </Button>
      )}
    </StatusIslandCard>
  );
}
