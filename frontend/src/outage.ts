import { useEffect, useRef, useState } from "react";
import { fetchEngines, type Connection, type EngineList } from "./api";
import { fetchConsoleState } from "./history-api";
import { outageView, type OutageInput, type OutageView } from "./outage-view";

export type { OutageView } from "./outage-view";

/**
 * Feeds `outageView`: while the engine is unreachable the console process (which answers locally)
 * is asked every 3 s for its engine list and for how long recent restarts took.
 */
export function useOutage(
  connection: Connection,
  phase: OutageInput["phase"],
  offlineSince: number | null,
  updatedAt: number | null,
  status: OutageInput["status"],
): OutageView {
  const offline = phase === "offline";
  const [list, setList] = useState<EngineList | null>(null);
  const [usualS, setUsualS] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const statusRef = useRef(status);
  statusRef.current = status;
  useEffect(() => {
    if (!offline) {
      setList(null);
      return;
    }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const [next, state] = await Promise.all([
          fetchEngines(connection, {
            signal: controller.signal,
            timeoutMs: 3000,
          }),
          fetchConsoleState(connection, controller.signal),
        ]);
        if (controller.signal.aborted) return;
        setList(next.mode === "auto" ? next : null);
        setUsualS(state?.outages?.medianDownS ?? null);
      } catch {
        if (!controller.signal.aborted) setList(null);
      }
      if (!controller.signal.aborted)
        timer = setTimeout(() => void tick(), 3000);
    };
    void tick();
    const clock = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
      clearInterval(clock);
    };
  }, [offline, connection]);
  return outageView({
    phase,
    offlineSince,
    updatedAt,
    status,
    list,
    usualS,
    now: offline ? now : Date.now(),
  });
}
