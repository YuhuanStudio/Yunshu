import { useEffect } from "react";
import type { Connection } from "./api";
import {
  appendSample,
  fetchHost,
  type HostSample,
  type HostSnapshot,
} from "./host-api";
import { connectionScope, useScopedState } from "./scoped-state";

export type HostState = {
  /** null until the first answer; "unsupported" when the engine has no host route / telemetry section. */
  host: HostSnapshot | "unsupported" | null;
  history: HostSample[];
  /** Local time of the last successful read, for the "how old" label. */
  readAt: number | null;
};

const EMPTY: HostState = { host: null, history: [], readAt: null };

/**
 * Polls GET /v1/yunshu/host every 2 s while the tab is visible (the sampler runs at 1 Hz; the OS
 * readings beside it are cached 15 s engine-side). History is keyed by the engine's `sampled_at`
 * and bounded to 60 samples; a failed poll keeps the last data. Stops once unsupported.
 */
export function useHostTelemetry(
  connection: Connection,
  enabled: boolean,
  intervalMs = 2000,
): HostState {
  const [state, setState] = useScopedState<HostState>(
    connectionScope(connection),
    EMPTY,
  );
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      if (document.visibilityState === "visible") {
        try {
          const r = await fetchHost(connection, controller.signal);
          if (controller.signal.aborted) return;
          if (r.kind === "unsupported" || !r.host.telemetry) {
            setState({ host: "unsupported", history: [], readAt: null });
            return;
          }
          const telemetry = r.host.telemetry;
          setState((s) => ({
            host: r.host,
            history: appendSample(s.history, telemetry),
            readAt: Date.now(),
          }));
        } catch {
          // Keep the last data; the age label tells the truth.
        }
      }
      timer = setTimeout(() => void tick(), intervalMs);
    };
    void tick();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, enabled, intervalMs]);
  return state;
}
