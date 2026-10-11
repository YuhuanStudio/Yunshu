import { useEffect, useRef, useState } from "react";
import { Button } from "@yuhuanowo/yunui";
import { StreamHealth, type StreamState } from "@yuhuanowo/yunui/patterns";
import type { Connection } from "./api";
import { t, useLocale } from "./i18n/index.ts";
import { number } from "./i18n/format";
import {
  IDLE_PROBE,
  probeRealtime,
  type RealtimeProbe,
} from "./realtime-probe";

/**
 * A Realtime connection test run by this browser: open the socket, wait for the engine's first
 * event, close. It measures the path from here to the engine, not the engine's own speech
 * latency: the engine reports no server-side realtime latency or buffer depth, so those rows are
 * an em dash with the reason, never a guessed number.
 */
export function RealtimeHealth({ connection }: { connection: Connection }) {
  useLocale();
  const [probe, setProbe] = useState<RealtimeProbe>(IDLE_PROBE);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const running = probe.state === "connecting" || probe.state === "open";
  const state: StreamState =
    probe.state === "closed" && probe.firstEventMs != null
      ? "open"
      : probe.state;
  const ms = (v: number | null) => (v == null ? null : number(v, 0));
  return (
    <StreamHealth
      className="mt-4"
      title={t("diagnostics.realtime.title")}
      state={state}
      stateLabel={
        probe.state === "idle"
          ? t("diagnostics.realtime.idle")
          : probe.state === "connecting" || probe.state === "open"
            ? t("diagnostics.realtime.running")
            : probe.firstEventMs != null
              ? t("diagnostics.realtime.ok")
              : t("diagnostics.realtime.failed", { reason: probe.error ?? "" })
      }
      action={
        <Button
          size="sm"
          variant="outline"
          disabled={running}
          data-testid="realtime-test"
          onClick={() =>
            void probeRealtime(connection, (p) => alive.current && setProbe(p))
          }
        >
          {t("diagnostics.realtime.run")}
        </Button>
      }
      rows={[
        {
          label: t("diagnostics.realtime.open"),
          value: ms(probe.openMs),
          unit: "ms",
          source: t("diagnostics.realtime.browser"),
          unknownTitle: t("diagnostics.realtime.notMeasured"),
        },
        {
          label: t("diagnostics.realtime.first"),
          value:
            probe.firstEventMs == null
              ? null
              : `${number(probe.firstEventMs, 0)}`,
          unit: probe.firstEventType ? `ms · ${probe.firstEventType}` : "ms",
          source: t("diagnostics.realtime.browser"),
          unknownTitle: t("diagnostics.realtime.notMeasured"),
        },
        {
          label: t("diagnostics.realtime.latency"),
          value: null,
          source: t("diagnostics.realtime.server"),
          unknownTitle: t("diagnostics.realtime.serverMissing"),
        },
        {
          label: t("diagnostics.realtime.buffer"),
          value: null,
          source: t("diagnostics.realtime.server"),
          unknownTitle: t("diagnostics.realtime.serverMissing"),
        },
      ]}
      note={t("diagnostics.realtime.note", {
        scheme: probe.scheme ?? "ws",
      })}
    />
  );
}
