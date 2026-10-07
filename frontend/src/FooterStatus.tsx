import { useMemo } from "react";
import { StatusPill, StatusPillBar } from "@yuhuanowo/yunui/patterns";
import type { Connection } from "./api";
import type { Engine } from "./ui";
import { footerPills, gpuBusyFraction } from "./footer-status";
import { t } from "./i18n/index.ts";
import { useMemoryLedger } from "./memory-api";

/** The status band: engine, what it does now, the machine. Current state only. */
export function FooterStatus({
  engine,
  connection,
}: {
  engine: Engine;
  connection: Connection;
}) {
  const online = engine.phase === "online";
  const ledger = useMemoryLedger(connection, online, 5000);
  const history = engine.history;
  const gpuBusy = useMemo(() => {
    const latest = history.at(-1);
    if (!latest) return null;
    // Compare with the sample about 9 s earlier (three polls) to smooth single ticks.
    const earlier =
      [...history].reverse().find((p) => latest.at - p.at >= 8_000) ??
      history.at(-2);
    return earlier && earlier !== latest
      ? gpuBusyFraction(earlier.status, latest.status)
      : null;
  }, [history]);
  const pills = footerPills({
    phase: engine.phase,
    errorStatus: engine.errorStatus,
    status: engine.status,
    gpuBusy: online ? gpuBusy : null,
    ledger: online ? ledger.data : null,
  });
  return (
    <StatusPillBar
      ariaLabel={t("shell.footer.ariaLabel")}
      className="shrink-0 px-4 lg:px-6"
    >
      {pills.map((p) => (
        <StatusPill
          key={p.key}
          className={p.key === "engine" ? "pill-sans" : undefined}
          label={p.label}
          value={p.value}
          valueMinCh={p.minCh}
          tone={p.tone}
          dot={p.dot}
          help={p.help}
        />
      ))}
    </StatusPillBar>
  );
}
