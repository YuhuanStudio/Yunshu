import { memoryPairText } from "./byte-format.ts";
import type { EngineStatus } from "./api.ts";
import { activity, phaseLabels } from "./engineView.ts";
import { offlineCause } from "./errors.ts";
import { has, t, tr } from "./i18n/index.ts";
import { fixed } from "./i18n/format.ts";
import type { EngineConnectionPhase } from "./useEngine.ts";
import type { MemoryLedgerData } from "./memory-api.ts";

/**
 * The footer shows what is true NOW: the engine, what it is doing, the machine.
 * Last-request figures belong to the Requests page. Each pill is two words plus a
 * tooltip sentence; a pill with nothing to say is not produced.
 */
export type PillTone = "neutral" | "success" | "warning" | "danger";
export interface FooterPill {
  key: string;
  label: string;
  value?: string;
  tone: PillTone;
  dot: boolean;
  help: string;
  /** Realistic width of the value in `ch`, so changing digits never move neighbours. */
  minCh?: number;
  /** A live number that glides between samples; when set the band draws it instead of `value`. */
  tween?: { value: number; digits: number; suffix: string };
}

const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

const n = (v: number, d = 0) => fixed(v, d);

export const MEMORY_WARN = 0.8;
export const MEMORY_DANGER = 0.92;

const pressureKey: Record<string, string> = {
  normal: "normal",
  warn: "warn",
  warning: "warn",
  critical: "critical",
};
export const pressureText = (level: string) =>
  pressureKey[level]
    ? tr(`shell.footer.pressure.${pressureKey[level]}`)
    : level;

export function uptimeText(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d > 0) return t("shell.footer.uptime.dh", { d, h });
  if (h > 0) return t("shell.footer.uptime.hm", { h, m });
  if (m > 0) return t("shell.footer.uptime.m", { m });
  return t("shell.footer.uptime.s", { s });
}

type BusyMeter = { busy_seconds?: unknown; uptime_seconds?: unknown };
const slices = (gpu: unknown, id: string): BusyMeter | null => {
  const entry = (gpu as Record<string, unknown> | undefined)?.[id];
  const m = (entry as { slices?: unknown } | null | undefined)?.slices;
  return m && typeof m === "object" ? (m as BusyMeter) : null;
};

/**
 * GPU busy fraction between two status samples, from the engine's cumulative
 * busy seconds (the status `gpu` field). The busiest loaded engine wins; null
 * when the engine does not account busy time or the samples are not comparable.
 */
export function gpuBusyFraction(
  earlier: EngineStatus | null | undefined,
  later: EngineStatus | null | undefined,
): number | null {
  if (!earlier?.gpu || !later?.gpu) return null;
  let best: number | null = null;
  for (const id of Object.keys(later.gpu)) {
    const a = slices(earlier.gpu, id);
    const b = slices(later.gpu, id);
    if (!a || !b) continue;
    if (
      !finite(a.busy_seconds) ||
      !finite(b.busy_seconds) ||
      !finite(a.uptime_seconds) ||
      !finite(b.uptime_seconds)
    )
      continue;
    const wall = b.uptime_seconds - a.uptime_seconds;
    if (wall <= 0.5) continue;
    const f = Math.min(
      Math.max((b.busy_seconds - a.busy_seconds) / wall, 0),
      1,
    );
    best = best == null ? f : Math.max(best, f);
  }
  return best;
}

export interface FooterInput {
  phase: EngineConnectionPhase;
  errorStatus: number | null;
  status: EngineStatus | null;
  /** Busy fraction 0-1 over the last few polls, or null when unknown. */
  gpuBusy: number | null;
  ledger: MemoryLedgerData | null;
}

export function footerPills(input: FooterInput): FooterPill[] {
  const { phase, status } = input;
  const pills: FooterPill[] = [];
  if (phase !== "online" || !status) {
    const cause = offlineCause(phase, input.errorStatus);
    const text =
      phase === "connecting"
        ? t("shell.footer.engine.connecting")
        : phase === "unauthorized"
          ? t("shell.footer.engine.unauthorized")
          : t("shell.footer.engine.offline");
    pills.push({
      key: "engine",
      label: text,
      value: phase === "offline" ? cause.short : undefined,
      tone:
        phase === "connecting"
          ? "neutral"
          : phase === "unauthorized"
            ? "warning"
            : "danger",
      dot: true,
      help: t("shell.footer.engine.help", { state: cause.short }),
    });
    return pills;
  }

  const loaded = status.models.filter((m) => m.loaded);
  const first = loaded[0]?.id.split("/").filter(Boolean).at(-1);
  pills.push({
    key: "engine",
    label: t("shell.footer.engine.running"),
    value: first
      ? loaded.length > 1
        ? `${first} +${loaded.length - 1}`
        : first
      : undefined,
    tone: "success",
    dot: true,
    help:
      t("shell.footer.engine.okHelp", {
        version: status.version,
        uptime: uptimeText(status.uptime_s),
      }) +
      (loaded.length
        ? t("shell.footer.engine.loadedList", {
            models: loaded
              .map((m) => m.id.split("/").filter(Boolean).at(-1))
              .join(t("shell.footer.engine.listSep")),
          })
        : t("shell.footer.engine.noModel")),
  });

  const mem = status.memory;
  // One source per view: the engine status (3 s) feeds the sidebar, the overview
  // and this band. The host ledger (5 s) only adds pressure and swap, never a second GB figure.
  const total = finite(mem.total_gb) ? mem.total_gb : input.ledger?.total_gb;
  const active = finite(mem.active_gb)
    ? mem.active_gb
    : input.ledger?.mlx.active_gb;
  if (finite(active) && finite(total) && total > 0) {
    const level = input.ledger?.host.pressure_level ?? null;
    const usage = active / total;
    const tone: PillTone =
      level === "critical" || usage > MEMORY_DANGER
        ? "danger"
        : level === "warn" || level === "warning" || usage > MEMORY_WARN
          ? "warning"
          : "neutral";
    const levelText = level ? pressureText(level) : null;
    pills.push({
      key: "memory",
      label: t("shell.footer.memory.label"),
      value: memoryPairText(active, total),
      tone,
      dot: true,
      minCh: 14,
      help:
        t("shell.footer.memory.help", {
          active: n(active, 1),
          total: n(total, 0),
          pct: n(usage * 100),
        }) +
        (levelText
          ? t("shell.footer.memory.pressure", { level: levelText })
          : "") +
        t("shell.footer.sentenceEnd") +
        t("shell.footer.memory.source"),
    });
  }
  const a = activity(status);
  let value: string | undefined;
  const tween =
    a.phase === "decode" && a.decodeNow != null
      ? { value: a.decodeNow, digits: 0, suffix: " tok/s" } // i18n-ignore
      : undefined;
  if (a.phase === "decode")
    value = a.decodeNow != null ? `${n(a.decodeNow)} tok/s` : "—";
  else if (a.phase === "prefill")
    value =
      a.progress != null
        ? `${n(a.progress)}%`
        : a.preparing
          ? t("shell.engine.phase.starting")
          : "—";
  const loadingModel =
    a.phase === "idle" ? status.models.find((m) => m.loading) : undefined;
  if (loadingModel)
    value =
      loadingModel.id.split("/").filter(Boolean).at(-1) ?? loadingModel.id;
  pills.push({
    key: "now",
    label: loadingModel ? t("shell.engine.live.loading") : phaseLabels[a.phase],
    value: value ?? "\u00a0",
    // The same state-to-colour map as the top pill: decode green, other work amber, idle no dot.
    tone:
      a.phase === "decode"
        ? "success"
        : a.phase === "idle" && !loadingModel
          ? "neutral"
          : "warning",
    dot: a.phase !== "idle" || !!loadingModel,
    // the value area is always as wide as "1234 tok/s", so the pills after it never move
    minCh: 9,
    tween,
    help:
      a.phase === "idle"
        ? t("shell.footer.now.help.idle")
        : a.phase === "decode"
          ? t("shell.footer.now.help.decode")
          : a.phase === "prefill"
            ? t("shell.footer.now.help.prefill")
            : t("shell.footer.now.help.queued"),
  });
  // Always present and before the pills that come and go (swap, GPU), so nothing after it moves.
  pills.push({
    key: "load",
    label: t("shell.footer.load.label"),
    value:
      a.counts.queued > 0
        ? t("shell.footer.load.both", {
            active: a.counts.active,
            queued: a.counts.queued,
          })
        : t("shell.footer.load.active", { active: a.counts.active }),
    tone: "neutral",
    dot: false,
    minCh: 14,
    help: t("shell.footer.load.help", {
      active: a.counts.active,
      queued: a.counts.queued,
    }),
  });
  const swap = input.ledger?.host.swap_used_gb;
  if (finite(swap) && swap >= 0.05)
    pills.push({
      key: "swap",
      label: t("shell.footer.swap.label"),
      value: `${n(swap, 1)} GB`,
      tone: "neutral",
      dot: false,
      help: t("shell.footer.swap.help"),
    });
  if (input.gpuBusy != null)
    pills.push({
      key: "gpu",
      label: t("shell.footer.gpu.label"),
      value: `${n(input.gpuBusy * 100)}%`,
      tone: "neutral",
      dot: false,
      minCh: 4,
      help: t("shell.footer.gpu.help"),
    });

  return pills;
}
