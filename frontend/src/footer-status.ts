import type { EngineStatus } from "./api.ts";
import { activity, phaseLabels } from "./engineView.ts";
import { offlineCause } from "./errors.ts";
import type { EngineConnectionPhase } from "./useEngine.ts";
import type { MemoryLedgerData } from "./memory-api.ts";

/**
 * The footer shows what is true NOW: the engine, what it is doing, the machine.
 * Last-request figures belong to the Requests page. Each pill is two words plus a
 * tooltip sentence; a pill with nothing to say is not produced.
 */
export type PillTone = "neutral" | "warning" | "danger";
export interface FooterPill {
  key: string;
  label: string;
  value?: string;
  tone: PillTone;
  dot: boolean;
  help: string;
  /** Realistic width of the value in `ch`, so changing digits never move neighbours. */
  minCh?: number;
}

const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

const n = (v: number, d = 0) => v.toFixed(d);

export const MEMORY_WARN = 0.8;
export const MEMORY_DANGER = 0.92;

const pressureText: Record<string, string> = {
  normal: "正常",
  warn: "警告",
  warning: "警告",
  critical: "嚴重",
};

export function uptimeText(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d} 天 ${h} 小時`;
  if (h > 0) return `${h} 小時 ${m} 分`;
  if (m > 0) return `${m} 分`;
  return `${s} 秒`;
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
        ? "連線中"
        : phase === "unauthorized"
          ? "未授權"
          : "離線";
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
      help: `引擎目前：${cause.short}。`,
    });
    return pills;
  }

  const loaded = status.models.filter((m) => m.loaded);
  const first = loaded[0]?.id.split("/").filter(Boolean).at(-1);
  pills.push({
    key: "engine",
    label: "運作中",
    value: first
      ? loaded.length > 1
        ? `${first} +${loaded.length - 1}`
        : first
      : undefined,
    tone: "neutral",
    dot: true,
    help: `引擎連線正常 · yunshu ${status.version} · 已運作 ${uptimeText(status.uptime_s)}${
      loaded.length
        ? ` · 已載入 ${loaded.map((m) => m.id.split("/").filter(Boolean).at(-1)).join("、")}`
        : " · 尚未載入模型"
    }`,
  });

  const mem = status.memory;
  const total = input.ledger?.total_gb ?? mem.total_gb;
  const active = input.ledger?.mlx.active_gb ?? mem.active_gb;
  if (finite(active) && finite(total) && total > 0) {
    const level = input.ledger?.host.pressure_level ?? null;
    const usage = active / total;
    const tone: PillTone =
      level === "critical" || usage > MEMORY_DANGER
        ? "danger"
        : level === "warn" || level === "warning" || usage > MEMORY_WARN
          ? "warning"
          : "neutral";
    const levelText = level ? (pressureText[level] ?? level) : null;
    pills.push({
      key: "memory",
      label: "記憶體",
      value: `${n(active, 1)}/${n(total, 0)} GB`,
      tone,
      dot: true,
      minCh: 12,
      help: `MLX 使用 ${n(active, 1)} GB，統一記憶體共 ${n(total, 0)} GB（${n(usage * 100)}%）${
        levelText ? `；系統記憶體壓力：${levelText}` : ""
      }。`,
    });
  }
  const a = activity(status);
  let value: string | undefined;
  if (a.phase === "decode")
    value = a.decodeNow != null ? `${n(a.decodeNow)} tok/s` : "—";
  else if (a.phase === "prefill")
    value =
      a.progress != null ? `${n(a.progress)}%` : a.preparing ? "準備中" : "—";
  pills.push({
    key: "now",
    label: phaseLabels[a.phase],
    value,
    tone: "neutral",
    dot: false,
    minCh: a.phase === "decode" ? 9 : a.phase === "prefill" ? 4 : undefined,
    help:
      a.phase === "idle"
        ? "引擎目前沒有進行中的請求。"
        : a.phase === "decode"
          ? "目前所有解碼中請求的即時合計速度。"
          : a.phase === "prefill"
            ? "正在讀取輸入（預填），百分比為首個預填請求的進度。"
            : "請求正在排隊等候。",
  });
  const swap = input.ledger?.host.swap_used_gb;
  if (finite(swap) && swap >= 0.05)
    pills.push({
      key: "swap",
      label: "交換",
      value: `${n(swap, 1)} GB`,
      tone: "warning",
      dot: true,
      help: "系統正在使用磁碟交換空間；推論可能變慢。",
    });
  if (input.gpuBusy != null)
    pills.push({
      key: "gpu",
      label: "GPU 忙碌",
      value: `${n(input.gpuBusy * 100)}%`,
      tone: "neutral",
      dot: false,
      minCh: 4,
      help: "最近幾次狀態更新之間，GPU 執行推論的時間占比。",
    });
  if (a.counts.active > 0 || a.counts.queued > 0)
    pills.push({
      key: "load",
      label: "請求",
      value:
        a.counts.queued > 0
          ? `${a.counts.active} 進行 ${a.counts.queued} 排隊`
          : `${a.counts.active} 進行`,
      tone: "neutral",
      dot: false,
      help: `進行中 ${a.counts.active} 個，排隊 ${a.counts.queued} 個。`,
    });

  return pills;
}
