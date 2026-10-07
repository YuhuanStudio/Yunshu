import { TraceTimeline } from "@yuhuanowo/yunui/patterns";
import type { Row } from "./RequestTrace";
import { number, Readout } from "./ui";

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export const formatMs = (ms: number) =>
  ms >= 10_000
    ? `${number(ms / 1000, 1)} s`
    : ms >= 1000
      ? `${number(ms / 1000, 2)} s`
      : `${number(ms, ms < 100 ? 1 : 0)} ms`;

/** Wall-clock pieces of one finished request, all measured by the server; null when unreported. */
export function breakdown(row: Row) {
  const o = row.offsets_ms ?? {},
    admit = num(o.admit),
    first = num(o.first_token),
    last = num(o.last_token),
    done = num(o.done);
  return {
    queue: admit ?? num(row.queue_wait_ms),
    ttft: num(row.ttft_ms) ?? first,
    decode:
      first != null && (last ?? done) != null ? (last ?? done)! - first : null,
    total: done,
    admit,
    first,
    last,
    done,
  };
}

/** Time-proportional queue / prefill / decode lanes from the server's `offsets_ms`. */
export function RequestTimeline({ row }: { row: Row }) {
  const b = breakdown(row),
    reload = num(row.cache?.reload_ms);
  const end = b.done ?? b.last ?? b.first ?? b.admit;
  if (end == null || end <= 0)
    return (
      <p className="text-xs text-muted-foreground">
        伺服器沒有回報各階段時間點，無法繪製時間軸。
      </p>
    );
  const spans = [];
  if (b.admit != null && b.admit > 0)
    spans.push({
      id: "queue",
      track: "t",
      start: 0,
      end: b.admit,
      label: "排隊",
      tone: "neutral" as const,
    });
  if (b.admit != null && b.first != null && b.first > b.admit)
    spans.push({
      id: "prefill",
      track: "t",
      start: b.admit,
      end: b.first,
      label: "預填",
      tone: "info" as const,
      detail: (
        <p className="text-xs">
          命中 {number(row.cached_tokens, 0)} token · 新增{" "}
          {number(
            Math.max((row.prompt_tokens ?? 0) - (row.cached_tokens ?? 0), 0),
            0,
          )}{" "}
          token
          {reload != null &&
            ` · 前綴載入 ${formatMs(reload)}（${row.cache?.tier ?? "—"}）`}
        </p>
      ),
    });
  const decodeEnd = b.last ?? b.done;
  if (b.first != null && decodeEnd != null && decodeEnd > b.first)
    spans.push({
      id: "decode",
      track: "t",
      start: b.first,
      end: decodeEnd,
      label: "解碼",
      tone: "success" as const,
      detail: (
        <p className="text-xs">
          輸出 {number(row.completion_tokens, 0)} token ·{" "}
          {number(row.decode_tps)} tok/s
        </p>
      ),
    });
  return (
    <TraceTimeline
      label={`請求 ${row.id} 時間軸`}
      tracks={[{ id: "t", label: "請求" }]}
      spans={spans}
      markers={
        b.first != null ? [{ id: "ft", at: b.first, label: "首 token" }] : []
      }
      duration={end}
      formatTime={formatMs}
      labels={{
        view: "檢視",
        timeline: "時間軸",
        table: "表格",
        track: "軌道",
        span: "階段",
        start: "開始",
        end: "結束",
        duration: "耗時",
        marker: "標記",
        running: "進行中",
        expand: "展開詳情",
      }}
    />
  );
}

/** Header numbers: queue, TTFT, decode, total; cached vs new tokens; spec acceptance. */
export function RequestBreakdown({ row }: { row: Row }) {
  const b = breakdown(row),
    prompt = row.prompt_tokens ?? null,
    cached = row.cached_tokens ?? 0,
    spec = row.speculative;
  const ms = (v: number | null) => (v == null ? "—" : formatMs(v));
  return (
    <div className="space-y-4">
      <div className="grid min-h-[3.5rem] grid-cols-2 gap-x-5 gap-y-4">
        <Readout label="排隊" value={ms(b.queue)} />
        <Readout label="首 token 延遲 (TTFT)" value={ms(b.ttft)} />
        <Readout label="解碼" value={ms(b.decode)} />
        <Readout label="總計" value={ms(b.total)} />
      </div>
      <div className="grid min-h-[3.5rem] grid-cols-2 gap-x-5 gap-y-4">
        <Readout
          label="命中 token"
          value={number(prompt == null ? null : cached, 0)}
        />
        <Readout
          label="新增 token"
          value={number(
            prompt == null ? null : Math.max(prompt - cached, 0),
            0,
          )}
        />
        <Readout label="輸出 token" value={number(row.completion_tokens, 0)} />
        <Readout
          label="推測接受率"
          value={
            spec?.acceptance_rate == null
              ? "—"
              : `${number(spec.acceptance_rate * 100, 0)}%`
          }
          hint={spec ? undefined : "未啟用推測解碼"}
        />
      </div>
    </div>
  );
}
