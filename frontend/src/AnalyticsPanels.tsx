import { useMemo, useState } from "react";
import {
  Badge,
  BarChart,
  Button,
  Card,
  DonutChart,
  EmptyState,
  Heatmap,
  Sheet,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { ArrowRight } from "lucide-react";
import {
  activityHeatmap,
  latencyDistribution,
  observedRequests,
  percentile,
  phaseDistribution,
  type LatencyBucket,
} from "./analytics";
import { clock, elapsed, number, type Engine } from "./ui";
import type { EngineHistoryPoint } from "./useEngine";

export function PhasePanel({
  engine,
  navigate,
}: {
  engine: Engine;
  navigate: (page: string) => void;
}) {
  const [phase, setPhase] = useState<string | null>(null);
  const data = phaseDistribution(engine.status);
  const active = engine.status?.requests.items ?? [];
  const selected = phase ? active.filter((row) => row.phase === phase) : active;
  return (
    <Card className="min-w-0 p-5 sm:p-6" data-testid="phase-panel">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <h2 className="heading-md">請求階段分布</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            點選圖例，查看目前正在處理的工作
          </p>
        </div>
        <Badge variant="outline">即時</Badge>
      </div>
      <DonutChart
        monochrome
        data={data}
        size={154}
        ariaLabel="目前請求階段"
        emptyLabel={engine.status ? "目前沒有活動請求" : "尚未取得請求"}
        unavailableLabel="未回報"
        center={
          <div>
            <strong className="block text-2xl">
              {number(engine.status?.requests.active, 0)}
            </strong>
            <span className="text-[10px] text-muted-foreground">活動請求</span>
          </div>
        }
        onSelect={(datum) =>
          setPhase((value) => (value === datum.id ? null : datum.id))
        }
      />
      {phase && (
        <Button
          size="sm"
          variant="ghost"
          className="mt-3"
          onClick={() => setPhase(null)}
        >
          清除階段篩選
        </Button>
      )}
      <div className="mt-4 divide-y divide-border/60 border-t border-border/60">
        {selected.slice(0, 3).map((row) => (
          <div
            key={row.request_id}
            className="flex justify-between gap-3 py-2.5 text-xs"
          >
            <span className="min-w-0 truncate font-mono">{row.request_id}</span>
            <span className="shrink-0 text-muted-foreground">
              {elapsed(row.elapsed_s)}
            </span>
          </div>
        ))}
        {!selected.length && (
          <p className="py-3 text-xs text-muted-foreground">
            {engine.status
              ? "目前沒有符合這個階段的請求"
              : "連線後顯示工作清單"}
          </p>
        )}
      </div>
      <Button
        variant="ghost"
        size="sm"
        className="mt-2"
        onClick={() => navigate("requests")}
      >
        開啟請求工作區
        <ArrowRight size={13} />
      </Button>
    </Card>
  );
}

export function LatencyPanel({
  history,
}: {
  history: readonly EngineHistoryPoint[];
}) {
  const records = useMemo(() => observedRequests(history), [history]);
  const bins = useMemo(() => latencyDistribution(records), [records]);
  const [selection, setSelection] = useState<LatencyBucket | null>(null);
  const valid = records.filter(
    (row) =>
      row.ttft_ms != null && Number.isFinite(row.ttft_ms) && row.ttft_ms >= 0,
  );
  const selected = selection
    ? valid.filter(
        (row) => row.ttft_ms! >= selection.min && row.ttft_ms! < selection.max,
      )
    : [];
  return (
    <Card className="min-w-0 p-5 sm:p-6" data-testid="latency-panel">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="heading-md">首 Token 延遲分布</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            點選長條查看請求 · 單位 ms
          </p>
        </div>
        <Badge variant="outline">{valid.length} 筆已觀測</Badge>
      </div>
      <div className="mb-4 flex gap-6">
        <div>
          <p className="text-[10px] text-muted-foreground">P50</p>
          <p className="mt-1 font-mono text-lg">
            {number(
              percentile(
                valid.map((row) => row.ttft_ms),
                0.5,
              ),
              0,
            )}{" "}
            <span className="text-xs">ms</span>
          </p>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground">P95</p>
          <p className="mt-1 font-mono text-lg">
            {number(
              percentile(
                valid.map((row) => row.ttft_ms),
                0.95,
              ),
              0,
            )}{" "}
            <span className="text-xs">ms</span>
          </p>
        </div>
      </div>
      <BarChart
        data={bins.map((bin) => ({ ...bin, tone: "neutral" as const }))}
        height={185}
        ariaLabel="已觀測首 Token 延遲分布"
        emptyLabel="尚未觀測到帶有延遲資料的請求"
        unavailableLabel="未回報"
        formatValue={(v) => `${v} 筆`}
        onSelect={(datum) =>
          setSelection(bins.find((bin) => bin.id === datum.id) ?? null)
        }
      />
      <p className="mt-4 text-[11px] leading-5 text-muted-foreground">
        採樣只能取得服務最新一筆結束記錄；已按 request ID
        去重，這不是完整流量的延遲統計。
      </p>
      <Sheet
        open={!!selection}
        onClose={() => setSelection(null)}
        title={`延遲 ${selection?.label ?? ""} ms`}
        closeLabel="關閉延遲明細"
      >
        <p className="mb-4 text-xs text-muted-foreground">
          目前觀測範圍內，共 {selected.length} 筆請求。
        </p>
        {selected.length ? (
          <Table scrollLabel="延遲區間請求">
            <Thead>
              <Tr>
                <Th>Request</Th>
                <Th>TTFT</Th>
                <Th>重用 tokens</Th>
              </Tr>
            </Thead>
            <Tbody>
              {selected.map((row) => (
                <Tr key={row.request_id}>
                  <Td>
                    <span className="block max-w-36 truncate font-mono text-[11px]">
                      {row.request_id}
                    </span>
                  </Td>
                  <Td>{number(row.ttft_ms, 0)} ms</Td>
                  <Td>{number(row.cached_tokens, 0)}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        ) : (
          <EmptyState
            size="inline"
            title="這個區間沒有觀測記錄"
            description="可關閉面板，選擇另一個延遲區間。"
          />
        )}
      </Sheet>
    </Card>
  );
}

export function ActivityPanel({
  history,
  start,
  end,
  onSelectTime,
}: {
  history: readonly EngineHistoryPoint[];
  start: number;
  end: number;
  onSelectTime: (at: number | null) => void;
}) {
  const heat = useMemo(
    () => activityHeatmap(history, start, end),
    [history, start, end],
  );
  const [selection, setSelection] = useState<{
    row: number;
    column: number;
  } | null>(null);
  const columns = heat.starts.map(clock);
  const select = (row: number, column: number) => {
    setSelection({ row, column });
    const candidates = history.filter(
      (sample) =>
        sample.at >= heat.starts[column] &&
        sample.at <=
          (column === heat.starts.length - 1
            ? heat.ends[column]
            : heat.ends[column] - 1),
    );
    const value = (sample: EngineHistoryPoint) =>
      [
        sample.status.requests.active,
        sample.status.requests.queued,
        sample.status.requests.prefill,
        sample.status.requests.decode,
      ][row];
    const peak = candidates.reduce<EngineHistoryPoint | null>(
      (best, next) => (!best || value(next) > value(best) ? next : best),
      null,
    );
    onSelectTime(peak?.at ?? null);
  };
  return (
    <Card className="min-w-0 p-5 sm:p-6" data-testid="activity-panel">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="heading-md">請求活動熱圖</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            每個區間的已採樣峰值 · 點選格子，聯動時序圖游標
          </p>
        </div>
        <Badge variant="outline">{history.length} 次採樣</Badge>
      </div>
      <Heatmap
        rows={heat.rows}
        columns={columns}
        data={heat.data}
        ariaLabel="請求階段活動熱圖"
        unavailableLabel="—"
        emptyLabel="沒有採樣"
        tone="neutral"
        formatValue={(v) => String(v)}
        onSelect={select}
      />
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <p>0 代表已觀測到閒置；— 代表沒有採樣。色階表示區間峰值。</p>
        {selection && (
          <p role="status">
            {clock(heat.starts[selection.column])}–
            {clock(heat.ends[selection.column])} ·{" "}
            {heat.coverage[selection.column]} 次觀測 · 峰值{" "}
            {number(heat.data[selection.row][selection.column], 0)}
          </p>
        )}
      </div>
    </Card>
  );
}
