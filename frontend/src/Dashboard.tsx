import { useEffect, useMemo, useState } from "react";
import {
  AnimatedNumber,
  Badge,
  Button,
  Card,
  EmptyState,
  IconButton,
  Progress,
  ScrollFade,
  SegmentedBar,
  SegmentedSelect,
  Sparkline,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  TimeSeriesChart,
  Tr,
} from "@yuhuanowo/yunui";
import {
  ArrowRight,
  Download,
  Pause,
  Play,
  RefreshCw,
  Server,
} from "lucide-react";
import {
  DetailList,
  DetailRow,
  HoverRow,
  PageHeader,
} from "@yuhuanowo/yunui/patterns";
import { ActivityPanel, LatencyPanel, PhasePanel } from "./AnalyticsPanels";
import { observationCsv, timeSeries } from "./analytics";
import type { RequestRow } from "./api";
import {
  clock,
  elapsed,
  modelLabel,
  number,
  phaseDot,
  Readout,
  supportsChat,
  type Engine,
} from "./ui";
const chartLabels = {
  emptyLabel: "等待第一筆採樣",
  unavailableLabel: "目前沒有可用的數值",
  missingValueLabel: "未回報",
  hiddenLabel: "序列已隱藏，可點選圖例重新顯示",
  legendLabel: "顯示或隱藏序列",
  keyboardHint: "使用左右方向鍵、Home 或 End 查看採樣",
};
const memorySeries = [
  { key: "active", label: "活躍配置", tone: "accent" as const },
  { key: "cache", label: "配置器快取", tone: "warning" as const, dashed: true },
];
const requestSeries = [
  { key: "requests", label: "全部", tone: "accent" as const },
  { key: "queued", label: "排隊", tone: "warning" as const, dashed: true },
  { key: "prefillRequests", label: "Prefill", tone: "info" as const },
  { key: "decodeRequests", label: "Decode", tone: "success" as const },
];
const rateSeries = {
  decode: [{ key: "decode", label: "Decode 平均", tone: "accent" as const }],
  prefill: [{ key: "prefill", label: "Prefill 平均", tone: "info" as const }],
  both: [
    { key: "decode", label: "Decode 平均", tone: "success" as const },
    { key: "prefill", label: "Prefill 平均", tone: "info" as const },
  ],
};
const formatNumber = (value: number) => number(value);
const formatCount = (value: number) => number(value, 0);
const phaseText: Record<string, string> = {
  queued: "排隊",
  starting: "啟動",
  prefill: "Prefill",
  decode: "Decode",
  running: "執行",
};
function RequestLane({ row }: { row: RequestRow }) {
  const phase = String(row.phase);
  const prompt = row.prompt_tokens ?? 0,
    cached = row.cached_tokens ?? 0;
  const progress =
    phase === "prefill" || phase === "starting"
      ? (row.percent ??
        (prompt > 0 && row.processed_tokens != null
          ? (row.processed_tokens / prompt) * 100
          : null))
      : null;
  return (
    <li className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2 py-3">
      <div className="flex min-w-0 items-center gap-2.5">
        <StatusIndicator
          status={phaseDot(phase)}
          pulse={phase === "decode" || phase === "prefill"}
        />
        <span className="text-sm font-medium">{phaseText[phase] ?? phase}</span>
        <span className="truncate font-mono text-xs text-muted-foreground">
          {row.request_id}
        </span>
        {row.model && (
          <span className="hidden truncate text-xs text-muted-foreground sm:inline">
            · {modelLabel(row.model)}
          </span>
        )}
      </div>
      <div className="flex items-center gap-4 text-xs tabular-nums text-muted-foreground">
        <span>
          {phase === "decode"
            ? `${number(row.completion_tokens, 0)} tok`
            : `${number(prompt, 0)} tok`}
        </span>
        <span className="w-20 text-right text-foreground">
          {row.tokens_per_second == null
            ? "—"
            : `${number(row.tokens_per_second)} tok/s`}
        </span>
        <span className="w-12 text-right">{elapsed(row.elapsed_s)}</span>
      </div>
      {progress != null && (
        <Progress
          className="col-span-2 h-1"
          value={Math.max(0, Math.min(100, progress))}
          label={`Prefill ${number(progress, 0)}%`}
        />
      )}
      {phase !== "prefill" && cached > 0 && prompt > 0 && (
        <p className="col-span-2 -mt-1 text-[11px] text-muted-foreground">
          前綴命中 {number(cached, 0)} / {number(prompt, 0)} tokens
        </p>
      )}
    </li>
  );
}

export function Dashboard({
  engine,
  navigate,
}: {
  engine: Engine;
  navigate: (page: string) => void;
}) {
  const [range, setRange] = useState("15m"),
    [metric, setMetric] = useState("decode"),
    [table, setTable] = useState(false),
    [activeX, setActiveX] = useState<number | null>(null);
  const status = engine.status,
    online = engine.phase === "online";
  const end = engine.updatedAt ?? Date.now(),
    start = end - (range === "5m" ? 300 : range === "15m" ? 900 : 3600) * 1000;
  const points = useMemo(
    () =>
      engine.history.filter((sample) => sample.at >= start && sample.at <= end),
    [engine.history, start, end],
  );
  const data = useMemo(() => timeSeries(points), [points]);
  const throughputSeries = rateSeries[metric as keyof typeof rateSeries];
  useEffect(() => {
    if (activeX !== null && !points.some((point) => point.at === activeX))
      setActiveX(null);
  }, [points, activeX]);
  const last = status?.last,
    cache =
      last && last.prompt_tokens > 0
        ? (last.cached_tokens / last.prompt_tokens) * 100
        : null;
  const activePoint =
    activeX == null ? null : points.find((sample) => sample.at === activeX);
  const items = status?.requests.items ?? [];
  const decodeTrend = points.flatMap((p) =>
    p.status.throughput.mean_decode_tps == null
      ? []
      : [p.status.throughput.mean_decode_tps],
  );
  const liveDecode =
    status?.throughput.live_decode_tps ?? status?.throughput.mean_decode_tps;
  const memory = status?.memory;
  const prefilling = items.find(
    (r) => r.phase === "prefill" || r.phase === "starting",
  );
  const prefillPct = prefilling
    ? (prefilling.percent ??
      ((prefilling.prompt_tokens ?? 0) > 0 &&
      prefilling.processed_tokens != null
        ? (prefilling.processed_tokens / (prefilling.prompt_tokens ?? 1)) * 100
        : null))
    : null;
  const loaded = (status?.models ?? []).filter((m) => m.loaded);
  const chooseRange = (next: string) => {
    setRange(next);
    setActiveX(null);
  };
  function exportData() {
    const url = URL.createObjectURL(
        new Blob([observationCsv(points)], { type: "text/csv;charset=utf-8" }),
      ),
      a = document.createElement("a");
    a.href = url;
    a.download = `yunshu-observations-${range}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  if (!status && engine.phase !== "connecting")
    return (
      <section
        className="mx-auto w-full max-w-6xl space-y-6"
        data-testid="overview"
      >
        <PageHeader
          title="引擎總覽"
          description="模型、請求與效能，都從你的本機引擎開始。"
        />
        <Card className="p-6 sm:p-10">
          <EmptyState
            icon={<Server size={30} strokeWidth={1.5} />}
            title="連接本機推理引擎"
            description="填入 Yunshu 服務位址與存取權杖，取得真實模型、資源和請求狀態。"
            action={
              <Button onClick={() => navigate("settings")}>
                開啟連線設定
                <ArrowRight size={14} />
              </Button>
            }
          />
          <ol className="mt-8 grid gap-6 border-t border-border/60 pt-6 sm:grid-cols-3">
            {[
              ["01", "連接服務", "使用現有的 Yunshu HTTP 服務"],
              ["02", "載入模型", "管理權重與閒置保留時間"],
              ["03", "觀察與驗證", "查看圖表、請求與推理結果"],
            ].map(([step, title, description]) => (
              <li key={step}>
                <span className="font-mono text-xs text-muted-foreground">
                  {step}
                </span>
                <h2 className="mt-2 text-sm font-medium">{title}</h2>
                <p className="mt-1 text-caption">{description}</p>
              </li>
            ))}
          </ol>
        </Card>
      </section>
    );
  return (
    <section
      className="mx-auto w-full max-w-6xl space-y-5"
      data-testid="overview"
    >
      <PageHeader
        title="引擎總覽"
        description="觀察這台 Mac 如何處理每一次推理。"
        actions={
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => engine.setPolling(!engine.polling)}
            >
              {engine.polling ? <Pause size={13} /> : <Play size={13} />}
              {engine.polling ? "暫停更新" : "恢復更新"}
            </Button>
            <IconButton
              icon={<RefreshCw size={14} />}
              label="更新"
              onClick={() => void engine.refresh()}
            />
            <Button
              size="sm"
              disabled={
                !online ||
                !status?.models.some(
                  (model) => model.loaded && supportsChat(model),
                )
              }
              onClick={() => navigate("playground")}
            >
              <Play size={13} />
              開始測試
            </Button>
          </div>
        }
      />
      <p className="-mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
        <StatusIndicator
          status={online ? "online" : "offline"}
          pulse={online && items.length > 0}
        >
          <span className="text-foreground">
            {online
              ? status?.state === "running"
                ? "運行中"
                : (status?.state ?? "已連線")
              : "未連線"}
          </span>
        </StatusIndicator>
        <span>Yunshu {status?.version ?? "—"}</span>
        <span>· 運行 {elapsed(status?.uptime_s)}</span>
        <span>
          ·{" "}
          {engine.updatedAt
            ? `更新於 ${clock(engine.updatedAt)}`
            : "等待服務回應"}
          {!engine.polling ? "（已暫停）" : ""}
        </span>
        {status?.load_error && (
          <span className="text-error">{status.load_error}</span>
        )}
      </p>

      {/* Live: what the engine is doing right now. */}
      <Card
        className="grid min-w-0 overflow-hidden p-0 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]"
        data-testid="live-panel"
      >
        <div className="flex min-w-0 flex-col justify-between gap-5 border-b border-border/60 p-5 sm:p-6 lg:border-b-0 lg:border-r">
          {prefilling ? (
            <div>
              <p className="text-xs text-muted-foreground">
                正在處理提示詞 · {prefilling.request_id}
              </p>
              <p className="mt-2 text-5xl font-semibold tracking-tight tabular-nums">
                {number(prefillPct, 0)}
                <span className="ml-1 text-2xl font-normal text-muted-foreground">
                  %
                </span>
              </p>
              <Progress
                className="mt-3 h-1.5"
                value={prefillPct ?? 0}
                label={`Prefill ${number(prefillPct, 0)}%`}
              />
              <p className="mt-2 text-xs tabular-nums text-muted-foreground">
                {number(prefilling.processed_tokens, 0)} /{" "}
                {number(prefilling.prompt_tokens, 0)} tokens
                {prefilling.tokens_per_second != null &&
                  ` · ${number(prefilling.tokens_per_second, 0)} tok/s`}
                {prefilling.eta_s != null &&
                  ` · 約 ${elapsed(prefilling.eta_s)} 後開始輸出`}
              </p>
            </div>
          ) : (
            <div>
              <p className="text-xs text-muted-foreground">
                {status?.throughput.live_decode_tps != null
                  ? "目前 Decode 總速率"
                  : "Decode · 近 5 分鐘平均（閒置）"}
              </p>
              <p
                className={
                  "mt-2 text-5xl font-semibold tracking-tight tabular-nums" +
                  (status?.throughput.live_decode_tps != null
                    ? ""
                    : " text-muted-foreground")
                }
              >
                {liveDecode == null ? (
                  "—"
                ) : (
                  <AnimatedNumber value={liveDecode} decimals={1} />
                )}
                <span className="ml-2 text-base font-normal text-muted-foreground">
                  tok/s
                </span>
              </p>
            </div>
          )}
          {decodeTrend.length > 1 ? (
            <Sparkline
              className="h-14 w-full"
              data={decodeTrend}
              tone="accent"
              area
              label="平均 Decode 速度趨勢"
            />
          ) : (
            <div className="flex h-14 items-end">
              <div className="h-px w-full bg-border" />
            </div>
          )}
          <div className="grid grid-cols-3 gap-4 border-t border-border/60 pt-4">
            <Readout
              label="Prefill 平均"
              value={number(status?.throughput.mean_prefill_tps, 0)}
              unit="tok/s"
            />
            <Readout
              label="首 Token 延遲"
              value={number(last?.ttft_ms, 0)}
              unit="ms"
              hint={
                last ? `最近一筆 · ${clock(last.t * 1000)}` : "尚無完成請求"
              }
            />
            <Readout
              label="前綴重用率"
              value={cache == null ? "—" : `${number(cache)}%`}
              hint={
                last
                  ? `${number(last.cached_tokens, 0)} / ${number(last.prompt_tokens, 0)} tokens`
                  : undefined
              }
            />
          </div>
        </div>
        <div className="flex min-w-0 flex-col p-5 sm:p-6">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-medium">
              進行中請求
              <span className="ml-2 text-muted-foreground tabular-nums">
                {number(status?.requests.active, 0)}
              </span>
            </h2>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>
                近 {number(status?.throughput.window_s ?? 60, 0)} 秒結束{" "}
                {number(status?.throughput.requests, 0)} 筆
              </span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigate("requests")}
              >
                全部
                <ArrowRight size={13} />
              </Button>
            </div>
          </div>
          {items.length ? (
            <ul className="mt-2 divide-y divide-border/60">
              {items.slice(0, 5).map((row) => (
                <RequestLane key={row.request_id} row={row} />
              ))}
            </ul>
          ) : (
            <EmptyState
              size="inline"
              className="flex-1"
              title="目前閒置"
              description={`沒有正在處理的請求。${
                memory?.cache_gb
                  ? ` 配置器快取 ${number(memory.cache_gb)} GB 會在閒置後歸還系統。`
                  : ""
              }`}
            />
          )}
        </div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Card className="min-w-0 p-5 sm:p-6" data-testid="throughput-panel">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-medium">吞吐觀測</h2>
              <p className="mt-1 text-xs text-muted-foreground">
                近 5 分鐘平均速度的時間變化 · tok/s
              </p>
            </div>
            <SegmentedSelect
              aria-label="吞吐指標"
              value={metric}
              onChange={setMetric}
              options={[
                { value: "decode", label: "Decode" },
                { value: "prefill", label: "Prefill" },
                { value: "both", label: "比較" },
              ]}
            />
          </div>
          <TimeSeriesChart
            {...chartLabels}
            data={data}
            series={throughputSeries}
            height={220}
            ariaLabel="吞吐速度時序圖，單位 tok/s"
            formatX={clock}
            formatY={formatNumber}
            maxGap={12000}
            activeX={activeX}
            onActiveXChange={setActiveX}
          />
          <Button
            className="mt-2"
            size="sm"
            variant="ghost"
            aria-expanded={table}
            onClick={() => setTable(!table)}
          >
            {table ? "收合數值" : "檢視採樣數值"}
          </Button>
          {table && (
            <ScrollFade className="mt-3 max-h-60 overflow-auto">
              <Table scrollLabel="吞吐採樣資料">
                <Thead>
                  <Tr>
                    <Th>時間</Th>
                    <Th>Decode</Th>
                    <Th>Prefill</Th>
                    <Th>Metal GB</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {points.map((p) => (
                    <Tr key={p.at}>
                      <Td>{clock(p.at)}</Td>
                      <Td>{number(p.status.throughput.mean_decode_tps)}</Td>
                      <Td>{number(p.status.throughput.mean_prefill_tps)}</Td>
                      <Td>{number(p.status.memory.active_gb)}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </ScrollFade>
          )}
        </Card>
        <Card className="min-w-0 p-5 sm:p-6" data-testid="memory-panel">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium">Metal 記憶體</h2>
            <Badge variant="outline">實體 {number(memory?.total_gb)} GB</Badge>
          </div>
          <p className="mt-4 text-3xl font-semibold tracking-tight tabular-nums">
            {number(memory?.active_gb)}
            <span className="ml-1.5 text-sm font-normal text-muted-foreground">
              / {number(memory?.total_gb)} GB 活躍配置
            </span>
          </p>
          <SegmentedBar
            className="mt-4"
            height={10}
            total={memory?.total_gb ?? undefined}
            label={`Metal 記憶體：活躍 ${number(memory?.active_gb)} GB，快取 ${number(memory?.cache_gb)} GB`}
            segments={[
              {
                value: memory?.active_gb ?? 0,
                tone: "accent",
                label: "活躍配置",
              },
              {
                value: memory?.cache_gb ?? 0,
                tone: "warning",
                label: "配置器快取",
              },
            ]}
            marks={
              memory?.peak_gb
                ? [{ value: memory.peak_gb, label: "本次峰值" }]
                : undefined
            }
            legend
            formatValue={(v) => `${number(v)} GB`}
          />
          <DetailList className="mt-4 border-t border-border/60 pt-4">
            <DetailRow
              label="本次峰值"
              value={`${number(memory?.peak_gb)} GB`}
            />
            <DetailRow
              label="已載入權重"
              value={
                loaded.length
                  ? `${number(loaded.reduce((s, m) => s + (m.size_gb ?? 0), 0))} GB · ${loaded.length} 個`
                  : "—"
              }
            />
          </DetailList>
          <TimeSeriesChart
            {...chartLabels}
            className="mt-4"
            data={data}
            series={memorySeries}
            height={120}
            ariaLabel="Metal 記憶體時序圖，單位 GB"
            formatX={clock}
            formatY={formatNumber}
            maxGap={12000}
            activeX={activeX}
            onActiveXChange={setActiveX}
          />
        </Card>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          本頁開啟後採樣 · {points.length} 筆 · 中斷期間不補資料
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <SegmentedSelect
            aria-label="觀測時間範圍"
            value={range}
            onChange={chooseRange}
            options={[
              { value: "5m", label: "5 分鐘" },
              { value: "15m", label: "15 分鐘" },
              { value: "1h", label: "1 小時" },
            ]}
          />
          <Button
            size="sm"
            variant="ghost"
            disabled={!points.length}
            onClick={exportData}
          >
            <Download size={13} />
            匯出觀測
          </Button>
        </div>
      </div>
      {activePoint && (
        <div
          className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/60 px-4 py-2 text-xs text-muted-foreground"
          role="status"
        >
          <span>
            選取 {clock(activePoint.at)} · {activePoint.status.requests.active}{" "}
            個活動請求 · {number(activePoint.status.memory.active_gb)} GB
          </span>
          <Button size="sm" variant="ghost" onClick={() => setActiveX(null)}>
            清除選取
          </Button>
        </div>
      )}
      <div className="grid min-w-0 gap-5 xl:grid-cols-2">
        <LatencyPanel history={points} />
        <PhasePanel engine={engine} navigate={navigate} />
      </div>
      <ActivityPanel
        history={points}
        start={start}
        end={end}
        onSelectTime={setActiveX}
      />
      <Card className="min-w-0 p-5 sm:p-6">
        <div className="mb-4">
          <h2 className="text-sm font-medium">請求並行趨勢</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            不同處理階段，共用請求數刻度
          </p>
        </div>
        <TimeSeriesChart
          {...chartLabels}
          data={data}
          series={requestSeries}
          height={180}
          ariaLabel="請求階段並行數時序圖"
          formatX={clock}
          formatY={formatCount}
          maxGap={12000}
          activeX={activeX}
          onActiveXChange={setActiveX}
        />
      </Card>
      <Card className="min-w-0 p-0">
        <div className="flex items-center justify-between px-5 pb-2 pt-5 sm:px-6">
          <h2 className="text-sm font-medium">模型</h2>
          <Button size="sm" variant="ghost" onClick={() => navigate("models")}>
            模型庫
            <ArrowRight size={13} />
          </Button>
        </div>
        <ul className="divide-y divide-border/60">
          {(status?.models ?? []).slice(0, 6).map((model) => (
            <li key={model.id} className="px-3 py-0.5 sm:px-4">
              <HoverRow
                onClick={() => navigate("models")}
                aria-label={`${modelLabel(model.id)}，開啟模型庫`}
                className="flex items-center gap-3 px-2 py-2.5"
              >
                <StatusIndicator
                  status={
                    model.loading ? "away" : model.loaded ? "online" : "neutral"
                  }
                  pulse={model.loading}
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium">
                  {modelLabel(model.id)}
                </span>
                <span className="hidden text-xs text-muted-foreground sm:inline">
                  {model.type}
                </span>
                <span className="w-20 text-right text-xs tabular-nums text-muted-foreground">
                  {number(model.size_gb)} GB
                </span>
                <span className="w-24 text-right text-xs text-muted-foreground">
                  {model.loading
                    ? "載入中"
                    : model.loaded
                      ? model.expires_in_s != null
                        ? `${elapsed(model.expires_in_s)} 後卸載`
                        : model.pinned
                          ? "固定保留"
                          : "已載入"
                      : "未載入"}
                </span>
              </HoverRow>
            </li>
          ))}
          {!status?.models.length && (
            <li className="px-5 py-5 text-sm text-muted-foreground sm:px-6">
              {status ? "服務尚未註冊模型" : "連線後顯示可用模型"}
            </li>
          )}
        </ul>
      </Card>
    </section>
  );
}
