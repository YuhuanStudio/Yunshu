import { useEffect, useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Gauge,
  SegmentedSelect,
  Sparkline,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  TimeSeriesChart,
  Tr,
} from "@yuhuanowo/yunui";
import { PageHeader, StatCard } from "@yuhuanowo/yunui/patterns";
import {
  Activity,
  ArrowRight,
  Clock3,
  Database,
  Download,
  Pause,
  Play,
  RefreshCw,
  Server,
  Zap,
} from "lucide-react";
import { ActivityPanel, LatencyPanel, PhasePanel } from "./AnalyticsPanels";
import { observationCsv, timeSeries } from "./analytics";
import {
  clock,
  elapsed,
  modelLabel,
  number,
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
      <section className="w-full max-w-6xl space-y-6" data-testid="overview">
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
          <div className="mt-8 grid gap-6 border-t border-border/60 pt-6 sm:grid-cols-3">
            {[
              ["01", "連接服務", "使用現有的 Yunshu HTTP 服務"],
              ["02", "載入模型", "管理權重與閒置保留時間"],
              ["03", "觀察與驗證", "查看圖表、請求與推理結果"],
            ].map(([step, title, description]) => (
              <div key={step}>
                <span className="font-mono text-xs text-muted-foreground">
                  {step}
                </span>
                <h2 className="mt-2 text-sm font-medium">{title}</h2>
                <p className="mt-1 text-caption">{description}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>
    );
  return (
    <section className="w-full max-w-6xl space-y-6" data-testid="overview">
      <PageHeader
        title="引擎總覽"
        description="觀察這台 Mac 如何處理每一次推理。"
        actions={
          <div className="flex flex-wrap gap-2">
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
            <Button
              size="sm"
              variant="secondary"
              onClick={() => engine.setPolling(!engine.polling)}
            >
              {engine.polling ? <Pause size={13} /> : <Play size={13} />}
              {engine.polling ? "暫停更新" : "恢復更新"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void engine.refresh()}
            >
              <RefreshCw size={13} />
              更新
            </Button>
          </div>
        }
      />
      <Card className="flex flex-wrap items-center justify-between gap-3 bg-muted/20 px-4 py-3 text-xs text-muted-foreground shadow-none">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={online ? "success" : "secondary"}>
            {online
              ? status?.state === "running"
                ? "運行中"
                : (status?.state ?? "已連線")
              : "未連線"}
          </Badge>
          <span>Yunshu {status?.version ?? "—"}</span>
          <span>· 運行 {elapsed(status?.uptime_s)}</span>
          {status?.load_error && (
            <span className="text-error">{status.load_error}</span>
          )}
        </div>
        <span>
          {engine.updatedAt
            ? `更新於 ${clock(engine.updatedAt)}`
            : "等待服務回應"}
          {!engine.polling ? " · 已暫停" : ""}
        </span>
      </Card>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          valueFirst
          icon={Activity}
          label="進行中請求"
          value={number(status?.requests.active, 0)}
          subtext={
            <div className="flex flex-wrap items-end justify-between gap-2">
              <span>
                近 {number(status?.throughput.window_s ?? 60, 0)} 秒結束{" "}
                {number(status?.throughput.requests, 0)} 筆
              </span>
              <Sparkline
                className="w-20"
                data={points.map((p) => p.status.requests.active)}
                label="活動請求趨勢"
              />
            </div>
          }
        />
        <StatCard
          valueFirst
          icon={Zap}
          label="Decode · 近 5 分鐘平均"
          value={
            <>
              {number(status?.throughput.mean_decode_tps)}
              <span className="ml-1 text-xs font-normal">tok/s</span>
            </>
          }
          subtext={
            <div className="flex flex-wrap items-end justify-between gap-2">
              <span>
                目前總速率 {number(status?.throughput.live_decode_tps)}
              </span>
              <Sparkline
                className="w-20"
                tone="success"
                data={points.flatMap((p) =>
                  p.status.throughput.mean_decode_tps == null
                    ? []
                    : [p.status.throughput.mean_decode_tps],
                )}
                label="平均 Decode 速度"
              />
            </div>
          }
        />
        <StatCard
          valueFirst
          icon={Clock3}
          label="首 Token 延遲"
          value={
            <>
              {number(last?.ttft_ms, 0)}
              <span className="ml-1 text-xs font-normal">ms</span>
            </>
          }
          subtext={
            last ? `最近一筆 · ${clock(last.t * 1000)}` : "等待第一筆結束的請求"
          }
        />
        <StatCard
          valueFirst
          icon={Database}
          label="前綴重用率"
          value={cache == null ? "—" : `${number(cache)}%`}
          subtext={
            last
              ? `最近一筆 · ${number(last.cached_tokens, 0)} / ${number(last.prompt_tokens, 0)} tokens`
              : "由真實請求的快取 token 計算"
          }
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs text-muted-foreground">
            本頁開啟後採樣 · {points.length} 筆 · 中斷期間不補資料
          </p>
          <p className="mt-1 text-[10px] text-muted-foreground">
            圖表間隔超過 12 秒會斷線；尚未採樣的時段保留空白。
          </p>
        </div>
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
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.65fr)_minmax(300px,1fr)]">
        <Card className="min-w-0 p-5 sm:p-6" data-testid="throughput-panel">
          <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="heading-md">吞吐觀測</h2>
              <p className="mt-1 text-xs text-muted-foreground">
                近 5 分鐘平均速度的時間變化 · tok/s
              </p>
              <p className="mt-3 text-2xl font-semibold">
                {metric === "both"
                  ? `${number(status?.throughput.mean_decode_tps)} / ${number(status?.throughput.mean_prefill_tps)}`
                  : number(
                      metric === "decode"
                        ? status?.throughput.mean_decode_tps
                        : status?.throughput.mean_prefill_tps,
                    )}{" "}
                <span className="text-xs font-normal text-muted-foreground">
                  {metric === "both" ? "Decode / Prefill" : "tok/s"}
                </span>
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
            height={235}
            ariaLabel="吞吐速度時序圖，單位 tok/s"
            formatX={clock}
            formatY={formatNumber}
            maxGap={12000}
            activeX={activeX}
            onActiveXChange={setActiveX}
          />
          <Button
            className="mt-3"
            size="sm"
            variant="ghost"
            aria-expanded={table}
            onClick={() => setTable(!table)}
          >
            {table ? "收合數值" : "檢視採樣數值"}
          </Button>
          {table && (
            <div className="mt-3 max-h-60 overflow-auto">
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
            </div>
          )}
        </Card>
        <Card className="min-w-0 p-5 sm:p-6" data-testid="memory-panel">
          <div className="flex items-center justify-between">
            <h2 className="heading-md">Metal 記憶體</h2>
            <Badge variant="outline">
              實體 {number(status?.memory.total_gb)} GB
            </Badge>
          </div>
          <div className="flex items-center gap-4 py-5">
            {status?.memory.active_gb != null && status.memory.total_gb ? (
              <Gauge
                value={(status.memory.active_gb / status.memory.total_gb) * 100}
                size={70}
                ariaLabel="Metal 活躍配置占實體記憶體比例"
              />
            ) : (
              <span className="text-2xl text-muted-foreground">—</span>
            )}
            <div>
              <p className="text-2xl font-semibold">
                {number(status?.memory.active_gb)}{" "}
                <span className="text-sm font-normal">GB</span>
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                活躍配置 · 峰值 {number(status?.memory.peak_gb)} GB
              </p>
            </div>
          </div>
          <TimeSeriesChart
            {...chartLabels}
            data={data}
            series={memorySeries}
            height={190}
            ariaLabel="Metal 記憶體時序圖，單位 GB"
            formatX={clock}
            formatY={formatNumber}
            maxGap={12000}
            activeX={activeX}
            onActiveXChange={setActiveX}
          />
          <p className="mt-3 text-[11px] text-muted-foreground">
            MLX 配置器回報的程序記憶體。整機資源可至引擎診斷查看。
          </p>
        </Card>
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
      <div className="grid min-w-0 gap-6 xl:grid-cols-2">
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
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="heading-md">請求並行趨勢</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              不同處理階段，共用請求數刻度
            </p>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => navigate("requests")}
          >
            檢查請求
            <ArrowRight size={13} />
          </Button>
        </div>
        <TimeSeriesChart
          {...chartLabels}
          data={data}
          series={requestSeries}
          height={205}
          ariaLabel="請求階段並行數時序圖"
          formatX={clock}
          formatY={formatCount}
          maxGap={12000}
          activeX={activeX}
          onActiveXChange={setActiveX}
        />
      </Card>
      <Card className="min-w-0 p-5 sm:p-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="heading-md">模型工作區</h2>
          <Button size="sm" variant="ghost" onClick={() => navigate("models")}>
            管理模型
            <ArrowRight size={13} />
          </Button>
        </div>
        <div className="grid gap-x-6 divide-y divide-border/60 md:grid-cols-2">
          {(status?.models ?? []).slice(0, 4).map((model) => (
            <div
              key={model.id}
              className="flex items-center justify-between gap-3 py-3"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">
                  {modelLabel(model.id)}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {model.type} · {number(model.size_gb)} GB
                  {model.expires_in_s != null
                    ? ` · ${elapsed(model.expires_in_s)} 後卸載`
                    : ""}
                </p>
              </div>
              <Badge variant={model.loaded ? "success" : "secondary"}>
                {model.loading ? "載入中" : model.loaded ? "已載入" : "未載入"}
              </Badge>
            </div>
          ))}
          {!status?.models.length && (
            <p className="py-5 text-sm text-muted-foreground">
              {status ? "服務尚未註冊模型" : "連線後顯示可用模型"}
            </p>
          )}
        </div>
      </Card>
    </section>
  );
}
