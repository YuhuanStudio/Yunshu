import { useMemo, useState } from "react";
import {
  AreaChart,
  Badge,
  Button,
  Card,
  Gauge,
  Progress,
  SegmentedSelect,
  Sparkline,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { PageHeader, StatCard } from "@yuhuanowo/yunui/patterns";
import {
  Activity,
  ArrowRight,
  Clock3,
  Database,
  Pause,
  Play,
  RefreshCw,
  Zap,
} from "lucide-react";
import {
  clock,
  elapsed,
  MetricChart,
  modelLabel,
  number,
  type Engine,
} from "./ui";
export function Dashboard({
  engine,
  navigate,
}: {
  engine: Engine;
  navigate: (page: string) => void;
}) {
  const [range, setRange] = useState("15m"),
    [metric, setMetric] = useState("decode"),
    [table, setTable] = useState(false);
  const status = engine.status,
    online = engine.phase === "online";
  const points = useMemo(() => {
    const cutoff =
      (engine.updatedAt ?? Date.now()) -
      (range === "5m" ? 300 : range === "15m" ? 900 : 3600) * 1000;
    return engine.history.filter((x) => x.at >= cutoff);
  }, [engine.history, engine.updatedAt, range]);
  const series = (
    read: (s: NonNullable<typeof status>) => number | null | undefined,
  ) =>
    points.flatMap((p) => {
      const v = read(p.status);
      return typeof v === "number" && Number.isFinite(v)
        ? [{ value: v, label: clock(p.at) }]
        : [];
    });
  const throughput = series((s) =>
      metric === "decode"
        ? s.throughput.mean_decode_tps
        : s.throughput.mean_prefill_tps,
    ),
    memory = series((s) => s.memory.active_gb),
    requests = series((s) => s.requests.active);
  const last = status?.last,
    cache =
      last && last.prompt_tokens > 0
        ? (last.cached_tokens / last.prompt_tokens) * 100
        : null;
  const phaseCounts = Object.entries(
    (status?.requests.items ?? []).reduce<Record<string, number>>(
      (counts, row) => {
        counts[row.phase] = (counts[row.phase] ?? 0) + 1;
        return counts;
      },
      {},
    ),
  );
  const phases: Record<string, string> = {
    queued: "排隊",
    starting: "準備中",
    prefill: "Prefill",
    decode: "Decode",
  };
  return (
    <section
      className="mx-auto w-full max-w-7xl space-y-5 p-4 sm:p-7"
      data-testid="overview"
    >
      <PageHeader
        title="引擎總覽"
        description="這台 Mac 的推理工作、模型與資源。"
        actions={
          <div className="flex gap-2">
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
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={online ? "success" : "secondary"}>
            {online ? (status?.state ?? "已連線") : "未連線"}
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
      </div>
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatCard
          compact
          valueFirst
          icon={Activity}
          label="進行中請求"
          value={number(status?.requests.active, 0)}
          subtext={
            <div className="flex flex-wrap items-end justify-between gap-2">
              <span>
                近 60 秒結束 {number(status?.throughput.requests, 0)} 筆
              </span>
              <Sparkline
                className="w-20"
                data={requests.map((p) => p.value)}
                label="活動請求趨勢"
              />
            </div>
          }
        />
        <StatCard
          compact
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
                data={series((s) => s.throughput.mean_decode_tps).map(
                  (p) => p.value,
                )}
                label="平均 Decode 速度"
              />
            </div>
          }
        />
        <StatCard
          compact
          valueFirst
          icon={Clock3}
          label="最近結束 · 首 Token 延遲"
          value={
            <>
              {number(last?.ttft_ms, 0)}
              <span className="ml-1 text-xs font-normal">ms</span>
            </>
          }
          subtext={
            last
              ? `${last.request_id} · ${clock(last.t * 1000)}`
              : "等待第一筆結束的請求"
          }
        />
        <StatCard
          compact
          valueFirst
          icon={Database}
          label="最近結束 · 前綴重用率"
          value={cache == null ? "—" : `${number(cache)}%`}
          subtext={
            last
              ? `${number(last.cached_tokens, 0)} / ${number(last.prompt_tokens, 0)} prompt tokens`
              : "由真實請求的快取 token 計算"
          }
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          本頁開啟後採樣 · {points.length} 筆 · 中斷期間不補資料
        </p>
        <SegmentedSelect
          aria-label="觀測時間範圍"
          value={range}
          onChange={setRange}
          options={[
            { value: "5m", label: "5 分鐘" },
            { value: "15m", label: "15 分鐘" },
            { value: "1h", label: "1 小時" },
          ]}
        />
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.65fr)_minmax(300px,1fr)]">
        <Card className="min-w-0 p-5">
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs text-muted-foreground">吞吐觀測</p>
              <p className="mt-1 text-2xl font-semibold">
                {number(
                  metric === "decode"
                    ? status?.throughput.mean_decode_tps
                    : status?.throughput.mean_prefill_tps,
                )}{" "}
                <span className="text-sm font-normal text-muted-foreground">
                  tok/s
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
              ]}
            />
          </div>
          <MetricChart
            title={metric === "decode" ? "生成速度" : "提示詞處理速度"}
            description="每次採樣記錄服務回報的近 5 分鐘平均值"
            data={throughput}
            unit="tok/s"
            tone={metric === "decode" ? "accent" : "info"}
            height={210}
          />
          <Button
            className="mt-4"
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
                    <Th>記憶體 GB</Th>
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
        <Card className="min-w-0 p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Metal 記憶體</h2>
            <Badge variant="outline">
              實體 {number(status?.memory.total_gb)} GB
            </Badge>
          </div>
          <div className="flex items-center gap-4 py-5">
            {status?.memory.active_gb != null && status.memory.total_gb ? (
              <Gauge
                value={(status.memory.active_gb / status.memory.total_gb) * 100}
                size={82}
                ariaLabel="Metal 活躍配置占實體記憶體比例"
              />
            ) : (
              <div className="text-2xl text-muted-foreground">—</div>
            )}
            <div>
              <p className="text-2xl font-semibold">
                {number(status?.memory.active_gb)}{" "}
                <span className="text-sm font-normal">GB</span>
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                活躍配置 · 非整機記憶體用量
              </p>
            </div>
          </div>
          <div className="space-y-3 border-y border-border/60 py-4">
            {[
              ["配置器快取", status?.memory.cache_gb],
              ["程序峰值", status?.memory.peak_gb],
            ].map(([label, value]) => (
              <div key={String(label)} className="flex justify-between text-xs">
                <span className="text-muted-foreground">{label}</span>
                <span className="font-mono">
                  {number(value as number | undefined)} GB
                </span>
              </div>
            ))}
          </div>
          <div className="mt-5">
            <MetricChart
              title="活躍配置趨勢"
              description="MLX 回報的程序配置"
              data={memory}
              unit="GB"
              tone="warning"
              height={110}
            />
          </div>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="min-w-0 p-5">
          <MetricChart
            title="請求並行數"
            description="每個採樣時刻仍在執行的請求"
            data={requests}
            unit="requests"
            tone="info"
            height={140}
          />
          <div className="mt-5 flex flex-wrap gap-x-6 gap-y-3">
            {phaseCounts.length ? (
              phaseCounts.map(([phase, count]) => (
                <span key={phase} className="text-xs text-muted-foreground">
                  {phases[phase] ?? phase}
                  <strong className="ml-2 text-foreground">{count}</strong>
                </span>
              ))
            ) : (
              <p className="text-xs text-muted-foreground">
                {status ? "目前沒有進行中的請求" : "尚未取得請求狀態"}
              </p>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="mt-3"
            onClick={() => navigate("requests")}
          >
            檢查請求
            <ArrowRight size={13} />
          </Button>
        </Card>
        <Card className="min-w-0 p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold">模型工作區</h2>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => navigate("models")}
            >
              管理模型
              <ArrowRight size={13} />
            </Button>
          </div>
          <div className="divide-y divide-border/60">
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
                  {model.loading
                    ? "載入中"
                    : model.loaded
                      ? "已載入"
                      : "未載入"}
                </Badge>
              </div>
            ))}
            {!status?.models.length && (
              <p className="py-8 text-sm text-muted-foreground">
                {status ? "服務尚未註冊模型" : "連線後顯示可用模型"}
              </p>
            )}
          </div>
          <p className="mt-4 text-xs text-muted-foreground">
            模型生命週期與服務共用；此處不建立模擬載入狀態。
          </p>
        </Card>
      </div>
    </section>
  );
}
