import { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  NavTabs,
  SegmentedBar,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import {
  DashboardPage,
  DetailList,
  DetailRow,
  PageHeader,
  SectionRow,
  StatCard,
  StatGrid,
} from "@yuhuanowo/yunui/patterns";
import {
  Activity,
  Check,
  Copy,
  Cpu,
  Database,
  HeartPulse,
  Layers,
  MemoryStick,
  RefreshCw,
  Server,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { ApiError, type Connection, type EngineStatus } from "./api";
import { requestServerJson } from "./management-api";
import { SectionCard, clock, elapsed, number, type Engine } from "./ui";
const groups = {
  system: [
    { key: "system", title: "主機資源", path: "/debug/system" },
    { key: "engine", title: "引擎計數", path: "/debug/engine" },
  ],
  requests: [{ key: "requests", title: "請求與延遲", path: "/debug/requests" }],
  cache: [
    { key: "kv", title: "KV 前綴快取", path: "/debug/kv-cache" },
    { key: "ssd", title: "SSD 快取", path: "/debug/ssd-cache" },
  ],
  decode: [
    { key: "spec", title: "推測解碼", path: "/debug/spec-decode" },
    { key: "perModel", title: "逐模型運行情況", path: "/debug/per-model" },
  ],
  memory: [
    { key: "guard", title: "記憶體保護", path: "/debug/memory-guard" },
    { key: "census", title: "配置明細", path: "/debug/memory-census" },
  ],
} as const;
type Group = keyof typeof groups;
type Result = {
  key: string;
  title: string;
  path: string;
  data?: unknown;
  error?: string;
  status?: number;
};
function at(value: unknown, ...keys: string[]): unknown {
  let next = value;
  for (const key of keys) {
    if (!next || typeof next !== "object" || !Object.hasOwn(next, key))
      return undefined;
    next = (next as Record<string, unknown>)[key];
  }
  return next;
}
const metric = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;
const gb = (value: unknown) => {
  const n = metric(value);
  return n == null ? undefined : n / 1e9;
};
type Health = {
  key: string;
  name: string;
  status: "online" | "offline" | "away" | "neutral";
  value: string;
  hint: string;
  /** Real proportion behind the row (loaded / registered, active / total, CPU %), when there is one. */
  bar?: {
    value: number;
    total: number;
    tone: "success" | "warning" | "neutral";
  };
};
const groupIcon: Record<Group, LucideIcon> = {
  system: Server,
  requests: Timer,
  cache: Layers,
  decode: Cpu,
  memory: MemoryStick,
};
/** Health checks derived only from /yunshu/status and the /debug/system reply. */
export function healthChecks(
  status: EngineStatus | null,
  system: unknown,
  systemState: "ok" | "disabled" | "error" | "pending",
): Health[] {
  const rows: Health[] = [];
  if (!status) {
    rows.push({
      key: "engine",
      name: "引擎狀態",
      status: "offline",
      value: "無法讀取",
      hint: "/v1/yunshu/status 沒有回應，請確認服務位址與存取權杖。",
    });
  } else {
    const running = ["running", "ready"].includes(status.state);
    rows.push({
      key: "engine",
      name: "引擎狀態",
      status: status.load_error ? "offline" : running ? "online" : "away",
      value: status.state,
      hint:
        status.load_error ??
        `版本 ${status.version}，已運行 ${elapsed(status.uptime_s)}`,
    });
    const loaded = status.models.filter((m) => m.loaded).length;
    rows.push({
      key: "models",
      name: "模型載入",
      status: loaded > 0 ? "online" : "neutral",
      value: `${loaded} / ${status.models.length}`,
      bar: status.models.length
        ? { value: loaded, total: status.models.length, tone: "success" }
        : undefined,
      hint:
        loaded > 0
          ? "已載入的模型可直接推論。"
          : "目前沒有已載入的模型，首個請求會觸發載入。",
    });
    const active = status.memory.active_gb,
      total = status.memory.total_gb;
    const ratio = active != null && total ? active / total : undefined;
    rows.push({
      key: "memory",
      name: "MLX 記憶體",
      status: ratio == null ? "neutral" : ratio > 0.9 ? "away" : "online",
      bar:
        ratio == null || !total
          ? undefined
          : {
              value: active ?? 0,
              total,
              tone: ratio > 0.9 ? "warning" : "success",
            },
      value:
        active != null && total
          ? `${number(active)} / ${number(total)} GB`
          : "—",
      hint:
        ratio != null && ratio > 0.9
          ? "活躍配置超過總量的 90%，可能觸發記憶體保護。"
          : "活躍配置佔總量的比例正常。",
    });
    rows.push({
      key: "queue",
      name: "請求佇列",
      status: status.requests.queued > 0 ? "away" : "online",
      value: `${number(status.requests.active, 0)} 進行 · ${number(status.requests.queued, 0)} 排隊`,
      hint:
        status.requests.queued > 0
          ? "有請求在排隊等待前一個請求完成。"
          : "沒有排隊中的請求。",
    });
    const tps =
      status.throughput.live_decode_tps ?? status.throughput.mean_decode_tps;
    rows.push({
      key: "throughput",
      name: "解碼速度",
      status: tps == null ? "neutral" : "online",
      value: tps == null ? "—" : `${number(tps)} tok/s`,
      hint: `近 ${number(status.throughput.window_s, 0)} 秒內 ${number(status.throughput.requests, 0)} 個請求。`,
    });
  }
  rows.push({
    key: "debug",
    name: "診斷介面",
    status:
      systemState === "ok"
        ? "online"
        : systemState === "pending"
          ? "neutral"
          : "offline",
    value:
      systemState === "ok"
        ? "可用"
        : systemState === "pending"
          ? "讀取中"
          : systemState === "disabled"
            ? "未啟用"
            : "讀取失敗",
    hint:
      systemState === "disabled"
        ? "服務需以 YUNSHU_DEBUG_ROUTES 啟動才提供 /debug。"
        : systemState === "error"
          ? "/debug/system 回應錯誤，可能需要有效的存取權杖。"
          : "/debug/system 可讀取。",
  });
  const cpu = metric(at(system, "cpu", "percent"));
  if (cpu != null)
    rows.push({
      key: "cpu",
      name: "主機 CPU",
      status: cpu > 90 ? "away" : "online",
      value: `${number(cpu)}%`,
      bar: { value: cpu, total: 100, tone: cpu > 90 ? "warning" : "success" },
      hint: `${number(metric(at(system, "cpu", "logical_cores")), 0)} 個邏輯核心。`,
    });
  return rows;
}
export function Diagnostics({
  connection,
  engine,
}: {
  connection: Connection;
  engine: Engine;
}) {
  const status = engine.status;
  const [group, setGroup] = useState<Group>("system"),
    [refresh, setRefresh] = useState(0),
    [results, setResults] = useState<Result[]>([]),
    [loading, setLoading] = useState(false),
    [updated, setUpdated] = useState<number | null>(null),
    [expanded, setExpanded] = useState<Record<string, boolean>>({}),
    [overviewSystem, setOverviewSystem] = useState<unknown>(undefined),
    [systemState, setSystemState] = useState<
      "ok" | "disabled" | "error" | "pending"
    >("pending"),
    [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");
  useEffect(() => {
    const controller = new AbortController();
    setSystemState("pending");
    void requestServerJson(connection, "/debug/system", {
      signal: controller.signal,
    })
      .then((data) => {
        if (controller.signal.aborted) return;
        setOverviewSystem(data);
        setSystemState("ok");
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setOverviewSystem(undefined);
        setSystemState(
          e instanceof ApiError && e.status === 404 ? "disabled" : "error",
        );
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, refresh]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setResults([]);
    setExpanded({});
    setUpdated(null);
    void Promise.all(
      groups[group].map(async (endpoint) => {
        try {
          const data = await requestServerJson(connection, endpoint.path, {
            signal: controller.signal,
          });
          return {
            ...endpoint,
            data,
            error:
              typeof at(data, "error") === "string"
                ? String(at(data, "error"))
                : undefined,
          } as Result;
        } catch (e) {
          return {
            ...endpoint,
            error: e instanceof Error ? e.message : "無法取得資料",
            status: e instanceof ApiError ? e.status : undefined,
          } as Result;
        }
      }),
    ).then((rows) => {
      if (!controller.signal.aborted) {
        setResults(rows);
        setUpdated(Date.now());
        setLoading(false);
      }
    });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, group, refresh]);
  const system = overviewSystem,
    engineCounters = results.find((r) => r.key === "engine")?.data,
    request = results.find((r) => r.key === "requests")?.data;
  const checks = healthChecks(status, system, systemState);
  async function copyBundle() {
    const bundle = JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        service: connection.baseUrl,
        status,
        debug_system: system ?? null,
        group,
        endpoints: results.map(({ key, path, data, error, status: code }) => ({
          key,
          path,
          http_status: code ?? null,
          error: error ?? null,
          data: data ?? null,
        })),
      },
      null,
      2,
    );
    try {
      await navigator.clipboard.writeText(bundle);
      setCopied("done");
    } catch {
      setCopied("failed");
    }
    setTimeout(() => setCopied("idle"), 2500);
  }
  const unavailable =
    results.length > 0 && results.every((row) => row.status === 404);
  return (
    <DashboardPage data-testid="diagnostics">
      <PageHeader
        title="引擎診斷"
        description="直接讀取服務的資源、請求、快取與解碼狀態。"
        actions={
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={loading}
              onClick={() => void copyBundle()}
            >
              {copied === "done" ? <Check size={13} /> : <Copy size={13} />}
              {copied === "done"
                ? "已複製"
                : copied === "failed"
                  ? "無法寫入剪貼簿"
                  : "複製診斷資料"}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={loading}
              onClick={() => {
                void engine.refresh();
                setRefresh((n) => n + 1);
              }}
            >
              <RefreshCw size={13} />
              {loading ? "讀取中" : "重新讀取"}
            </Button>
          </div>
        }
      />
      <StatGrid data-testid="resource-readouts">
        <StatCard
          compact
          icon={Cpu}
          label="CPU"
          value={`${number(metric(at(system, "cpu", "percent")))}%`}
          subtext={`${number(metric(at(system, "cpu", "logical_cores")), 0)} 個邏輯核心`}
        />
        <StatCard
          compact
          icon={Activity}
          label="系統記憶體"
          value={`${number(metric(at(system, "memory", "percent")))}%`}
          subtext={`${number(gb(at(system, "memory", "used_bytes")))} GB 已使用`}
        />
        <StatCard
          compact
          icon={Database}
          label="MLX 活躍配置"
          value={`${number(status?.memory.active_gb ?? gb(at(system, "gpu", "active_bytes")))} GB`}
          subtext={`峰值 ${number(status?.memory.peak_gb)} GB · 程序配置量，不是 GPU 使用率`}
        />
        <StatCard
          compact
          icon={Server}
          label="已處理請求"
          value={number(metric(at(engineCounters, "requests_processed")), 0)}
          subtext="服務計數器"
        />
      </StatGrid>
      <SectionCard
        icon={HeartPulse}
        title="健康檢查"
        description="只用 /v1/yunshu/status 與 /debug/system 的實際回報判斷。"
        data-testid="health-checks"
      >
        <ul className="-my-2 divide-y divide-border">
          {checks.map((check) => (
            <li key={check.key} className="py-3">
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1">
                <div className="min-w-0">
                  <StatusIndicator status={check.status}>
                    <span className="text-sm font-medium text-foreground">
                      {check.name}
                    </span>
                  </StatusIndicator>
                  <p className="mt-0.5 pl-4 text-xs text-muted-foreground">
                    {check.hint}
                  </p>
                </div>
                <span className="text-sm tabular-nums">{check.value}</span>
              </div>
              {check.bar && (
                <SegmentedBar
                  className="mt-2 pl-4"
                  height={6}
                  total={check.bar.total}
                  segments={[{ value: check.bar.value, tone: check.bar.tone }]}
                  label={`${check.name}：${check.value}`}
                />
              )}
            </li>
          ))}
        </ul>
      </SectionCard>
      <SectionRow
        title="逐項診斷"
        action={
          <p className="text-xs text-muted-foreground">
            {updated ? `讀取於 ${clock(updated)}` : "尚未取得資料"} · 手動更新
          </p>
        }
      />
      <div>
        <NavTabs
          ariaLabel="診斷類別"
          activeKey={group}
          onChange={(value) => setGroup(value as Group)}
          tabs={[
            {
              key: "system",
              label: (
                <>
                  <Server size={14} />
                  主機與引擎
                </>
              ),
            },
            {
              key: "requests",
              label: (
                <>
                  <Activity size={14} />
                  請求
                </>
              ),
            },
            {
              key: "cache",
              label: (
                <>
                  <Database size={14} />
                  快取
                </>
              ),
            },
            {
              key: "decode",
              label: (
                <>
                  <Cpu size={14} />
                  解碼
                </>
              ),
            },
            {
              key: "memory",
              label: (
                <>
                  <Database size={14} />
                  記憶體
                </>
              ),
            },
          ]}
        />
      </div>
      {loading && (
        <Card className="p-6 text-sm text-muted-foreground" role="status">
          正在讀取 {groups[group].map((item) => item.title).join("、")}…
        </Card>
      )}
      {unavailable && (
        <Card className="p-5">
          <EmptyState
            icon={<Server size={24} />}
            title="此服務未啟用診斷介面"
            description="需要在引擎啟動設定啟用 YUNSHU_DEBUG_ROUTES，並使用有效的存取權杖。控制台不會自行變更服務設定。"
          />
        </Card>
      )}
      {group === "requests" && request !== undefined && (
        <SectionCard
          icon={Timer}
          title="服務延遲百分位數"
          description="後端近 60 秒 HTTP 請求耗時統計，與首頁的已觀測 TTFT 分布不同。"
        >
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {["p50", "p90", "p95", "p99"].map((key) => (
              <div key={key}>
                <p className="text-xs uppercase text-muted-foreground">{key}</p>
                <p className="mt-2 text-xl tabular-nums">
                  {number(metric(at(request, "latency_percentiles", key)))}{" "}
                  <span className="text-xs">ms</span>
                </p>
              </div>
            ))}
          </div>
        </SectionCard>
      )}
      {results.map((row) => (
        <SectionCard
          key={row.key}
          icon={groupIcon[group]}
          title={row.title}
          description={<span className="font-mono">{row.path}</span>}
          className="min-w-0"
          action={
            <Badge variant={row.error ? "warning" : "success"}>
              {row.error
                ? row.status === 404
                  ? "未啟用"
                  : row.status === 401 || row.status === 403
                    ? "需要授權"
                    : "讀取失敗"
                : "已讀取"}
            </Badge>
          }
        >
          {row.error ? (
            <p role="status" className="text-sm text-muted-foreground">
              {row.error}
            </p>
          ) : (
            <div className="space-y-4">
              <DetailList className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
                {Object.entries(
                  row.data && typeof row.data === "object" ? row.data : {},
                )
                  .filter(
                    ([, value]) =>
                      typeof value === "number" ||
                      typeof value === "boolean" ||
                      typeof value === "string",
                  )
                  .slice(0, 12)
                  .map(([key, value]) => (
                    <DetailRow
                      key={key}
                      label={key}
                      value={
                        typeof value === "number"
                          ? number(value, 2)
                          : String(value)
                      }
                    />
                  ))}
              </DetailList>
              {Array.isArray(at(row.data, "caches")) && (
                <Table scrollLabel="模型快取診斷">
                  <Thead>
                    <Tr>
                      <Th>模型</Th>
                      <Th>前綴命中</Th>
                      <Th>Resident</Th>
                      <Th>SSD</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    {(at(row.data, "caches") as unknown[]).map(
                      (cache, index) => (
                        <Tr key={index}>
                          <Td>
                            <span className="block max-w-44 truncate text-xs">
                              {String(at(cache, "model_id") ?? "—")}
                            </span>
                          </Td>
                          <Td>
                            {number(
                              metric(at(cache, "apc", "token_hit_rate")) == null
                                ? undefined
                                : Number(at(cache, "apc", "token_hit_rate")) *
                                    100,
                            )}
                            %
                          </Td>
                          <Td>
                            {number(
                              metric(at(cache, "apc", "resident_bytes")) == null
                                ? undefined
                                : Number(at(cache, "apc", "resident_bytes")) /
                                    1e6,
                            )}{" "}
                            MB
                          </Td>
                          <Td>
                            {number(
                              metric(at(cache, "apc", "disk_bytes")) == null
                                ? undefined
                                : Number(at(cache, "apc", "disk_bytes")) / 1e6,
                            )}{" "}
                            MB
                          </Td>
                        </Tr>
                      ),
                    )}
                  </Tbody>
                </Table>
              )}
              <Button
                size="sm"
                variant="ghost"
                aria-expanded={!!expanded[row.key]}
                onClick={() =>
                  setExpanded((current) => ({
                    ...current,
                    [row.key]: !current[row.key],
                  }))
                }
              >
                {expanded[row.key] ? "收合原始資料" : "查看完整診斷資料"}
              </Button>
              {expanded[row.key] && (
                <CodeBlock language="json">
                  {JSON.stringify(row.data, null, 2) ?? "null"}
                </CodeBlock>
              )}
            </div>
          )}
        </SectionCard>
      ))}
    </DashboardPage>
  );
}
