import { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  NavTabs,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { CodeBlock, PageHeader, StatCard } from "@yuhuanowo/yunui/patterns";
import { Activity, Cpu, Database, RefreshCw, Server } from "lucide-react";
import { ApiError, type Connection } from "./api";
import { requestServerJson } from "./management-api";
import { clock, number } from "./ui";
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
export function Diagnostics({ connection }: { connection: Connection }) {
  const [group, setGroup] = useState<Group>("system"),
    [refresh, setRefresh] = useState(0),
    [results, setResults] = useState<Result[]>([]),
    [loading, setLoading] = useState(false),
    [updated, setUpdated] = useState<number | null>(null),
    [expanded, setExpanded] = useState<Record<string, boolean>>({});
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
  const system = results.find((r) => r.key === "system")?.data,
    engine = results.find((r) => r.key === "engine")?.data,
    request = results.find((r) => r.key === "requests")?.data;
  const unavailable =
    results.length > 0 && results.every((row) => row.status === 404);
  return (
    <section className="w-full max-w-5xl space-y-6" data-testid="diagnostics">
      <PageHeader
        title="引擎診斷"
        description="直接讀取服務的資源、請求、快取與解碼狀態。"
        actions={
          <Button
            size="sm"
            variant="secondary"
            disabled={loading}
            onClick={() => setRefresh((n) => n + 1)}
          >
            <RefreshCw size={13} />
            {loading ? "讀取中" : "重新讀取"}
          </Button>
        }
      />
      <div className="flex flex-wrap items-center justify-between gap-3">
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

        <p className="text-xs text-muted-foreground">
          {updated ? `讀取於 ${clock(updated)}` : "尚未取得資料"} · 手動更新
        </p>
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
      {group === "system" &&
        !loading &&
        (system !== undefined || engine !== undefined) && (
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
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
              subtext={`${number(metric(at(system, "memory", "used_bytes")) == null ? undefined : Number(at(system, "memory", "used_bytes")) / 1e9)} GB 已使用`}
            />
            <StatCard
              compact
              icon={Database}
              label="MLX 活躍配置"
              value={`${number(metric(at(system, "gpu", "active_bytes")) == null ? undefined : Number(at(system, "gpu", "active_bytes")) / 1e9)} GB`}
              subtext="程序配置量，不是 GPU 使用率"
            />
            <StatCard
              compact
              icon={Server}
              label="已處理請求"
              value={number(metric(at(engine, "requests_processed")), 0)}
              subtext="服務計數器"
            />
          </div>
        )}
      {group === "requests" && request !== undefined && (
        <Card className="p-5">
          <h2 className="mb-4 text-sm font-semibold">服務延遲百分位數</h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {["p50", "p90", "p95", "p99"].map((key) => (
              <div key={key}>
                <p className="text-xs uppercase text-muted-foreground">{key}</p>
                <p className="mt-2 font-mono text-xl">
                  {number(metric(at(request, "latency_percentiles", key)))}{" "}
                  <span className="text-xs">ms</span>
                </p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-xs text-muted-foreground">
            後端近 60 秒 HTTP 請求耗時統計，與首頁的已觀測 TTFT 分布不同。
          </p>
        </Card>
      )}
      {results.map((row) => (
        <Card key={row.key} className="min-w-0 space-y-4 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold">{row.title}</h2>
              <p className="mt-1 font-mono text-[10px] text-muted-foreground">
                {row.path}
              </p>
            </div>
            <Badge variant={row.error ? "warning" : "success"}>
              {row.error
                ? row.status === 404
                  ? "未啟用"
                  : row.status === 401 || row.status === 403
                    ? "需要授權"
                    : "讀取失敗"
                : "已讀取"}
            </Badge>
          </div>
          {row.error ? (
            <p role="status" className="text-sm text-muted-foreground">
              {row.error}
            </p>
          ) : (
            <>
              <div className="flex flex-wrap gap-x-6 gap-y-2">
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
                    <div key={key} className="min-w-0 text-xs">
                      <span className="text-muted-foreground">{key} </span>
                      <span className="break-all font-mono">
                        {typeof value === "number"
                          ? number(value, 2)
                          : String(value)}
                      </span>
                    </div>
                  ))}
              </div>
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
                <CodeBlock
                  language="json"
                  code={JSON.stringify(row.data, null, 2) ?? "null"}
                />
              )}
            </>
          )}
        </Card>
      ))}
    </section>
  );
}
