import { useEffect, useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { ExternalLink, Search } from "lucide-react";
import type { Connection } from "./api";
import { requestServerJson } from "./management-api";
type Operation = {
  method: string;
  path: string;
  summary: string;
  tag: string;
  id: string;
};
export function ApiCatalog({ connection }: { connection: Connection }) {
  const [operations, setOperations] = useState<Operation[]>([]),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [query, setQuery] = useState(""),
    [tag, setTag] = useState("all");
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setOperations([]);
    void requestServerJson<{ paths?: Record<string, Record<string, unknown>> }>(
      connection,
      "/openapi.json",
      { signal: controller.signal },
    )
      .then((schema) => {
        if (controller.signal.aborted) return;
        const next: Operation[] = [];
        for (const [path, methods] of Object.entries(schema?.paths ?? {})) {
          for (const [method, value] of Object.entries(methods)) {
            if (
              !["get", "post", "put", "patch", "delete"].includes(method) ||
              !value ||
              typeof value !== "object"
            )
              continue;
            const item = value as Record<string, unknown>;
            next.push({
              method: method.toUpperCase(),
              path,
              summary: typeof item.summary === "string" ? item.summary : "",
              tag:
                Array.isArray(item.tags) && typeof item.tags[0] === "string"
                  ? item.tags[0]
                  : "其他",
              id: typeof item.operationId === "string" ? item.operationId : "",
            });
          }
        }
        setOperations(next);
      })
      .catch((e) => {
        if (!controller.signal.aborted)
          setError(e instanceof Error ? e.message : "無法取得 API 定義");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token]);
  const tags = useMemo(
    () => [...new Set(operations.map((item) => item.tag))].sort(),
    [operations],
  );
  const rows = operations.filter(
    (item) =>
      (tag === "all" || item.tag === tag) &&
      `${item.method} ${item.path} ${item.summary}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const root = connection.baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
  return (
    <Card className="min-w-0 overflow-hidden" data-testid="api-catalog">
      <div className="space-y-4 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">此服務的完整 API</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              從目前引擎的 OpenAPI 定義讀取，共 {operations.length} 個操作。
            </p>
          </div>
          <Button
            size="sm"
            variant="secondary"
            onClick={() =>
              window.open(root + "/docs", "_blank", "noopener,noreferrer")
            }
          >
            <ExternalLink size={13} />
            API 文件
          </Button>
        </div>
        <div className="flex flex-wrap gap-3">
          <Input
            className="sm:max-w-sm"
            icon={<Search size={13} />}
            aria-label="搜尋 API"
            placeholder="搜尋路徑、方法或功能"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Select value={tag} onValueChange={setTag}>
            <SelectTrigger aria-label="API 類別" className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部功能</SelectItem>
              {tags.map((item) => (
                <SelectItem key={item} value={item}>
                  {item}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {error && (
          <p role="status" className="text-xs text-warning">
            {error}
          </p>
        )}
      </div>
      <div className="max-h-[32rem] overflow-auto">
        <Table scrollLabel="服務 API 目錄">
          <Thead>
            <Tr>
              <Th>方法</Th>
              <Th>路徑</Th>
              <Th>功能</Th>
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((item) => (
              <Tr key={item.method + item.path}>
                <Td>
                  <Badge
                    variant={
                      item.method === "GET"
                        ? "secondary"
                        : item.method === "DELETE"
                          ? "warning"
                          : "info"
                    }
                  >
                    {item.method}
                  </Badge>
                </Td>
                <Td>
                  <a
                    className="break-all font-mono text-xs underline underline-offset-4"
                    href={`${root}/docs#/${encodeURIComponent(item.tag)}/${encodeURIComponent(item.id)}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {item.path}
                  </a>
                </Td>
                <Td>
                  <span className="text-xs text-muted-foreground">
                    {item.summary}
                  </span>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </div>
      {!rows.length && (
        <EmptyState
          title={
            loading
              ? "讀取 API 定義…"
              : error
                ? "API 定義尚未取得"
                : "沒有符合的 API"
          }
          description="此目錄反映服務實際提供的路由，不會推測未提供的功能。"
        />
      )}
    </Card>
  );
}
