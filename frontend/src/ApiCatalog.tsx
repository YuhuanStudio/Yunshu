import { useEffect, useMemo, useState } from "react";
import {
  Button,
  EmptyState,
  ScrollFade,
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
import { ExternalLink, ListTree, Search } from "lucide-react";
import { ApiError, type Connection } from "./api";
import { requestServerJson } from "./management-api";
import { t } from "./i18n/index.ts";
import { SectionCard } from "./ui";
import { CopyIconButton, ErrorNote } from "./error-note";
type Operation = {
  method: string;
  path: string;
  summary: string;
  tag: string;
  id: string;
};
export function ApiCatalog({ connection }: { connection: Connection }) {
  const [operations, setOperations] = useState<Operation[]>([]),
    [error, setError] = useState<unknown>(null),
    [loading, setLoading] = useState(true),
    [query, setQuery] = useState(""),
    [tag, setTag] = useState("all");
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
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
                  : t("api.catalog.tag.other"),
              id: typeof item.operationId === "string" ? item.operationId : "",
            });
          }
        }
        setOperations(next);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e ?? new Error());
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
    <SectionCard
      icon={ListTree}
      title={t("api.catalog.title")}
      description={t("api.catalog.description", { count: operations.length })}
      className="min-w-0 overflow-hidden"
      bodyClassName="p-0"
      data-testid="api-catalog"
      action={
        <Button
          size="sm"
          variant="secondary"
          onClick={() =>
            window.open(root + "/docs", "_blank", "noopener,noreferrer")
          }
        >
          <ExternalLink size={13} />
          {t("api.catalog.docs")}
        </Button>
      }
    >
      <div className="space-y-4 px-5 pb-5">
        <div className="flex flex-wrap gap-3">
          <Input
            className="sm:max-w-sm"
            icon={<Search size={13} />}
            aria-label={t("api.catalog.search.aria")}
            placeholder={t("api.catalog.search.placeholder")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Select value={tag} onValueChange={setTag}>
            <SelectTrigger
              aria-label={t("api.catalog.tag.aria")}
              className="w-48"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t("api.catalog.tag.all")}</SelectItem>
              {tags.map((item) => (
                <SelectItem key={item} value={item}>
                  {item}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {error != null && (
          <ErrorNote
            tone="warning"
            message={
              error instanceof ApiError
                ? error.publicMessage
                : t("api.catalog.error")
            }
            error={error}
          />
        )}
      </div>
      <ScrollFade className="max-h-[32rem] overflow-auto">
        <Table scrollLabel={t("api.catalog.table.aria")}>
          <Thead>
            <Tr>
              <Th>{t("api.catalog.table.method")}</Th>
              <Th>{t("api.catalog.table.path")}</Th>
              <Th>{t("api.catalog.table.function")}</Th>
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((item) => (
              <Tr key={item.method + item.path}>
                <Td>
                  <span
                    className={`text-xs font-medium tabular-nums ${item.method === "GET" ? "text-muted-foreground" : "text-foreground"}`}
                  >
                    {item.method}
                  </span>
                </Td>
                <Td>
                  <span className="inline-flex items-center gap-1">
                    <a
                      className="break-all font-mono text-xs underline underline-offset-4"
                      href={`${root}/docs#/${encodeURIComponent(item.tag)}/${encodeURIComponent(item.id)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {item.path}
                    </a>
                    <CopyIconButton
                      value={root + item.path}
                      label={t("api.catalog.copy", {
                        method: item.method,
                        path: item.path,
                      })}
                    />
                  </span>
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
      </ScrollFade>
      {!rows.length && (
        <EmptyState
          size="inline"
          title={
            loading
              ? t("api.catalog.loading")
              : error
                ? t("api.catalog.unavailable")
                : t("api.catalog.empty")
          }
          description={t("api.catalog.emptyNote")}
        />
      )}
    </SectionCard>
  );
}
