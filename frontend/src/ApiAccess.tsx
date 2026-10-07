import { useEffect, useMemo, useState } from "react";
import {
  Card,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@yuhuanowo/yunui";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import { PageHeader } from "@yuhuanowo/yunui/patterns";
import type { Connection } from "./api";
import { ApiCatalog } from "./ApiCatalog";
import { buildIntegrations, serviceRoot } from "./integrations";
import { modelLabel, type Engine } from "./ui";

export function ApiView({
  connection,
  engine,
}: {
  connection: Connection;
  engine: Engine;
}) {
  const root = serviceRoot(connection.baseUrl),
    [models, setModels] = useState<string[]>([]),
    [model, setModel] = useState("");
  const status = engine.status;
  useEffect(() => {
    if (!status) return;
    const ids = status.models.map((item) => item.id);
    setModels(ids);
    setModel(
      (current) =>
        (ids.includes(current) ? current : undefined) ??
        status.models.find((item) => item.loaded)?.id ??
        ids[0] ??
        "",
    );
  }, [status]);
  const integrations = useMemo(
    () => buildIntegrations(root, model),
    [root, model],
  );
  return (
    <section className="mx-auto w-full max-w-7xl space-y-6" data-testid="api">
      <PageHeader
        title="API 接入"
        description="使用熟悉的 SDK 或程式代理，讓你的應用連接本機模型。"
      />
      <Card className="grid gap-5 p-5 sm:grid-cols-[1fr_1fr_auto] sm:items-end">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">OpenAI API Base URL</p>
          <p className="mt-1.5 break-all font-mono text-sm">{root}/v1</p>
        </div>
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">Anthropic Base URL</p>
          <p className="mt-1.5 break-all font-mono text-sm">{root}</p>
        </div>
        <div className="min-w-0">
          <p className="mb-1.5 text-xs text-muted-foreground">
            命令中使用的模型
          </p>
          <Select
            value={model || undefined}
            onValueChange={setModel}
            disabled={!models.length}
          >
            <SelectTrigger
              aria-label="接入使用的模型"
              className="w-full sm:w-56"
            >
              <SelectValue placeholder="local" />
            </SelectTrigger>
            <SelectContent>
              {models.map((id) => (
                <SelectItem key={id} value={id}>
                  {modelLabel(id)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2" data-testid="integrations">
        {integrations.map((item) => (
          <Card
            key={item.id}
            className="min-w-0 space-y-3 p-5"
            data-testid={`integration-${item.id}`}
          >
            <div>
              <h2 className="text-sm font-semibold">{item.title}</h2>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                {item.description}
              </p>
            </div>
            <CodeBlock language={item.language} filename={item.filename}>
              {item.code}
            </CodeBlock>
          </Card>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        命令只引用環境變數 YUNSHU_AUTH_TOKEN（未設定時為
        local），不會寫入真實權杖。
      </p>
      <ApiCatalog connection={connection} />
    </section>
  );
}
