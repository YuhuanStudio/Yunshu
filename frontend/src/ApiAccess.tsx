import { useEffect, useMemo, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@yuhuanowo/yunui";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import {
  DashboardPage,
  PageHeader,
  SectionRow,
} from "@yuhuanowo/yunui/patterns";
import {
  Bot,
  Code2,
  Globe,
  KeyRound,
  Terminal,
  type LucideIcon,
} from "lucide-react";
import type { Connection } from "./api";
import { ApiCatalog } from "./ApiCatalog";
import { buildIntegrations, serviceRoot } from "./integrations";
import { CopyField, SectionCard, modelLabel, type Engine } from "./ui";

const integrationIcon: Record<string, LucideIcon> = {
  "claude-code": Bot,
  codex: Bot,
  opencode: Bot,
  openai: Code2,
  anthropic: Code2,
  curl: Terminal,
};

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
    <DashboardPage data-testid="api">
      <PageHeader
        title="API 接入"
        description="使用熟悉的 SDK 或程式代理，讓你的應用連接本機模型。"
      />
      <SectionCard
        icon={Globe}
        title="服務位址"
        description="貼進 SDK 或程式代理的 Base URL；範例命令使用下方選定的模型。"
      >
        <div className="grid gap-5 lg:grid-cols-[1fr_1fr_auto] lg:items-end">
          <CopyField label="OpenAI API Base URL" value={`${root}/v1`} />
          <CopyField label="Anthropic Base URL" value={root} />
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
                className="w-full lg:w-56"
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
        </div>
      </SectionCard>
      <SectionRow title="用戶端設定" />
      <div className="grid gap-4 lg:grid-cols-2" data-testid="integrations">
        {integrations.map((item) => (
          <SectionCard
            key={item.id}
            icon={integrationIcon[item.id] ?? KeyRound}
            title={item.title}
            description={item.description}
            className="min-w-0"
            data-testid={`integration-${item.id}`}
          >
            <CodeBlock language={item.language} filename={item.filename}>
              {item.code}
            </CodeBlock>
          </SectionCard>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        命令只引用環境變數 YUNSHU_AUTH_TOKEN（未設定時為
        local），不會寫入真實權杖。
      </p>
      <ApiCatalog connection={connection} />
    </DashboardPage>
  );
}
