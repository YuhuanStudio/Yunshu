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
import { t, useLocale } from "./i18n/index.ts";
import {
  LAUNCH_COMMAND,
  TOKEN_ENV,
  buildIntegrations,
  serviceRoot,
} from "./integrations";
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
  const locale = useLocale();
  // The descriptions are translated inside buildIntegrations, so the language is a dependency.
  const integrations = useMemo(
    () => buildIntegrations(root, model),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [root, model, locale],
  );
  return (
    <DashboardPage data-testid="api">
      <PageHeader
        title={t("api.page.title")}
        description={t("api.page.description")}
      />
      <SectionCard
        icon={Globe}
        title={t("api.address.title")}
        description={t("api.address.description")}
      >
        <div className="grid gap-5 lg:grid-cols-[1fr_1fr_auto] lg:items-end">
          <CopyField label={t("api.address.openai")} value={`${root}/v1`} />
          <CopyField label={t("api.address.anthropic")} value={root} />
          <div className="min-w-0">
            <p className="mb-1.5 text-xs text-muted-foreground">
              {t("api.address.modelLabel")}
            </p>
            <Select
              value={model || undefined}
              onValueChange={setModel}
              disabled={!models.length}
            >
              <SelectTrigger
                aria-label={t("api.address.modelAria")}
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
      <SectionRow title={t("api.clients.title")} />
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
            {LAUNCH_COMMAND[item.id] && (
              <div className="mb-3" data-testid={`launch-${item.id}`}>
                <p className="mb-1.5 text-xs text-muted-foreground">
                  {t("api.launch.label")}
                </p>
                <CodeBlock language="bash">{LAUNCH_COMMAND[item.id]}</CodeBlock>
              </div>
            )}
            <CodeBlock language={item.language} filename={item.filename}>
              {item.code}
            </CodeBlock>
          </SectionCard>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        {t("api.clients.note", { env: TOKEN_ENV, fallback: "local" })}{" "}
        {t("api.clients.tokenWhere", { env: TOKEN_ENV })}{" "}
        <a className="underline underline-offset-2" href="#/keys">
          {t("api.clients.keysLink")}
        </a>
      </p>
      <ApiCatalog connection={connection} />
    </DashboardPage>
  );
}
