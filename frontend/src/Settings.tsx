import { useState } from "react";
import { Button, Card, Input, Switch } from "@yuhuanowo/yunui";
import { PageHeader, SettingRow, CodeBlock } from "@yuhuanowo/yunui/patterns";
import { ModelLeaseSettings } from "./ModelLeaseSettings";
import type { Engine } from "./ui";
import type { Perform } from "./Models";
import { ApiCatalog } from "./ApiCatalog";
import type { Connection } from "./api";
export function Settings({
  connection,
  save,
  dark,
  setDark,
  disabled,
  engine,
  perform,
}: {
  connection: Connection;
  save: (next: Connection) => void;
  dark: boolean;
  setDark: (v: boolean) => void;
  disabled: boolean;
  engine: Engine;
  perform: Perform;
}) {
  const [url, setUrl] = useState(connection.baseUrl),
    [token, setToken] = useState(connection.token),
    [error, setError] = useState("");
  function submit() {
    try {
      const parsed = new URL(url.trim());
      if (
        !["http:", "https:"].includes(parsed.protocol) ||
        parsed.username ||
        parsed.password ||
        parsed.search ||
        parsed.hash
      )
        throw Error("請使用不含帳密、查詢參數或錨點的 HTTP(S) 服務位址。");
      save({ baseUrl: url.trim().replace(/\/+$/, ""), token: token.trim() });
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "服務位址無效");
    }
  }
  return (
    <section className="w-full max-w-4xl space-y-6" data-testid="settings">
      <PageHeader
        title="設定"
        description="連線到本機服務，並調整控制台偏好。"
      />
      <Card className="space-y-5 p-5">
        <div>
          <h2 className="text-sm font-semibold">引擎連線</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            預設使用同一個服務來源。開發模式由 Vite 轉送至本機 8000 埠。
          </p>
        </div>
        <div>
          <label htmlFor="base-url" className="text-sm">
            服務位址
          </label>
          <Input
            id="base-url"
            className="mt-2 font-mono"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              setToken("");
            }}
            placeholder="http://127.0.0.1:8000"
          />
        </div>
        <div>
          <label htmlFor="access-token" className="text-sm">
            存取權杖
          </label>
          <Input
            id="access-token"
            className="mt-2"
            type="password"
            autoComplete="off"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="服務沒有啟用驗證時可留空"
          />
          <p className="mt-2 text-xs text-muted-foreground">
            只保留在此頁記憶體；重新整理後需要再輸入。更改服務位址會清除權杖。
          </p>
        </div>
        {error && (
          <p role="alert" className="text-sm text-error">
            {error}
          </p>
        )}
        <Button disabled={disabled} onClick={submit}>
          儲存並連線
        </Button>
      </Card>
      <ModelLeaseSettings
        connection={connection}
        engine={engine}
        perform={perform}
        busy={disabled}
      />
      <Card className="px-5">
        <SettingRow
          title="深色介面"
          description="儲存於此瀏覽器。"
          control={
            <Switch label="深色介面" checked={dark} onCheckedChange={setDark} />
          }
        />
      </Card>
      <Card className="space-y-3 p-5">
        <h2 className="text-sm font-semibold">模型操作權限</h2>
        <p className="text-xs leading-6 text-muted-foreground">
          模型載入與卸載需要服務允許的權限。若出現 401，請使用服務設定的
          YUNSHU_AUTH_TOKEN。此頁不會修改引擎啟動參數或關閉驗證。
        </p>
      </Card>
    </section>
  );
}
export function ApiView({ connection }: { connection: Connection }) {
  const base =
    connection.baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "") + "/v1";
  return (
    <section className="w-full max-w-5xl space-y-6" data-testid="api">
      <PageHeader
        title="API 接入"
        description="使用熟悉的 SDK，讓你的應用連接本機模型。"
      />
      <Card className="space-y-2 p-5">
        <p className="text-xs text-muted-foreground">OpenAI API Base URL</p>
        <p className="break-all font-mono text-sm">{base}</p>
      </Card>
      <CodeBlock
        code=""
        tabs={[
          {
            id: "python",
            label: "Python",
            language: "python",
            code: `import os\nfrom openai import OpenAI\n\nclient = OpenAI(\n    base_url=${JSON.stringify(base)},\n    api_key=os.environ.get("YUNSHU_AUTH_TOKEN", "local"),\n)\nmodels = client.models.list()\nprint([model.id for model in models.data])`,
          },
          {
            id: "curl",
            label: "cURL",
            language: "bash",
            code: `curl ${JSON.stringify(base + "/models")} \\\n  -H "Authorization: Bearer $YUNSHU_AUTH_TOKEN"`,
          },
        ]}
      />
      <ApiCatalog connection={connection} />
    </section>
  );
}
