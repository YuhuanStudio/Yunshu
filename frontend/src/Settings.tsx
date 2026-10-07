import { useState } from "react";
import { Button, Card, Input, Kbd, Switch } from "@yuhuanowo/yunui";
import {
  PageHeader,
  SettingRow,
  SettingsShell,
} from "@yuhuanowo/yunui/patterns";
import { Keyboard, Link2, MemoryStick, Palette } from "lucide-react";
import { ModelLeaseSettings } from "./ModelLeaseSettings";
import type { Engine } from "./ui";
import type { Perform } from "./Models";
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
  const [section, setSection] = useState("connection");
  const go = (key: string) => {
    setSection(key);
    document
      .getElementById(`settings-${key}`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const shortcuts: [string, string][] = [
    ["⌘ K / Ctrl K", "開啟命令面板，快速切換頁面與模型"],
    ["Enter", "在測試台送出訊息"],
    ["Esc", "關閉對話框與選單"],
  ];
  return (
    <section className="w-full max-w-5xl" data-testid="settings">
      <SettingsShell
        header={
          <PageHeader
            title="設定"
            description="連線到本機服務，並調整控制台偏好。"
          />
        }
        navigationLabel="設定分類"
        value={section}
        onValueChange={go}
        groups={[
          {
            key: "console",
            items: [
              { key: "connection", label: "引擎連線", icon: Link2 },
              { key: "appearance", label: "外觀", icon: Palette },
              { key: "models", label: "模型保留", icon: MemoryStick },
              { key: "shortcuts", label: "鍵盤快速鍵", icon: Keyboard },
            ],
          },
        ]}
      >
        <div className="space-y-6">
          <Card id="settings-connection" className="scroll-mt-4 px-5 pb-5">
            <h2 className="pt-5 text-sm font-semibold">引擎連線</h2>
            <SettingRow
              title={<label htmlFor="base-url">服務位址</label>}
              description="預設使用同一個服務來源。開發模式由 Vite 轉送至本機 8000 埠。"
              control={
                <Input
                  id="base-url"
                  className="w-full font-mono sm:w-72"
                  value={url}
                  onChange={(e) => {
                    setUrl(e.target.value);
                    setToken("");
                  }}
                  placeholder="http://127.0.0.1:8000"
                />
              }
            />
            <SettingRow
              title={<label htmlFor="access-token">存取權杖</label>}
              description="只保留在此頁記憶體；重新整理後需要再輸入。更改服務位址會清除權杖。"
              control={
                <Input
                  id="access-token"
                  className="w-full sm:w-72"
                  type="password"
                  autoComplete="off"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="服務沒有啟用驗證時可留空"
                />
              }
            />
            {error && (
              <p role="alert" className="pt-2 text-sm text-error">
                {error}
              </p>
            )}
            <div className="pt-4">
              <Button disabled={disabled} onClick={submit}>
                儲存並連線
              </Button>
            </div>
          </Card>
          <Card id="settings-appearance" className="scroll-mt-4 px-5">
            <SettingRow
              title="深色介面"
              description="儲存於此瀏覽器。"
              control={
                <Switch
                  label="深色介面"
                  checked={dark}
                  onCheckedChange={setDark}
                />
              }
            />
          </Card>
          <div id="settings-models" className="scroll-mt-4 space-y-6">
            <ModelLeaseSettings
              connection={connection}
              engine={engine}
              perform={perform}
              busy={disabled}
            />
            <Card className="space-y-2 p-5">
              <h2 className="text-sm font-semibold">模型操作權限</h2>
              <p className="text-xs leading-6 text-muted-foreground">
                模型載入與卸載需要服務允許的權限。若出現 401，請使用服務設定的
                YUNSHU_AUTH_TOKEN。此頁不會修改引擎啟動參數或關閉驗證。
              </p>
            </Card>
          </div>
          <Card id="settings-shortcuts" className="scroll-mt-4 px-5 pb-2">
            <h2 className="pt-5 text-sm font-semibold">鍵盤快速鍵</h2>
            {shortcuts.map(([keys, text]) => (
              <SettingRow
                key={keys}
                title={text}
                control={
                  <span className="flex items-center gap-1">
                    {keys.split(" ").map((k, i) =>
                      k === "/" ? (
                        <span key={i} className="text-xs text-muted-foreground">
                          /
                        </span>
                      ) : (
                        <Kbd key={i}>{k}</Kbd>
                      ),
                    )}
                  </span>
                }
              />
            ))}
          </Card>
        </div>
      </SettingsShell>
    </section>
  );
}

export { ApiView } from "./ApiAccess";
