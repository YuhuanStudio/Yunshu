import { useState } from "react";
import { Button, Input, Kbd, PasswordInput, Switch } from "@yuhuanowo/yunui";
import {
  Banner,
  DashboardPage,
  PageHeader,
  SettingsShell,
} from "@yuhuanowo/yunui/patterns";
import {
  Keyboard,
  Link2,
  MemoryStick,
  Palette,
  Server,
  Globe,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import { StackRow } from "./stack-row";
import { t, useLocale } from "./i18n/index.ts";
import { LanguageSwitch } from "./LanguageSwitch";
import { ConfigView } from "./ConfigView";
import { CorsSection, NetworkSection, ServiceSection } from "./Service";
import {
  forgetRememberedToken,
  isTokenRemembered,
  setRememberedToken,
} from "./token-store";
import { ModelLeaseSettings } from "./ModelLeaseSettings";
import { SectionCard, type Engine } from "./ui";
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
  useLocale();
  const [url, setUrl] = useState(connection.baseUrl),
    [token, setToken] = useState(connection.token),
    [error, setError] = useState(""),
    [remember, setRemember] = useState(isTokenRemembered);
  function applyRemember(on: boolean) {
    setRemember(on);
    if (!on) forgetRememberedToken();
  }
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
        throw Error(t("settings.connection.invalid"));
      const baseUrl = url.trim().replace(/\/+$/, "");
      save({ baseUrl, token: token.trim() });
      if (remember && token.trim()) {
        if (!setRememberedToken(baseUrl, token.trim())) {
          setError(t("settings.connection.rememberFailed"));
          return;
        }
      } else if (!remember) forgetRememberedToken();
      setError("");
    } catch (e) {
      setError(
        e instanceof Error ? e.message : t("settings.connection.invalidShort"),
      );
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
    ["⌘ K / Ctrl K", t("settings.shortcuts.palette")],
    ["Enter", t("settings.shortcuts.send")],
    ["Esc", t("settings.shortcuts.close")],
  ];
  return (
    <DashboardPage data-testid="settings">
      <SettingsShell
        className="h-auto"
        sidebarClassName="border-r-0 bg-transparent [&_.nav-item.active]:bg-(--bg-card) [&_.nav-item.active]:shadow-sm [&_.nav-item.active]:ring-1 [&_.nav-item.active]:ring-border"
        contentClassName="overflow-visible"
        header={
          <PageHeader
            title={t("settings.title")}
            description={t("settings.description")}
          />
        }
        navigationLabel={t("settings.nav.label")}
        value={section}
        onValueChange={go}
        groups={[
          {
            key: "console",
            items: [
              {
                key: "connection",
                label: t("settings.nav.connection"),
                icon: Link2,
              },
              {
                key: "appearance",
                label: t("settings.nav.appearance"),
                icon: Palette,
              },
              {
                key: "models",
                label: t("settings.nav.models"),
                icon: MemoryStick,
              },
              {
                key: "config",
                label: t("settings.nav.config"),
                icon: SlidersHorizontal,
              },
              {
                key: "service",
                label: t("settings.nav.service"),
                icon: Server,
              },
              {
                key: "network",
                label: t("settings.nav.network"),
                icon: Globe,
              },
              {
                key: "shortcuts",
                label: t("settings.nav.shortcuts"),
                icon: Keyboard,
              },
            ],
          },
        ]}
      >
        <div className="min-w-0 space-y-6 sm:pl-6">
          <SectionCard
            id="settings-connection"
            icon={Link2}
            title={t("settings.connection.title")}
            description={t("settings.connection.description")}
            className="scroll-mt-4"
            bodyClassName="px-4 pb-4"
          >
            <StackRow
              title={
                <label htmlFor="base-url">{t("settings.connection.url")}</label>
              }
              description={t("settings.connection.urlHelp")}
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
            <StackRow
              title={
                <label htmlFor="access-token">
                  {t("settings.connection.token")}
                </label>
              }
              description={t("settings.connection.tokenHelp")}
              control={
                <PasswordInput
                  id="access-token"
                  className="w-full sm:w-72"
                  autoComplete="off"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder={t("settings.connection.tokenPlaceholder")}
                  labels={{
                    show: t("settings.connection.showToken"),
                    hide: t("settings.connection.hideToken"),
                  }}
                />
              }
            />
            <StackRow
              title={t("settings.connection.remember")}
              description={t("settings.connection.rememberHelp")}
              control={
                <Switch
                  label={t("settings.connection.remember")}
                  checked={remember}
                  onCheckedChange={applyRemember}
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
                {t("settings.connection.save")}
              </Button>
            </div>
          </SectionCard>
          <SectionCard
            id="settings-appearance"
            icon={Palette}
            title={t("settings.appearance.title")}
            className="scroll-mt-4"
            bodyClassName="px-4"
          >
            <StackRow
              title={t("settings.appearance.dark")}
              description={t("settings.appearance.darkHelp")}
              control={
                <Switch
                  label={t("settings.appearance.dark")}
                  checked={dark}
                  onCheckedChange={setDark}
                />
              }
            />
            <StackRow
              title={t("settings.appearance.language")}
              description={t("settings.appearance.languageHelp")}
              control={<LanguageSwitch variant="pill" />}
            />
          </SectionCard>
          <div id="settings-models" className="scroll-mt-4 space-y-6">
            <ModelLeaseSettings
              connection={connection}
              engine={engine}
              perform={perform}
              busy={disabled}
            />
            <Banner
              tone="neutral"
              icon={<ShieldCheck size={16} />}
              title={t("settings.permissions.title")}
              description={t("settings.permissions.description")}
            />
          </div>
          <div id="settings-config" className="scroll-mt-4">
            <ConfigView
              connection={connection}
              loadedModels={(engine.status?.models ?? [])
                .filter((m) => m.loaded)
                .map((m) => m.id)}
            />
          </div>
          <ServiceSection connection={connection} />
          <div id="settings-network" className="scroll-mt-4 space-y-6">
            <NetworkSection connection={connection} />
            <CorsSection connection={connection} />
          </div>
          <SectionCard
            id="settings-shortcuts"
            icon={Keyboard}
            title={t("settings.shortcuts.title")}
            className="scroll-mt-4"
            bodyClassName="px-4 pb-2"
          >
            {shortcuts.map(([keys, text]) => (
              <StackRow
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
          </SectionCard>
        </div>
      </SettingsShell>
    </DashboardPage>
  );
}

export { ApiView } from "./ApiAccess";
