import { useState } from "react";
import { Button, IconButton, Input, Sheet } from "@yuhuanowo/yunui";
import { Trash2 } from "lucide-react";
import { SegmentedTray } from "./SegmentedTray";
import { fixed, number } from "./ui";
import { relative } from "./i18n/format.ts";
import { t, tr, useLocale } from "./i18n/index.ts";
import { newId, type HistoryEntry, type Preset } from "./playground-library.ts";

// i18n-keys: playground.library.builtin.
const builtinName = (id: string) => tr(`playground.library.builtin.${id}`);
type Tab = "presets" | "history";

/** The built-in presets: parameters only, the names come from the dictionary. */
export const BUILTIN_PRESETS: readonly {
  id: "precise" | "creative" | "json";
  temperature: number;
  format: "text" | "json";
}[] = [
  { id: "precise", temperature: 0.2, format: "text" },
  { id: "creative", temperature: 1.0, format: "text" },
  { id: "json", temperature: 0.2, format: "json" },
];

/**
 * Saved presets (system prompt + parameters + model + format) and the local conversation list,
 * in one sheet. All state lives in the caller; this only renders and reports choices.
 */
export function PlaygroundLibrary({
  open,
  onClose,
  presets,
  history,
  stored,
  currentPreset,
  onApplyBuiltin,
  onApply,
  onSave,
  onDeletePreset,
  onResume,
  onDeleteHistory,
  onClearHistory,
}: {
  open: boolean;
  onClose: () => void;
  presets: readonly Preset[];
  history: readonly HistoryEntry[];
  /** false when the browser refused the last write. */
  stored: boolean;
  currentPreset: () => Omit<Preset, "id" | "name">;
  onApplyBuiltin: (id: (typeof BUILTIN_PRESETS)[number]["id"]) => void;
  onApply: (preset: Preset) => void;
  onSave: (preset: Preset) => void;
  onDeletePreset: (id: string) => void;
  onResume: (entry: HistoryEntry) => void;
  onDeleteHistory: (id: string) => void;
  onClearHistory: () => void;
}) {
  useLocale();
  const [tab, setTab] = useState<Tab>("presets");
  const [name, setName] = useState("");
  const save = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    onSave({ ...currentPreset(), id: newId(), name: trimmed });
    setName("");
  };
  return (
    <Sheet
      open={open}
      onClose={onClose}
      title={t("playground.library.title")}
      closeLabel={t("playground.params.close")}
    >
      <div className="space-y-5" data-testid="playground-library">
        <SegmentedTray
          aria-label={t("playground.library.title")}
          value={tab}
          onChange={setTab}
          fillOnPhone
          options={[
            { value: "presets", label: t("playground.library.tab.presets") },
            { value: "history", label: t("playground.library.tab.history") },
          ]}
        />
        {!stored && (
          <p role="status" className="text-xs text-muted-foreground">
            {t("playground.library.storageFailed")}
          </p>
        )}
        {tab === "presets" ? (
          <>
            <ul
              className="space-y-1"
              aria-label={t("playground.library.tab.presets")}
            >
              {BUILTIN_PRESETS.map((p) => (
                <li
                  key={p.id}
                  className="flex items-center gap-3 rounded-lg px-2 py-2"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">
                      {builtinName(p.id)}
                    </span>
                    <span className="block text-xs tabular-nums text-muted-foreground">
                      {t("playground.library.builtin.summary", {
                        temperature: fixed(p.temperature, 1),
                      })}
                    </span>
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => onApplyBuiltin(p.id)}
                  >
                    {t("playground.library.preset.apply")}
                  </Button>
                </li>
              ))}
              {presets.map((p) => (
                <li
                  key={p.id}
                  data-testid="saved-preset"
                  className="flex items-center gap-3 rounded-lg px-2 py-2"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{p.name}</span>
                    <span className="block text-xs tabular-nums text-muted-foreground">
                      {t("playground.library.preset.summary", {
                        temperature: fixed(p.temperature, 1),
                        tokens: number(p.maxTokens, 0),
                      })}
                    </span>
                  </span>
                  <Button size="sm" variant="ghost" onClick={() => onApply(p)}>
                    {t("playground.library.preset.apply")}
                  </Button>
                  <IconButton
                    icon={<Trash2 size={14} />}
                    label={t("playground.library.preset.delete", {
                      name: p.name,
                    })}
                    onClick={() => onDeletePreset(p.id)}
                  />
                </li>
              ))}
            </ul>
            {!presets.length && (
              <p className="text-xs text-muted-foreground">
                {t("playground.library.preset.empty")}
              </p>
            )}
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                save();
              }}
            >
              <Input
                aria-label={t("playground.library.preset.name")}
                placeholder={t("playground.library.preset.namePlaceholder")}
                value={name}
                maxLength={80}
                onChange={(e) => setName(e.target.value)}
              />
              <Button type="submit" size="sm" disabled={!name.trim()}>
                {t("playground.library.preset.save")}
              </Button>
            </form>
          </>
        ) : (
          <>
            {history.length ? (
              <ul className="space-y-1" data-testid="history-list">
                {history.map((h) => (
                  <li
                    key={h.id}
                    className="flex items-center gap-3 rounded-lg px-2 py-2"
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm">
                        {h.title || t("playground.library.history.untitled")}
                      </span>
                      <span className="block truncate text-xs tabular-nums text-muted-foreground">
                        {t("playground.library.history.messages", {
                          count: h.messages.length,
                        })}
                        {" · "}
                        {relative((Date.now() - h.at) / 1000)}
                      </span>
                    </span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => onResume(h)}
                    >
                      {t("playground.library.history.resume")}
                    </Button>
                    <IconButton
                      icon={<Trash2 size={14} />}
                      label={t("playground.library.history.delete", {
                        title: h.title,
                      })}
                      onClick={() => onDeleteHistory(h.id)}
                    />
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-muted-foreground">
                {t("playground.library.history.empty")}
              </p>
            )}
            {history.length > 0 && (
              <Button size="sm" variant="ghost" onClick={onClearHistory}>
                {t("playground.library.history.clear")}
              </Button>
            )}
          </>
        )}
      </div>
    </Sheet>
  );
}
