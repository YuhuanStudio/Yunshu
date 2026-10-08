import { useState } from "react";
import {
  Button,
  FileDropzone,
  IconButton,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Sheet,
} from "@yuhuanowo/yunui";
import { Download, GitBranch, Pencil, Trash2, Upload } from "lucide-react";
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
  onRename,
  onOverwrite,
  onBranch,
  onExport,
  onImport,
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
  onRename: (id: string, name: string) => void;
  onOverwrite: (id: string) => void;
  onBranch: (entry: HistoryEntry, keep: number) => void;
  onExport: () => void;
  onImport: (file: File) => void;
}) {
  useLocale();
  const [tab, setTab] = useState<Tab>("presets");
  const [name, setName] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [branching, setBranching] = useState<string | null>(null);
  const [branchKeep, setBranchKeep] = useState("");
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
                  className="flex flex-wrap items-center gap-3 rounded-lg px-2 py-2"
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
                    icon={<Pencil size={14} />}
                    label={t("playground.library.preset.edit", {
                      name: p.name,
                    })}
                    onClick={() => {
                      setEditing(editing === p.id ? null : p.id);
                      setEditName(p.name);
                    }}
                  />
                  <IconButton
                    icon={<Trash2 size={14} />}
                    label={t("playground.library.preset.delete", {
                      name: p.name,
                    })}
                    onClick={() => onDeletePreset(p.id)}
                  />
                  {editing === p.id && (
                    <form
                      className="flex w-full basis-full flex-wrap items-center gap-2"
                      data-testid="preset-editor"
                      onSubmit={(e) => {
                        e.preventDefault();
                        if (!editName.trim()) return;
                        onRename(p.id, editName.trim());
                        setEditing(null);
                      }}
                    >
                      <Input
                        className="min-w-0 flex-1"
                        aria-label={t("playground.library.preset.rename")}
                        value={editName}
                        maxLength={80}
                        onChange={(e) => setEditName(e.target.value)}
                      />
                      <Button
                        type="submit"
                        size="sm"
                        variant="ghost"
                        disabled={
                          !editName.trim() || editName.trim() === p.name
                        }
                      >
                        {t("playground.library.preset.renameSave")}
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          onOverwrite(p.id);
                          setEditing(null);
                        }}
                      >
                        {t("playground.library.preset.overwrite")}
                      </Button>
                    </form>
                  )}
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
                    className="flex flex-wrap items-center gap-3 rounded-lg px-2 py-2"
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
                      icon={<GitBranch size={14} />}
                      label={t("playground.library.history.branch", {
                        title: h.title,
                      })}
                      onClick={() => {
                        setBranching(branching === h.id ? null : h.id);
                        setBranchKeep(String(h.messages.length));
                      }}
                    />
                    <IconButton
                      icon={<Trash2 size={14} />}
                      label={t("playground.library.history.delete", {
                        title: h.title,
                      })}
                      onClick={() => onDeleteHistory(h.id)}
                    />
                    {branching === h.id && (
                      <div
                        className="flex w-full basis-full flex-wrap items-center gap-2"
                        data-testid="branch-editor"
                      >
                        <Select
                          value={branchKeep}
                          onValueChange={setBranchKeep}
                        >
                          <SelectTrigger
                            aria-label={t(
                              "playground.library.history.branchAt",
                              { n: "" },
                            )}
                            className="w-44"
                          >
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {h.messages.flatMap((m, i) =>
                              m.role === "assistant"
                                ? [
                                    <SelectItem key={i} value={String(i + 1)}>
                                      {t(
                                        "playground.library.history.branchAt",
                                        {
                                          n: (i + 1) / 2,
                                        },
                                      )}
                                    </SelectItem>,
                                  ]
                                : [],
                            )}
                          </SelectContent>
                        </Select>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            onBranch(h, Number(branchKeep));
                            setBranching(null);
                          }}
                        >
                          {t("playground.library.history.branchCreate")}
                        </Button>
                      </div>
                    )}
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
        <div className="space-y-2" data-testid="library-transfer">
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="ghost" onClick={onExport}>
              <Download size={13} />
              {t("playground.library.transfer.export")}
            </Button>
            <FileDropzone
              accept="application/json,.json"
              onFiles={(files) => files[0] && onImport(files[0])}
              className="w-auto min-h-0 flex-row gap-1.5 border-0 bg-transparent px-2 py-1 text-sm"
            >
              <Upload size={13} />
              {t("playground.library.transfer.import")}
            </FileDropzone>
          </div>
          <p className="text-xs text-muted-foreground">
            {t("playground.library.transfer.help")}
          </p>
        </div>
      </div>
    </Sheet>
  );
}
