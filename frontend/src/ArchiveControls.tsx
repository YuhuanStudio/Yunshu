import { Button, Switch } from "@yuhuanowo/yunui";
import { Download, Trash2 } from "lucide-react";
import { saveBlob } from "./admin-logs-api";
import { t, useLocale } from "./i18n/index.ts";
import { number } from "./i18n/format";
import {
  ARCHIVE_MAX_AGE_S,
  ARCHIVE_MAX_ROWS,
  exportEnvelope,
} from "./request-archive";
import type { Row } from "./RequestTrace";
import type { RequestArchive } from "./useRequestArchive";

/** Where request history lives, how long, and the switch to keep it across engine restarts. */
export function ArchiveControls({
  archive,
  capacity,
  shown,
  baseUrl,
}: {
  archive: RequestArchive;
  capacity: number;
  shown: readonly Row[];
  baseUrl: string;
}) {
  useLocale();
  const exportJson = () =>
    saveBlob(
      "yunshu-requests.json",
      new Blob([JSON.stringify(exportEnvelope(shown, baseUrl), null, 2)], {
        type: "application/json",
      }),
    );
  return (
    <div className="space-y-2" data-testid="archive-controls">
      <label className="flex items-center gap-2 text-sm text-foreground">
        <Switch
          checked={archive.enabled}
          onCheckedChange={archive.setEnabled}
          aria-label={t("requests.archive.label")}
        />
        {t("requests.archive.label")}
      </label>
      <p className="max-w-3xl">
        {t("requests.archive.help", {
          capacity: number(capacity, 0),
          rows: number(ARCHIVE_MAX_ROWS, 0),
          days: number(ARCHIVE_MAX_AGE_S / 86400, 0),
        })}
      </p>
      {archive.enabled && !archive.available && (
        <p role="status" className="text-warning">
          {t("requests.archive.unavailable")}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {archive.enabled && archive.archivedOnly > 0 && (
          <span data-testid="archive-count" className="tabular-nums">
            {t("requests.archive.count", {
              n: number(archive.archivedOnly, 0),
            })}
          </span>
        )}
        <span
          title={
            shown.length === 0
              ? t("requests.archive.exportDisabled")
              : undefined
          }
        >
          <Button
            size="sm"
            variant="ghost"
            disabled={shown.length === 0}
            onClick={exportJson}
          >
            <Download size={13} />
            {t("requests.archive.export")}
          </Button>
        </span>
        {archive.enabled && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void archive.clear()}
          >
            <Trash2 size={13} />
            {t("requests.archive.clear")}
          </Button>
        )}
      </div>
      <p className="max-w-3xl">{t("requests.archive.engineGap")}</p>
    </div>
  );
}
