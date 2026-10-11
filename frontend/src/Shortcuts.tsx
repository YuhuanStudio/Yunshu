import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  Kbd,
} from "@yuhuanowo/yunui";
import { t, tr, useLocale } from "./i18n/index.ts";
import { CHORDS } from "./route";

const GENERAL = [
  { keys: ["⌘", "K"], id: "palette" },
  { keys: ["?"], id: "help" },
  { keys: ["Esc"], id: "esc" },
  { keys: ["Esc"], id: "stop" },
  { keys: ["Enter"], id: "send" },
] as const;

/** The keyboard shortcuts sheet (`?`): what each key does, grouped. */
export function ShortcutsSheet({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  useLocale();
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent closeLabel={t("shell.keys.close")}>
        <DialogTitle>{t("shell.keys.title")}</DialogTitle>
        <DialogDescription>{t("shell.keys.desc")}</DialogDescription>
        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
          <div className="space-y-2">
            <h3 className="yunui-section-title text-base font-semibold">
              {t("shell.keys.general")}
            </h3>
            {GENERAL.map((k) => (
              <div
                key={k.id}
                className="flex items-center justify-between gap-3"
              >
                <dt className="text-muted-foreground">
                  {/* i18n-keys: shell.keys.item. */}
                  {tr(`shell.keys.item.${k.id}`)}
                </dt>
                <dd className="flex shrink-0 gap-1">
                  {k.keys.map((key) => (
                    <Kbd key={key}>{key}</Kbd>
                  ))}
                </dd>
              </div>
            ))}
          </div>
          <div className="space-y-2">
            <h3 className="yunui-section-title text-base font-semibold">
              {t("shell.keys.goto")}
            </h3>
            {Object.entries(CHORDS).map(([key, page]) => (
              <div
                key={key}
                className="flex items-center justify-between gap-3"
              >
                <dt className="text-muted-foreground">
                  {/* i18n-keys: shell.page. */}
                  {tr(`shell.page.${page}`)}
                </dt>
                <dd className="flex shrink-0 gap-1">
                  <Kbd>g</Kbd>
                  <Kbd>{key}</Kbd>
                </dd>
              </div>
            ))}
          </div>
        </dl>
      </DialogContent>
    </Dialog>
  );
}
