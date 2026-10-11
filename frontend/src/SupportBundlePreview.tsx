import { useId, useState } from "react";
import { Button, Card } from "@yuhuanowo/yunui";
import { Eye, EyeOff } from "lucide-react";
import { t, useLocale } from "./i18n/index.ts";

/**
 * Shown before anything is shared: for each way of exporting diagnostics, what is included,
 * what is redacted and what is never there. The engine bundle is described from its fixed
 * specification (the engine has no manifest route), and the page copy is described from the
 * endpoints this page actually queried.
 */
export function SupportBundlePreview({ endpoints }: { endpoints: string[] }) {
  useLocale();
  const [open, setOpen] = useState(false);
  const id = useId();
  const groups = [
    {
      key: "engine",
      title: t("diagnostics.preview.engine"),
      included: t("diagnostics.preview.engine.included"),
      redacted: t("diagnostics.preview.engine.redacted"),
      excluded: t("diagnostics.preview.engine.excluded"),
    },
    {
      key: "page",
      title: t("diagnostics.preview.page"),
      included: t("diagnostics.preview.page.included", {
        endpoints: endpoints.length ? endpoints.join(", ") : "—",
      }),
      redacted: t("diagnostics.preview.page.redacted"),
      excluded: t("diagnostics.preview.page.excluded"),
    },
  ];
  const labels = {
    included: t("diagnostics.preview.included"),
    redacted: t("diagnostics.preview.redacted"),
    excluded: t("diagnostics.preview.excluded"),
  };
  return (
    <div data-testid="bundle-preview">
      <Button
        size="sm"
        variant="ghost"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <EyeOff size={13} /> : <Eye size={13} />}
        {t("diagnostics.preview.toggle")}
      </Button>
      {open && (
        <Card id={id} className="mt-2 min-w-0 space-y-4 p-4">
          <div>
            <h2 className="heading-md">{t("diagnostics.preview.title")}</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              {t("diagnostics.preview.desc")}
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {groups.map((g) => (
              <section key={g.key} data-bundle={g.key} className="min-w-0">
                <h3 className="text-sm font-semibold">{g.title}</h3>
                <dl className="mt-2 space-y-2 text-sm">
                  {(["included", "redacted", "excluded"] as const).map((k) => (
                    <div key={k} className="grid grid-cols-[4.5rem_1fr] gap-2">
                      <dt className="text-xs text-muted-foreground">
                        {labels[k]}
                      </dt>
                      <dd className="min-w-0 break-words">{g[k]}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            {t("diagnostics.preview.source")}
          </p>
        </Card>
      )}
    </div>
  );
}
