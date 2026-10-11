import { useState } from "react";
import { LanguageSwitcher } from "@yuhuanowo/yunui/ai";
import {
  LOCALES,
  LOCALE_NAMES,
  isLocale,
  setLocale,
  t,
  useLocale,
} from "./i18n/index.ts";

/** The YunUI language picker wired to the console's locale store; switching never reloads. */
export function LanguageSwitch({
  variant = "pill",
  align = "right",
  className,
}: {
  variant?: "icon" | "pill";
  align?: "left" | "right";
  className?: string;
}) {
  const locale = useLocale();
  const [pending, setPending] = useState(false);
  return (
    <LanguageSwitcher
      locales={LOCALES.map((value) => ({ value, label: LOCALE_NAMES[value] }))}
      currentLocale={locale}
      onChange={(next) => {
        if (!isLocale(next)) return;
        setPending(true);
        void setLocale(next).finally(() => setPending(false));
      }}
      variant={variant}
      align={align}
      label={t("common.language.label")}
      pending={pending}
      className={className}
    />
  );
}
