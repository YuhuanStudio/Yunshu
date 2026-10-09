import { defineI18n } from "fumadocs-core/i18n";

export const i18n = defineI18n({
  defaultLanguage: "en",
  languages: ["en", "zh-TW", "zh-CN"],
  // The locale is part of the URL: a static export has no cookies or middleware.
  hideLocale: "never",
});

export type Lang = "en" | "zh-TW" | "zh-CN";
export const LANGS = i18n.languages as readonly Lang[];
export const LANG_NAMES: Record<Lang, string> = { en: "English", "zh-TW": "繁體中文", "zh-CN": "简体中文" };
export const DEFAULT_LANG: Lang = "zh-TW";
export function isLang(v: string): v is Lang {
  return (LANGS as readonly string[]).includes(v);
}
