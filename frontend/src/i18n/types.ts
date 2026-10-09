import type zhTW from "./locales/zh-TW/index.ts";

export const LOCALES = ["zh-TW", "zh-CN", "en"] as const;
export type Locale = (typeof LOCALES)[number];
export const DEFAULT_LOCALE: Locale = "zh-TW";

/** Native names: a language is always named in itself. */
export const LOCALE_NAMES: Record<Locale, string> = {
  "zh-TW": "繁體中文",
  "zh-CN": "简体中文",
  en: "English",
};
/** BCP 47 tags for <html lang>, so fonts and screen readers pick the script. */
export const HTML_LANG: Record<Locale, string> = {
  "zh-TW": "zh-Hant-TW",
  "zh-CN": "zh-Hans-CN",
  en: "en",
};

/** The zh-TW dictionary is the source of truth; every other locale must match its shape. */
export type Dict = typeof zhTW;
export type Namespace = keyof Dict;
/** `shape` for translated files: same keys as the zh-TW file, string values. */
export type Shape<T> = { [K in keyof T]: string };
export type Key = {
  [N in Namespace]: `${N}.${keyof Dict[N] & string}`;
}[Namespace];
export type Vars = Record<string, string | number | null | undefined>;
