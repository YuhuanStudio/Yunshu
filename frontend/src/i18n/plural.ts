import type { Locale } from "./types.ts";

const rules = new Map<string, Intl.PluralRules>();
export function pluralCategory(locale: Locale, n: number): string {
  let r = rules.get(locale);
  if (!r) {
    r = new Intl.PluralRules(locale);
    rules.set(locale, r);
  }
  return r.select(n);
}

type Part =
  string | { v: string } | { v: string; plural: Record<string, Part[]> };

const parsed = new Map<string, Part[]>();

function parse(src: string, from: number, inPlural: boolean): [Part[], number] {
  const out: Part[] = [];
  let buf = "";
  let i = from;
  const flush = () => {
    if (buf) out.push(buf);
    buf = "";
  };
  while (i < src.length) {
    const c = src[i];
    if (c === "}") break;
    if (c === "#" && inPlural) {
      flush();
      out.push({ v: "#" });
      i++;
    } else if (c === "{") {
      flush();
      let j = i + 1;
      while (j < src.length && src[j] !== "}" && src[j] !== ",") j++;
      const name = src.slice(i + 1, j).trim();
      if (src[j] === "}") {
        out.push({ v: name });
        i = j + 1;
        continue;
      }
      // {name, plural, one {...} other {...}}
      const kind = src.slice(j + 1, src.indexOf(",", j + 1)).trim();
      if (kind !== "plural") throw new Error(`i18n: unsupported "${kind}"`);
      j = src.indexOf(",", j + 1) + 1;
      const forms: Record<string, Part[]> = {};
      for (;;) {
        while (src[j] === " ") j++;
        if (src[j] === "}") break;
        let k = j;
        while (k < src.length && src[k] !== "{" && src[k] !== " ") k++;
        const cat = src.slice(j, k);
        while (src[k] === " ") k++;
        const [body, end] = parse(src, k + 1, true);
        forms[cat] = body;
        j = end + 1;
      }
      out.push({ v: name, plural: forms });
      i = j + 1;
    } else {
      buf += c;
      i++;
    }
  }
  flush();
  return [out, i];
}

function run(
  parts: Part[],
  vars: Record<string, unknown>,
  locale: Locale,
  hash: string,
): string {
  let s = "";
  for (const p of parts) {
    if (typeof p === "string") s += p;
    else if ("plural" in p) {
      const n = Number(vars[p.v]);
      const body =
        p.plural[`=${n}`] ??
        p.plural[pluralCategory(locale, n)] ??
        p.plural.other ??
        [];
      s += run(body, vars, locale, String(n));
    } else if (p.v === "#") s += hash;
    else s += vars[p.v] == null ? "" : String(vars[p.v]);
  }
  return s;
}

/** ICU-lite: `{name}` and `{n, plural, =0 {none} one {# item} other {# items}}`. */
export function interpolate(
  msg: string,
  vars: Record<string, unknown> | undefined,
  locale: Locale,
): string {
  if (!msg.includes("{")) return msg;
  let p = parsed.get(msg);
  if (!p) {
    p = parse(msg, 0, false)[0];
    parsed.set(msg, p);
  }
  return run(p, vars ?? {}, locale, "");
}
