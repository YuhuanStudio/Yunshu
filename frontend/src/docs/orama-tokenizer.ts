/** Same deterministic tokenizer at build time and in the browser (and in the Python
 * docs Ask AI ranking: python/yunshu_console/docs_ask.py, pinned by
 * tests/fixtures/docsask/orama.json). CJK bigrams avoid an English stemmer silently
 * dropping Traditional/Simplified Chinese. */

/** Function words carry no topic; with prefix matching "i" or "do" match everything. */
export const STOP_WORDS: ReadonlySet<string> = new Set(
  (
    "a an and are as at be by can do does did for from has have how i in is it its " +
    "me my of on or so that the this to was we what when where which who why will " +
    "with you your"
  ).split(" "),
);
/** Han question/function words (both scripts) split a Han run instead of becoming
 * tokens. Lone Han characters inside a longer run are not indexed: bigrams carry the
 * words, single characters matched everything. */
export const STOP_HAN: ReadonlySet<string> = new Set(
  // i18n-ignore: retrieval data, not UI text
  "如何 怎麼 怎么 怎樣 怎样 什麼 什么 是否 可以 能否 哪些 哪個 哪个 請問 请问 一個 一个 我們 我们 你們 你们 以及 或者 還是 还是".split(
    " ",
  ),
);
export const HAN_STOP_SPLIT = new RegExp([...STOP_HAN].join("|"), "u");
export const splitHanRuns = (word: string): string[] =>
  word.split(new RegExp(HAN_STOP_SPLIT.source, "gu"));

export function tokenizeDocs(input: string): string[] {
  const tokens =
    input.toLowerCase().match(/[a-z0-9_]+|[\p{Script=Han}]+/gu) ?? [];
  return [
    ...new Set(
      tokens.flatMap((word) => {
        if (!/\p{Script=Han}/u.test(word))
          return STOP_WORDS.has(word) ? [] : [word];
        const out: string[] = [];
        for (const run of splitHanRuns(word)) {
          const chars = Array.from(run);
          if (chars.length === 1) out.push(run);
          for (let i = 0; i + 1 < chars.length; i++)
            out.push(chars[i] + chars[i + 1]);
        }
        return out;
      }),
    ),
  ];
}
