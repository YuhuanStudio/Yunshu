/** Same deterministic tokenizer at build time and in the browser. CJK bigrams
 * avoid an English stemmer silently dropping Traditional/Simplified Chinese. */
export function tokenizeDocs(input: string): string[] {
  const tokens =
    input.toLowerCase().match(/[a-z0-9_]+|[\p{Script=Han}]+/gu) ?? [];
  return [
    ...new Set(
      tokens.flatMap((word) =>
        /\p{Script=Han}/u.test(word)
          ? Array.from(word).flatMap((c, i, chars) =>
              i + 1 < chars.length ? [c, c + chars[i + 1]] : [c],
            )
          : [word],
      ),
    ),
  ];
}
