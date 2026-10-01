# wcx

A tiny word-count tool.

    python -m wcx [--top N] [-i | --ignore-case] FILE [FILE ...]

Prints `lines<TAB>words<TAB>chars<TAB>name` for each file, plus a `total` line when
more than one file is given.

Options:

- `--top N`: also print the N most frequent words, as `count<TAB>word`.
- `--ignore-case` / `-i`: with `--top`, count words case-insensitively.
