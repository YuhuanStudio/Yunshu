# Tiny config format

`confparse.parse(text)` turns a config document into nested `dict`s. Standard library only.

## Lines

Each line is one of: blank (ignored), a comment (first non-space character is `#`, ignored),
a section header, or a `key = value` pair. Leading and trailing whitespace on a line is ignored.

## Sections

`[name]` starts a table; `[a.b]` starts the table `b` inside table `a`, creating `a` if needed.
Names are dot-separated keys (see Keys). Whitespace around the name and around dots is ignored:
`[ a . b ]` is `[a.b]`. Key/value pairs before the first section belong to the root table.
Defining the same section twice is an error, but `[a.b]` may come before `[a]`, and `[a]` after
`[a.b]` is fine (it was only created implicitly). If a name on the path already holds a
non-table value, that is an error.

## Keys

A key matches `[A-Za-z_][A-Za-z0-9_-]*`. A key may appear only once per table; a repeat is an error.
Whitespace around `=` is optional.

## Values

- integer: optional `-` or `+`, digits, with single underscores allowed between digits (`1_000` is 1000).
- float: like an integer, then `.` and digits, and/or an exponent `e`/`E` with optional sign and digits
  (`1.5`, `-0.25`, `2e3`, `1_0.5` is 10.5). An exponent alone (`2e3`) is a float (2000.0).
- boolean: `true` or `false` (lower case only).
- basic string: `"..."` with escapes `\n`, `\t`, `\"`, `\\`. Any other escape is an error.
  A newline cannot occur inside (a line is one pair), so an unterminated string is an error.
- literal string: `'...'` taken verbatim, no escapes.
- array: `[` values separated by `,` `]`, elements may be any of the above scalars or nested arrays.
  Whitespace is free between tokens, `[]` is the empty array, one trailing comma is allowed
  (`[1, 2,]`), and a comma with no element (`[,]`, `[1,,2]`) is an error. Arrays are single-line.

A `#` outside a string starts a comment that runs to the end of the line, after a value or a
section header as well as on its own line. After the value only whitespace or a comment may follow.

## Errors

`ParseError` is a subclass of `ValueError` with attribute `line` (1-based line number of the
problem) and a message that contains `line N`. Errors include: a line that is neither blank,
comment, header nor pair; bad section header; invalid key; missing value; invalid value;
duplicate key; duplicate section; a name conflicting with a non-table value; anything after a value.

Examples:

    name = "x"          -> {"name": "x"}
    [server.http]
    port = 8_080        -> {"server": {"http": {"port": 8080}}}
