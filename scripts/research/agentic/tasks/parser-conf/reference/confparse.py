import re

KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
NUM = re.compile(r"[+-]?\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?(?:[eE][+-]?\d+)?")


class ParseError(ValueError):
    def __init__(self, message, line):
        super().__init__(f"line {line}: {message}")
        self.line = line


def _string(s, i, ln):
    out = []
    i += 1
    while i < len(s):
        c = s[i]
        if c == '"':
            return "".join(out), i + 1
        if c == "\\":
            i += 1
            e = s[i] if i < len(s) else ""
            m = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
            if e not in m:
                raise ParseError("bad escape", ln)
            out.append(m[e])
        else:
            out.append(c)
        i += 1
    raise ParseError("unterminated string", ln)


def _value(s, i, ln):
    while i < len(s) and s[i] in " \t":
        i += 1
    if i >= len(s):
        raise ParseError("missing value", ln)
    c = s[i]
    if c == '"':
        return _string(s, i, ln)
    if c == "'":
        j = s.find("'", i + 1)
        if j < 0:
            raise ParseError("unterminated string", ln)
        return s[i + 1 : j], j + 1
    if c == "[":
        items = []
        i += 1
        while True:
            while i < len(s) and s[i] in " \t":
                i += 1
            if i >= len(s):
                raise ParseError("unterminated array", ln)
            if s[i] == "]":
                return items, i + 1
            v, i = _value(s, i, ln)
            items.append(v)
            while i < len(s) and s[i] in " \t":
                i += 1
            if i < len(s) and s[i] == ",":
                i += 1
                continue
            if i < len(s) and s[i] == "]":
                return items, i + 1
            raise ParseError("bad array", ln)
    if s.startswith("true", i) and not s[i + 4 : i + 5].isalnum():
        return True, i + 4
    if s.startswith("false", i) and not s[i + 5 : i + 6].isalnum():
        return False, i + 5
    m = NUM.match(s, i)
    if m:
        t = m.group(0).replace("_", "")
        if "." in t or "e" in t or "E" in t:
            return float(t), m.end()
        return int(t), m.end()
    raise ParseError("invalid value", ln)


def _tail(s, i, ln):
    rest = s[i:].strip()
    if rest and not rest.startswith("#"):
        raise ParseError("unexpected text after value", ln)


def parse(text):
    root = {}
    cur = root
    defined = set()
    for ln, raw in enumerate(text.split("\n"), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("["):
            end = s.find("]")
            if end < 0:
                raise ParseError("bad section header", ln)
            _tail(s, end + 1, ln)
            parts = [p.strip() for p in s[1:end].split(".")]
            if not all(KEY.fullmatch(p) for p in parts):
                raise ParseError("bad section name", ln)
            path = tuple(parts)
            if path in defined:
                raise ParseError("duplicate section", ln)
            defined.add(path)
            cur = root
            for p in parts:
                if p not in cur:
                    cur[p] = {}
                elif not isinstance(cur[p], dict):
                    raise ParseError("name conflicts with a value", ln)
                cur = cur[p]
            continue
        m = KEY.match(s)
        if not m:
            raise ParseError("invalid line", ln)
        rest = s[m.end() :].lstrip()
        if not rest.startswith("="):
            raise ParseError("expected '='", ln)
        v, i = _value(rest, 1, ln)
        _tail(rest, i, ln)
        if m.group(0) in cur:
            raise ParseError("duplicate key", ln)
        cur[m.group(0)] = v
    return root
