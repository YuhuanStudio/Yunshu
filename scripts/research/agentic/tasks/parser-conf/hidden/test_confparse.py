import pytest

from confparse import ParseError, parse


def test_empty_and_comments():
    assert parse("") == {}
    assert parse("\n  # hello\n\n   \n") == {}


def test_scalars():
    d = parse(
        "a = 1\nb=-2\nc = +3\nd = 1_000\ne = 1.5\nf = -0.25\ng = 2e3\nh = 1_0.5\ni = 1.5E-2\n"
    )
    assert d == {
        "a": 1,
        "b": -2,
        "c": 3,
        "d": 1000,
        "e": 1.5,
        "f": -0.25,
        "g": 2000.0,
        "h": 10.5,
        "i": 0.015,
    }
    assert isinstance(d["a"], int) and isinstance(d["g"], float)


def test_booleans_and_strings():
    d = parse(
        't = true\nf = false\ns = "a\\tb\\n\\"q\\" \\\\"\nl = \'raw \\n # not a comment\'\n'
    )
    assert d["t"] is True and d["f"] is False
    assert d["s"] == 'a\tb\n"q" \\'
    assert d["l"] == "raw \\n # not a comment"


def test_hash_inside_string_is_not_comment():
    assert parse('x = "a # b" # real comment') == {"x": "a # b"}


def test_arrays():
    d = parse(
        "a = []\nb = [1, 2,3]\nc = [ 'x' , \"y\", true ]\nd = [[1, 2], [3], []]\ne = [1, 2,]\n"
    )
    assert d == {
        "a": [],
        "b": [1, 2, 3],
        "c": ["x", "y", True],
        "d": [[1, 2], [3], []],
        "e": [1, 2],
    }


def test_sections_nested():
    d = parse(
        "top = 1\n[server]\nhost = 'h'\n[server.http]\nport = 8_080\n[ other . deep . er ]\nz = false\n"
    )
    assert d == {
        "top": 1,
        "server": {"host": "h", "http": {"port": 8080}},
        "other": {"deep": {"er": {"z": False}}},
    }


def test_section_order_implicit_parent():
    d = parse("[a.b]\nx = 1\n[a]\ny = 2\n")
    assert d == {"a": {"b": {"x": 1}, "y": 2}}


def test_key_order_preserved():
    assert list(parse("z = 1\na = 2\nm = 3")) == ["z", "a", "m"]


def test_trailing_comments():
    assert parse("[s] # c\nk = 5 # c\n") == {"s": {"k": 5}}


def test_key_chars():
    assert parse("_a-b9 = 1\nA_1 = 2") == {"_a-b9": 1, "A_1": 2}


def test_crlf_lines():
    assert parse("a = 1\r\nb = 2\r\n") == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "text,line",
    [
        ("a = 1\na = 2", 2),
        ("[s]\n[s]", 2),
        ("[a.b]\n[a.b]", 2),
        ("x = 1\n[x]", 2),
        ("x = 1\n[x.y]", 2),
        ("what", 1),
        ("a", 1),
        ("a =", 1),
        ("a = @", 1),
        ("1a = 3", 1),
        ("a-b c = 3", 1),
        ("ok = 1\n\nbad line here", 3),
        ("[]", 1),
        ("[a", 1),
        ("[a b]", 1),
        ("[a.]", 1),
        ("[a] junk", 1),
        ("a = 1 2", 1),
        ("a = 'x", 1),
        ('a = "x', 1),
        ('a = "\\q"', 1),
        ("a = TRUE", 1),
        ("a = 1__0", 1),
        ("a = [1,,2]", 1),
        ("a = [,]", 1),
        ("a = [1, 2", 1),
        ("a = 1.", 1),
        ("a = --1", 1),
        ("# c\n\n[s]\nk = 1\nk = 2", 5),
    ],
)
def test_errors(text, line):
    with pytest.raises(ParseError) as ei:
        parse(text)
    assert ei.value.line == line
    assert f"line {line}" in str(ei.value)
    assert isinstance(ei.value, ValueError)


def test_realistic_document():
    doc = """
# service config
name = "svc"
retries = 3
ratio = 0.75

[db]
url = 'postgres://u:p@h/db'
pool = [1, 5, 10]
ssl = true

[db.replica]
hosts = ["a", "b"]  # two of them
weights = [0.5, 0.5]
"""
    assert parse(doc) == {
        "name": "svc",
        "retries": 3,
        "ratio": 0.75,
        "db": {
            "url": "postgres://u:p@h/db",
            "pool": [1, 5, 10],
            "ssl": True,
            "replica": {"hosts": ["a", "b"], "weights": [0.5, 0.5]},
        },
    }
