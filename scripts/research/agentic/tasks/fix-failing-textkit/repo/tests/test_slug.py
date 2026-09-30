from textkit import slugify


def test_basic():
    assert slugify("Hello, World") == "hello-world"


def test_accents_and_edges():
    assert slugify("  Café Déjà Vu!  ") == "cafe-deja-vu"


def test_collapse():
    assert slugify("a --- b") == "a-b"
