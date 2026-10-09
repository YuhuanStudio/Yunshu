from yunshu_engine.vlm_batch_runner import VLMBatchRunner, _source_image_digest


def _files(tmp_path, *bodies):
    paths = []
    for i, body in enumerate(bodies):
        p = tmp_path / f"{i}.bin"
        p.write_bytes(body)
        paths.append(str(p))
    return tuple(paths)


def test_same_bytes_same_key_even_from_different_paths(tmp_path):
    a = _files(tmp_path, b"image-one")
    (tmp_path / "copy").mkdir()
    b = _files(tmp_path / "copy", b"image-one")
    assert _source_image_digest(a, "cfg", 7) == _source_image_digest(b, "cfg", 7)


def test_bytes_config_token_index_and_order_all_separate_keys(tmp_path):
    p = _files(tmp_path, b"A", b"B")
    base = _source_image_digest(p, "cfg", 7)
    assert _source_image_digest(p[::-1], "cfg", 7) != base
    assert _source_image_digest(p, "cfg2", 7) != base
    assert _source_image_digest(p, "cfg", 8) != base
    assert _source_image_digest(p[:1], "cfg", 7) != base
    # length framing: moving a byte between files must not collide
    q = _files(tmp_path, b"AB", b"")
    assert _source_image_digest(q, "cfg", 7) != _source_image_digest(
        _files(tmp_path, b"A", b"B"), "cfg", 7
    )


def test_unreadable_source_falls_back_to_the_tensor_hash(tmp_path):
    assert _source_image_digest((str(tmp_path / "gone"),), "cfg", 1) is None

    class Apc:
        @staticmethod
        def hash_image_payload(pixel_values):
            return 99

    class Fut:
        @staticmethod
        def result():
            return None

    assert VLMBatchRunner._image_hash(Apc, object(), Fut) == 99
    assert VLMBatchRunner._image_hash(Apc, None, Fut) == 0
