"""Tests for TTS text segmentation."""


class TestSplitTextSegments:
    def test_short_text(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        result = _split_text_segments("Hello world", 300)
        assert result == ["Hello world"]

    def test_exact_boundary(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "A" * 300
        result = _split_text_segments(text, 300)
        assert result == [text]

    def test_sentence_boundary(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "First sentence. Second sentence. Third sentence."
        result = _split_text_segments(text, 30)
        assert len(result) > 1
        for seg in result:
            assert len(seg) <= 30

    def test_comma_boundary(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "This is a long phrase, followed by another phrase, and yet another one."
        result = _split_text_segments(text, 35)
        assert len(result) > 1

    def test_chinese_sentence_boundary(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "第一句話。第二句話。第三句話。"
        result = _split_text_segments(text, 10)
        assert len(result) > 1

    def test_hard_split(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "A" * 500
        result = _split_text_segments(text, 300)
        assert len(result) == 2
        assert len(result[0]) <= 300
        assert len(result[1]) <= 300

    def test_whitespace_stripped(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "First part, second part, third part."
        result = _split_text_segments(text, 15)
        for seg in result:
            assert not seg.startswith(" ")

    def test_preserves_content(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "Hello world. This is a test. How are you doing today? Fine thanks!"
        result = _split_text_segments(text, 30)
        rejoined = " ".join(result)
        # Content preserved (minus stripped whitespace)
        for word in ["Hello", "test", "Fine"]:
            assert word in rejoined

    def test_single_long_word(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "a" * 600
        result = _split_text_segments(text, 300)
        assert len(result) == 2

    def test_custom_max_chars(self):
        from yunshu_gateway.routers.audio import _split_text_segments

        text = "First. Second. Third. Fourth. Fifth."
        result = _split_text_segments(text, 15)
        assert len(result) >= 3
