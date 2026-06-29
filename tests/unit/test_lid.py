"""Tests for Language Identification (LID)."""


class TestDetectLanguage:
    def test_detect_chinese(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("你好世界這是一個測試")
        assert result.language == "zh"
        assert result.confidence > 0

    def test_detect_english(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("The quick brown fox jumps over the lazy dog")
        assert result.language == "en"
        assert result.confidence > 0

    def test_detect_japanese(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("こんにちは世界")
        assert result.language == "ja"

    def test_detect_korean(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("안녕하세요 세계")
        assert result.language == "ko"

    def test_detect_arabic(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("مرحبا بالعالم")
        assert result.language == "ar"

    def test_detect_russian(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("Привет мир")
        assert result.language == "ru"

    def test_empty_text(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("")
        assert result.language == "und"
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("   ")
        assert result.language == "und"

    def test_french_detection(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("Le chat est sur la table dans le jardin")
        assert result.language == "fr"

    def test_german_detection(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("Der Hund ist in dem Haus und der Garten")
        assert result.language == "de"

    def test_result_has_all_scores(self):
        from yunshu_engine.lid import detect_language

        result = detect_language("Hello world this is a test")
        assert isinstance(result.all_scores, dict)
        assert len(result.all_scores) > 0


class TestDetectLanguageFromAudio:
    def test_returns_und(self):
        from yunshu_engine.lid import detect_language_from_audio

        result = detect_language_from_audio(b"\x00" * 1024)
        assert result.language == "und"
