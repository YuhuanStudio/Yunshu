"""Tests for BFCL evaluation integration.

Covers:
- BFCLEvalConfig creation and validation
- BFCLEvalResult creation
- load_test_cases stub generation
- Function call parsing (JSON, XML, Python-callable formats)
- _calls_match comparison logic
- format_results table output
- BFCLEvaluator with no engine (stub path)
"""

import pytest

from yunshu_engine.bfcl_eval import (
    VALID_CATEGORIES,
    BFCLEvalConfig,
    BFCLEvalResult,
    BFCLEvaluator,
    _calls_match,
    parse_function_call_json,
    parse_function_call_python,
    parse_function_call_xml,
    parse_function_calls,
)

# ── BFCLEvalConfig ───────────────────────────────────────────────────────────


class TestBFCLEvalConfig:
    def test_default_config(self):
        config = BFCLEvalConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.test_categories == list(VALID_CATEGORIES)
        assert config.max_samples == 0
        assert config.output_dir == "bfcl_results"

    def test_custom_categories(self):
        config = BFCLEvalConfig(
            model_name="m",
            test_categories=["simple", "parallel"],
        )
        assert config.test_categories == ["simple", "parallel"]

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid BFCL category"):
            BFCLEvalConfig(model_name="m", test_categories=["invalid"])

    def test_all_categories_valid(self):
        config = BFCLEvalConfig(
            model_name="m",
            test_categories=list(VALID_CATEGORIES),
        )
        assert len(config.test_categories) == 4

    def test_max_samples_zero_means_all(self):
        config = BFCLEvalConfig(model_name="m", max_samples=0)
        assert config.max_samples == 0

    def test_max_samples_positive(self):
        config = BFCLEvalConfig(model_name="m", max_samples=10)
        assert config.max_samples == 10


# ── BFCLEvalResult ───────────────────────────────────────────────────────────


class TestBFCLEvalResult:
    def test_default_result(self):
        result = BFCLEvalResult(category="simple")
        assert result.category == "simple"
        assert result.total == 0
        assert result.correct == 0
        assert result.accuracy == 0.0
        assert result.avg_latency_ms == 0.0
        assert result.errors == []

    def test_result_with_values(self):
        result = BFCLEvalResult(
            category="parallel",
            total=100,
            correct=85,
            accuracy=0.85,
            avg_latency_ms=42.5,
            errors=["Case 3: timeout"],
        )
        assert result.category == "parallel"
        assert result.total == 100
        assert result.correct == 85
        assert result.accuracy == 0.85
        assert len(result.errors) == 1


# ── load_test_cases ──────────────────────────────────────────────────────────


class TestLoadTestCases:
    def test_stub_simple(self):
        config = BFCLEvalConfig(model_name="m", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        cases = evaluator.load_test_cases("simple")
        assert len(cases) > 0
        for case in cases:
            assert "prompt" in case
            assert "tools" in case
            assert "expected_calls" in case

    def test_stub_parallel(self):
        config = BFCLEvalConfig(model_name="m", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        cases = evaluator.load_test_cases("parallel")
        assert len(cases) > 0
        # Parallel cases should have multiple expected calls
        assert len(cases[0]["expected_calls"]) >= 2

    def test_stub_multiple(self):
        config = BFCLEvalConfig(model_name="m", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        cases = evaluator.load_test_cases("multiple")
        assert len(cases) > 0
        # Multiple: should have > 1 tool to choose from
        assert len(cases[0]["tools"]) > 1

    def test_stub_parallel_multiple(self):
        config = BFCLEvalConfig(model_name="m", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        cases = evaluator.load_test_cases("parallel_multiple")
        assert len(cases) > 0

    def test_invalid_category_raises(self):
        config = BFCLEvalConfig(model_name="m")
        evaluator = BFCLEvaluator(config, engine=None)
        with pytest.raises(ValueError, match="Invalid category"):
            evaluator.load_test_cases("nonexistent")

    def test_all_stub_categories_nonempty(self):
        config = BFCLEvalConfig(model_name="m", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        for cat in VALID_CATEGORIES:
            cases = evaluator.load_test_cases(cat)
            assert len(cases) > 0, f"Stub for '{cat}' should be non-empty"


# ── JSON Parsing ─────────────────────────────────────────────────────────────


class TestParseFunctionCallJSON:
    def test_single_object(self):
        text = '{"name": "get_weather", "arguments": {"city": "SF"}}'
        calls = parse_function_call_json(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "SF"

    def test_array_of_objects(self):
        text = '[{"name": "fn1", "arguments": {"a": 1}}, {"name": "fn2", "arguments": {"b": 2}}]'
        calls = parse_function_call_json(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "fn1"
        assert calls[1]["name"] == "fn2"

    def test_parameters_alias(self):
        text = '{"name": "calc", "parameters": {"x": 1}}'
        calls = parse_function_call_json(text)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"x": 1}

    def test_nested_in_tool_calls(self):
        text = '{"tool_calls": [{"name": "fn", "arguments": {}}]}'
        calls = parse_function_call_json(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "fn"

    def test_embedded_json(self):
        text = (
            'Some text before {"name": "search", "arguments": {"q": "test"}} and after'
        )
        calls = parse_function_call_json(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"

    def test_empty_string(self):
        assert parse_function_call_json("") == []

    def test_invalid_json(self):
        assert parse_function_call_json("not json at all") == []

    def test_json_without_name(self):
        text = '{"key": "value"}'
        calls = parse_function_call_json(text)
        assert len(calls) == 0


# ── XML Parsing ──────────────────────────────────────────────────────────────


class TestParseFunctionCallXML:
    def test_function_call_tag(self):
        text = '<function_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</function_call>'
        calls = parse_function_call_xml(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "NYC"

    def test_tool_call_self_closing_tag(self):
        text = '<tool_call/>{"name": "calc", "arguments": {"expr": "1+1"}}</tool_call/>'
        calls = parse_function_call_xml(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "calc"

    def test_multiple_xml_tags(self):
        text = (
            '<function_call>{"name": "fn1", "arguments": {}}</function_call>'
            '<function_call>{"name": "fn2", "arguments": {}}</function_call>'
        )
        calls = parse_function_call_xml(text)
        assert len(calls) == 2

    def test_python_format_inside_tag(self):
        text = '<function_call>get_weather(city="SF")</function_call>'
        calls = parse_function_call_xml(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "SF"

    def test_no_tags(self):
        text = "just plain text"
        calls = parse_function_call_xml(text)
        assert len(calls) == 0


# ── Python-Callable Parsing ──────────────────────────────────────────────────


class TestParseFunctionCallPython:
    def test_single_call(self):
        text = 'get_weather(city="SF")'
        calls = parse_function_call_python(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "SF"

    def test_multiple_args(self):
        text = 'search(query="MLX", num_results=5)'
        calls = parse_function_call_python(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["query"] == "MLX"
        assert calls[0]["arguments"]["num_results"] == 5

    def test_boolean_arg(self):
        text = "set_flag(verbose=True)"
        calls = parse_function_call_python(text)
        assert calls[0]["arguments"]["verbose"] is True

    def test_no_args(self):
        text = "ping()"
        calls = parse_function_call_python(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "ping"
        assert calls[0]["arguments"] == {}

    def test_multiple_calls(self):
        text = "fn1(a=1) fn2(b=2)"
        calls = parse_function_call_python(text)
        assert len(calls) == 2

    def test_empty_string(self):
        assert parse_function_call_python("") == []

    def test_no_parentheses(self):
        assert parse_function_call_python("just text") == []


# ── Unified parse_function_calls ─────────────────────────────────────────────


class TestParseFunctionCalls:
    def test_auto_detect_json(self):
        text = '{"name": "fn", "arguments": {"x": 1}}'
        calls = parse_function_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "fn"

    def test_auto_detect_xml(self):
        text = '<function_call>{"name": "fn", "arguments": {}}</function_call>'
        calls = parse_function_calls(text)
        assert len(calls) == 1

    def test_auto_detect_python(self):
        text = 'get_data(key="value")'
        calls = parse_function_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_data"

    def test_no_format_detected(self):
        assert parse_function_calls("plain text") == []


# ── _calls_match ─────────────────────────────────────────────────────────────


class TestCallsMatch:
    def test_exact_single_match(self):
        actual = [{"name": "get_weather", "arguments": {"city": "SF"}}]
        expected = [{"name": "get_weather", "arguments": {"city": "SF"}}]
        assert _calls_match(actual, expected) is True

    def test_argument_mismatch(self):
        actual = [{"name": "get_weather", "arguments": {"city": "NYC"}}]
        expected = [{"name": "get_weather", "arguments": {"city": "SF"}}]
        assert _calls_match(actual, expected) is False

    def test_name_mismatch(self):
        actual = [{"name": "calc", "arguments": {}}]
        expected = [{"name": "search", "arguments": {}}]
        assert _calls_match(actual, expected) is False

    def test_count_mismatch(self):
        actual = [{"name": "fn", "arguments": {}}]
        expected = [
            {"name": "fn", "arguments": {}},
            {"name": "fn2", "arguments": {}},
        ]
        assert _calls_match(actual, expected) is False

    def test_parallel_order_independent(self):
        actual = [
            {"name": "fn2", "arguments": {"b": 2}},
            {"name": "fn1", "arguments": {"a": 1}},
        ]
        expected = [
            {"name": "fn1", "arguments": {"a": 1}},
            {"name": "fn2", "arguments": {"b": 2}},
        ]
        assert _calls_match(actual, expected) is True

    def test_empty_expected_no_actual(self):
        assert _calls_match([], []) is True

    def test_no_expected_but_actual_exists(self):
        assert _calls_match([{"name": "fn", "arguments": {}}], []) is False

    def test_parameters_key_normalized(self):
        actual = [{"name": "fn", "arguments": {"x": 1}}]
        expected = [{"name": "fn", "parameters": {"x": 1}}]
        assert _calls_match(actual, expected) is True


# ── format_results ───────────────────────────────────────────────────────────


class TestFormatResults:
    def test_empty_results(self):
        config = BFCLEvalConfig(model_name="test-model")
        evaluator = BFCLEvaluator(config)
        output = evaluator.format_results([])
        assert "test-model" in output
        assert "OVERALL" in output

    def test_single_category(self):
        config = BFCLEvalConfig(model_name="m")
        evaluator = BFCLEvaluator(config)
        results = [
            BFCLEvalResult(
                category="simple",
                total=50,
                correct=45,
                accuracy=0.9,
                avg_latency_ms=12.3,
            ),
        ]
        output = evaluator.format_results(results)
        assert "simple" in output
        assert "50" in output
        assert "45" in output
        assert "90.0%" in output
        assert "OVERALL" in output

    def test_multiple_categories_with_errors(self):
        config = BFCLEvalConfig(model_name="m")
        evaluator = BFCLEvaluator(config)
        results = [
            BFCLEvalResult(
                category="simple",
                total=10,
                correct=8,
                accuracy=0.8,
                avg_latency_ms=10.0,
            ),
            BFCLEvalResult(
                category="parallel",
                total=5,
                correct=3,
                accuracy=0.6,
                avg_latency_ms=20.0,
                errors=["Case 0: mismatch for 'test prompt'"],
            ),
        ]
        output = evaluator.format_results(results)
        assert "simple" in output
        assert "parallel" in output
        assert "Errors:" in output
        assert "Case 0:" in output
        # Check totals
        assert "15" in output  # total
        assert "11" in output  # correct


# ── BFCLEvaluator without engine ─────────────────────────────────────────────


class TestEvaluatorNoEngine:
    def test_evaluate_single_returns_false_without_engine(self):
        config = BFCLEvalConfig(model_name="m")
        evaluator = BFCLEvaluator(config, engine=None)
        case = {
            "prompt": "test",
            "tools": [],
            "expected_calls": [{"name": "fn", "arguments": {}}],
        }
        assert evaluator.evaluate_single(case) is False

    def test_evaluate_category_returns_result(self):
        config = BFCLEvalConfig(model_name="m", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        result = evaluator.evaluate_category("simple")
        assert isinstance(result, BFCLEvalResult)
        assert result.category == "simple"
        assert result.total > 0
        # Without engine, all should be errors
        assert result.correct == 0

    def test_run_all_returns_per_category(self):
        config = BFCLEvalConfig(
            model_name="m",
            test_categories=["simple", "parallel"],
            output_dir="/nonexistent",
        )
        evaluator = BFCLEvaluator(config, engine=None)
        results = evaluator.run_all()
        assert len(results) == 2
        assert results[0].category == "simple"
        assert results[1].category == "parallel"

    def test_max_samples_limits_cases(self):
        config = BFCLEvalConfig(
            model_name="m",
            test_categories=["simple"],
            max_samples=1,
            output_dir="/nonexistent",
        )
        evaluator = BFCLEvaluator(config, engine=None)
        result = evaluator.evaluate_category("simple")
        assert result.total <= 1
