"""Tests for real BFCL-format test cases.

Validates:
- All 4 categories have 10+ test cases
- Each test case has required fields (prompt, tools, expected_calls)
- Tool definitions are valid JSON-schema function definitions
- Expected calls match the prompt intent
- Test cases are deterministic and verifiable
- Roundtrip: expected_calls survive JSON parse/serialize
- Cross-category correctness: simple < parallel, multiple < parallel_multiple
"""

import json

import pytest

from yunshu_engine.bfcl_eval import (
    VALID_CATEGORIES,
    BFCLEvalConfig,
    BFCLEvaluator,
    _calls_match,
    _generate_stub_cases,
)

# ── Test case counts ──


class TestTestCaseCounts:
    """Verify each category has 10+ test cases."""

    def test_simple_has_10_plus(self):
        cases = _generate_stub_cases("simple")
        assert len(cases) >= 10, f"simple has {len(cases)} cases, need >= 10"

    def test_parallel_has_10_plus(self):
        cases = _generate_stub_cases("parallel")
        assert len(cases) >= 10, f"parallel has {len(cases)} cases, need >= 10"

    def test_multiple_has_10_plus(self):
        cases = _generate_stub_cases("multiple")
        assert len(cases) >= 10, f"multiple has {len(cases)} cases, need >= 10"

    def test_parallel_multiple_has_10_plus(self):
        cases = _generate_stub_cases("parallel_multiple")
        assert len(cases) >= 10, f"parallel_multiple has {len(cases)} cases, need >= 10"

    def test_invalid_category_returns_empty(self):
        cases = _generate_stub_cases("nonexistent")
        assert cases == []


# ── Test case structure validation ──


class TestTestCaseStructure:
    """Each test case must have prompt, tools, expected_calls."""

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_required_fields(self, category):
        cases = _generate_stub_cases(category)
        for i, case in enumerate(cases):
            assert "prompt" in case, f"Case {i} in '{category}' missing 'prompt'"
            assert "tools" in case, f"Case {i} in '{category}' missing 'tools'"
            assert "expected_calls" in case, f"Case {i} in '{category}' missing 'expected_calls'"

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_prompt_is_nonempty_string(self, category):
        cases = _generate_stub_cases(category)
        for i, case in enumerate(cases):
            assert isinstance(case["prompt"], str), f"Case {i}: prompt must be string"
            assert len(case["prompt"]) > 0, f"Case {i}: prompt must not be empty"

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_tools_is_list_of_dicts(self, category):
        cases = _generate_stub_cases(category)
        for i, case in enumerate(cases):
            assert isinstance(case["tools"], list), f"Case {i}: tools must be list"
            assert len(case["tools"]) > 0, f"Case {i}: tools must not be empty"
            for tool in case["tools"]:
                assert isinstance(tool, dict), f"Case {i}: each tool must be dict"
                assert tool.get("type") == "function", f"Case {i}: tool type must be 'function'"
                assert "function" in tool, f"Case {i}: tool must have 'function' key"

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_expected_calls_is_list_of_dicts(self, category):
        cases = _generate_stub_cases(category)
        for i, case in enumerate(cases):
            assert isinstance(case["expected_calls"], list), f"Case {i}: expected_calls must be list"
            for call in case["expected_calls"]:
                assert isinstance(call, dict), f"Case {i}: each call must be dict"
                assert "name" in call, f"Case {i}: each call must have 'name'"
                assert "arguments" in call, f"Case {i}: each call must have 'arguments'"
                assert isinstance(call["arguments"], dict), f"Case {i}: arguments must be dict"


# ── Tool definition validation ──


class TestToolDefinitions:
    """Verify tool definitions are valid OpenAI-format function definitions."""

    ALL_TOOLS = set()
    ALL_TOOL_NAMES = set()

    @classmethod
    def _collect_tools(cls):
        if cls.ALL_TOOLS:
            return
        for cat in VALID_CATEGORIES:
            for case in _generate_stub_cases(cat):
                for tool in case["tools"]:
                    func = tool["function"]
                    cls.ALL_TOOLS.add(func["name"])
                    cls.ALL_TOOL_NAMES.add(func["name"])

    def test_tool_names_are_realistic(self):
        """Tool names should come from the expected set of realistic tools."""
        self._collect_tools()
        expected_tools = {
            "get_weather", "calculate", "search_web",
            "send_email", "file_operations", "database_query",
        }
        assert expected_tools == self.ALL_TOOL_NAMES, (
            f"Unexpected tool names: {self.ALL_TOOL_NAMES - expected_tools}"
        )

    def test_tools_have_descriptions(self):
        """Every tool should have a non-empty description."""
        self._collect_tools()
        seen = set()
        for cat in VALID_CATEGORIES:
            for case in _generate_stub_cases(cat):
                for tool in case["tools"]:
                    func = tool["function"]
                    name = func["name"]
                    if name in seen:
                        continue
                    seen.add(name)
                    assert "description" in func, f"Tool '{name}' missing description"
                    assert len(func["description"]) > 0, f"Tool '{name}' has empty description"

    def test_tools_have_parameters(self):
        """Every tool should have a parameters object."""
        self._collect_tools()
        seen = set()
        for cat in VALID_CATEGORIES:
            for case in _generate_stub_cases(cat):
                for tool in case["tools"]:
                    func = tool["function"]
                    name = func["name"]
                    if name in seen:
                        continue
                    seen.add(name)
                    assert "parameters" in func, f"Tool '{name}' missing parameters"
                    params = func["parameters"]
                    assert params.get("type") == "object", f"Tool '{name}' params type must be 'object'"
                    assert "properties" in params, f"Tool '{name}' missing properties"

    def test_tools_have_required_fields(self):
        """Tools should specify required fields where appropriate."""
        self._collect_tools()
        # get_weather must require 'city'
        for cat in VALID_CATEGORIES:
            for case in _generate_stub_cases(cat):
                for tool in case["tools"]:
                    func = tool["function"]
                    if func["name"] == "get_weather":
                        assert "city" in func["parameters"].get("required", []), (
                            "get_weather must require 'city'"
                        )

    def test_tools_are_json_serializable(self):
        """All tool definitions must be JSON-serializable."""
        for cat in VALID_CATEGORIES:
            for case in _generate_stub_cases(cat):
                for tool in case["tools"]:
                    serialized = json.dumps(tool)
                    deserialized = json.loads(serialized)
                    assert deserialized == tool


# ── Determinism & verifiability ──


class TestDeterminism:
    """Test cases must be deterministic across multiple calls."""

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_generate_is_deterministic(self, category):
        """Calling _generate_stub_cases twice should produce identical results."""
        cases1 = _generate_stub_cases(category)
        cases2 = _generate_stub_cases(category)
        # Compare via JSON for deep equality
        assert json.dumps(cases1) == json.dumps(cases2)

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_expected_calls_survive_json_roundtrip(self, category):
        """expected_calls should survive JSON serialize/deserialize."""
        cases = _generate_stub_cases(category)
        for case in cases:
            original = case["expected_calls"]
            serialized = json.dumps(original)
            recovered = json.loads(serialized)
            assert _calls_match(original, recovered), (
                f"Roundtrip failed for prompt: {case['prompt'][:50]}"
            )

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_expected_calls_parseable_as_function_calls(self, category):
        """expected_calls should be parseable by parse_function_calls."""
        cases = _generate_stub_cases(category)
        for case in cases:
            expected_json = json.dumps(case["expected_calls"])
            parsed = json.loads(expected_json)
            assert isinstance(parsed, list)
            for call in parsed:
                assert "name" in call
                assert "arguments" in call


# ── Category-specific constraints ──


class TestCategoryConstraints:
    """Verify category-specific constraints on test cases."""

    def test_simple_single_tool_single_call(self):
        """Simple category: each case should have 1 tool and 1 expected call."""
        cases = _generate_stub_cases("simple")
        for i, case in enumerate(cases):
            assert len(case["tools"]) == 1, f"Simple case {i}: should have exactly 1 tool"
            assert len(case["expected_calls"]) == 1, f"Simple case {i}: should have exactly 1 call"

    def test_parallel_single_tool_multiple_calls(self):
        """Parallel category: same tool, multiple calls."""
        cases = _generate_stub_cases("parallel")
        for i, case in enumerate(cases):
            assert len(case["expected_calls"]) >= 2, (
                f"Parallel case {i}: should have >= 2 expected calls"
            )

    def test_multiple_multiple_tools_single_call(self):
        """Multiple category: multiple tools available, single call made."""
        cases = _generate_stub_cases("multiple")
        for i, case in enumerate(cases):
            assert len(case["tools"]) >= 2, f"Multiple case {i}: should have >= 2 tools"
            assert len(case["expected_calls"]) == 1, f"Multiple case {i}: should have exactly 1 call"

    def test_parallel_multiple_multiple_tools_multiple_calls(self):
        """Parallel-multiple category: multiple tools, multiple parallel calls."""
        cases = _generate_stub_cases("parallel_multiple")
        for i, case in enumerate(cases):
            assert len(case["tools"]) >= 2, f"PM case {i}: should have >= 2 tools"
            assert len(case["expected_calls"]) >= 2, f"PM case {i}: should have >= 2 calls"

    def test_expected_call_names_exist_in_tools(self):
        """Every expected call should reference a tool that is available."""
        for cat in VALID_CATEGORIES:
            cases = _generate_stub_cases(cat)
            for i, case in enumerate(cases):
                available = {t["function"]["name"] for t in case["tools"]}
                for call in case["expected_calls"]:
                    assert call["name"] in available, (
                        f"[{cat}] Case {i}: expected call '{call['name']}' "
                        f"not in available tools {available}"
                    )

    def test_expected_call_arguments_are_valid_for_tool(self):
        """Expected call arguments should reference known tool parameters."""
        for cat in VALID_CATEGORIES:
            cases = _generate_stub_cases(cat)
            for i, case in enumerate(cases):
                tools_by_name = {t["function"]["name"]: t["function"] for t in case["tools"]}
                for call in case["expected_calls"]:
                    tool = tools_by_name.get(call["name"])
                    if tool is None:
                        continue
                    props = tool.get("parameters", {}).get("properties", {})
                    for arg_name in call["arguments"]:
                        assert arg_name in props, (
                            f"[{cat}] Case {i}: arg '{arg_name}' in call to "
                            f"'{call['name']}' not in tool parameters"
                        )


# ── Evaluator integration ──


class TestRealCasesWithEvaluator:
    """Verify the real test cases integrate properly with BFCLEvaluator."""

    def test_load_test_cases_returns_real_cases(self):
        config = BFCLEvalConfig(model_name="test", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config)
        for cat in VALID_CATEGORIES:
            cases = evaluator.load_test_cases(cat)
            assert len(cases) >= 10, f"'{cat}' should have >= 10 cases, got {len(cases)}"

    def test_evaluate_category_without_engine(self):
        """Without an engine, all test cases should fail gracefully."""
        config = BFCLEvalConfig(model_name="test", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        result = evaluator.evaluate_category("simple")
        assert result.total >= 10
        assert result.correct == 0  # No engine, all fail
        assert len(result.errors) > 0

    def test_format_results_with_real_cases(self):
        """format_results should produce readable output for real cases."""
        config = BFCLEvalConfig(model_name="test", output_dir="/nonexistent")
        evaluator = BFCLEvaluator(config, engine=None)
        results = evaluator.run_all()
        output = evaluator.format_results(results)
        assert "OVERALL" in output
        # Each category should appear
        for cat in VALID_CATEGORIES:
            assert cat in output
