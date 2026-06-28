from __future__ import annotations

"""Yunshu BFCL Evaluation Integration — Berkeley Function Calling Leaderboard.

.. deprecated:: This module is not used in the production pipeline. Kept for reference only.


Evaluates tool/function calling accuracy of the Yunshu engine against
BFCL-format test cases. Supports the four standard BFCL categories:
  - simple:       single function, single call
  - parallel:     multiple functions, parallel calls
  - multiple:     single function from a list, single call
  - parallel_multiple: multiple functions, multiple parallel calls

Function call parsing supports three response formats:
  1. JSON:  {"name": "...", "arguments": {...}}
  2. XML:   <function_call>{"name": "...", "arguments": {...}}</function_call>
  3. Python-callable: get_weather(city="SF")

Architecture:
  BFCLEvalConfig   — evaluation parameters
  BFCLEvalResult   — per-category metrics
  BFCLEvaluator    — orchestrates test loading, execution, and scoring
"""


import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── BFCL Categories ──────────────────────────────────────────────────────────

VALID_CATEGORIES = ("simple", "parallel", "multiple", "parallel_multiple")


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class BFCLEvalConfig:
    """Configuration for a BFCL evaluation run.

    Attributes:
        model_name: Name/identifier of the model under test.
        test_categories: BFCL categories to evaluate.
        max_samples: Max test cases per category (0 = all).
        output_dir: Directory for writing results JSON.
    """

    model_name: str
    test_categories: list[str] = field(
        default_factory=lambda: list(VALID_CATEGORIES),
    )
    max_samples: int = 0
    output_dir: str = "bfcl_results"

    def __post_init__(self) -> None:
        for cat in self.test_categories:
            if cat not in VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid BFCL category '{cat}'. "
                    f"Valid: {VALID_CATEGORIES}"
                )


@dataclass
class BFCLEvalResult:
    """Results for a single BFCL evaluation category.

    Attributes:
        category: The BFCL category name.
        total: Total number of test cases evaluated.
        correct: Number of correctly answered cases.
        accuracy: Fraction correct (correct / total, 0.0 if no cases).
        avg_latency_ms: Average per-case latency in milliseconds.
        errors: List of error descriptions for failed cases.
    """

    category: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── Function Call Parsing ────────────────────────────────────────────────────


def parse_function_call_json(text: str) -> list[dict[str, Any]]:
    """Parse function calls from JSON format.

    Handles both single objects and arrays of objects.
    Format: {"name": "...", "arguments": {...}}

    Returns:
        List of dicts with 'name' and 'arguments' keys.
    """
    text = text.strip()
    if not text:
        return []

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "name" in data:
                return [_normalize_call(data)]
            # Maybe nested inside a "calls" or "tool_calls" key
            for key in ("calls", "tool_calls", "function_calls"):
                if key in data and isinstance(data[key], list):
                    return [_normalize_call(c) for c in data[key]]
            return []
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    results.append(_normalize_call(item))
            return results
    except json.JSONDecodeError:
        pass

    # Try to find JSON objects embedded in text using bracket counting
    results: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Find matching closing brace
            depth = 0
            start = i
            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and "name" in data:
                        results.append(_normalize_call(data))
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "name" in item:
                                results.append(_normalize_call(item))
                except json.JSONDecodeError:
                    pass
        i += 1

    return results


def parse_function_call_xml(text: str) -> list[dict[str, Any]]:
    """Parse function calls from XML-tagged format.

    Handles: <function_call>...</function_call>,
    <tool_call/>...</tool_call/>, <tool_call ...>...</tool_call...>.

    Returns:
        List of dicts with 'name' and 'arguments' keys.
    """
    results: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()

    # Unified pattern: handles <function_call>, <tool_call/>, <tool_call > etc.
    tag_pattern = re.compile(
        r"<(?:function_call|tool_call)[^>]*>(.*?)</(?:function_call|tool_call)[^>]*>",
        re.DOTALL,
    )

    for match in tag_pattern.finditer(text):
        span = match.span()
        if span in seen_spans:
            continue
        seen_spans.add(span)

        inner = match.group(1).strip()
        # Try JSON parse of inner content
        parsed = parse_function_call_json(inner)
        if parsed:
            results.extend(parsed)
            continue
        # Try Python-callable format inside tags
        parsed_py = parse_function_call_python(inner)
        if parsed_py:
            results.extend(parsed_py)

    return results


def parse_function_call_python(text: str) -> list[dict[str, Any]]:
    """Parse function calls from Python-callable format.

    Format: function_name(arg1="value1", arg2="value2")
    Handles string, numeric, and boolean arguments.

    Returns:
        List of dicts with 'name' and 'arguments' keys.
    """
    text = text.strip()
    if not text:
        return []

    results: list[dict[str, Any]] = []

    # Match: name(args) potentially multiple separated by newlines/semicolons
    call_pattern = re.compile(
        r'(\w+)\s*\(([^)]*)\)',
    )
    for match in call_pattern.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()
        arguments: dict[str, Any] = {}

        if args_str:
            # Parse key=value pairs
            arg_pattern = re.compile(
                r'(\w+)\s*=\s*('
                r'"[^"]*"'        # double-quoted string
                r"|'[^']*'"       # single-quoted string
                r"|True|False"    # booleans
                r"|None"          # None
                r"|[\d.eE+-]+"    # numbers
                r')',
            )
            for arg_match in arg_pattern.finditer(args_str):
                key = arg_match.group(1)
                value_str = arg_match.group(2)
                arguments[key] = _parse_python_value(value_str)

        results.append({"name": name, "arguments": arguments})

    return results


def parse_function_calls(text: str) -> list[dict[str, Any]]:
    """Parse function calls from model output using all supported formats.

    Tries formats in order: XML, JSON, Python-callable.
    Returns results from the first format that succeeds.

    Returns:
        List of dicts with 'name' and 'arguments' keys.
    """
    # Try XML first (most explicit)
    xml_calls = parse_function_call_xml(text)
    if xml_calls:
        return xml_calls

    # Try JSON
    json_calls = parse_function_call_json(text)
    if json_calls:
        return json_calls

    # Try Python-callable
    py_calls = parse_function_call_python(text)
    if py_calls:
        return py_calls

    return []


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalize_call(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a parsed function call dict to {name, arguments}."""
    name = data.get("name", "")
    arguments = data.get("arguments") or data.get("parameters") or {}
    if not isinstance(arguments, dict):
        # If arguments is a string, try to parse it
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        else:
            arguments = {}
    return {"name": name, "arguments": arguments}


def _parse_python_value(value_str: str) -> Any:
    """Parse a Python literal value from a function call argument."""
    # String values
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]
    # Boolean
    if value_str == "True":
        return True
    if value_str == "False":
        return False
    # None
    if value_str == "None":
        return None
    # Number
    try:
        if "." in value_str or "e" in value_str.lower():
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str


def _normalize_value(val: Any) -> Any:
    """Normalize a value for comparison (recursively)."""
    if isinstance(val, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [_normalize_value(v) for v in val]
    if isinstance(val, str):
        return val.strip()
    return val


def _calls_match(actual: list[dict], expected: list[dict]) -> bool:
    """Compare parsed function calls against expected calls.

    Checks that every expected call has a matching actual call (by name and
    arguments). For parallel calls the order does not matter.
    """
    if not expected:
        return bool(actual) is False  # no calls expected

    if len(actual) != len(expected):
        return False

    # Build list of normalized expected calls
    expected_normalized = [_normalize_call(e) for e in expected]
    actual_normalized = [_normalize_call(a) for a in actual]

    # Match by name + arguments (order-independent for parallel)
    remaining = list(actual_normalized)
    for exp in expected_normalized:
        exp_name = exp["name"]
        exp_args = _normalize_value(exp["arguments"])

        found = False
        for i, act in enumerate(remaining):
            if act["name"] == exp_name and _normalize_value(act["arguments"]) == exp_args:
                remaining.pop(i)
                found = True
                break
        if not found:
            return False

    return True


# ── Real Test Case Generator ──────────────────────────────────────────────────


def _generate_stub_cases(category: str) -> list[dict[str, Any]]:
    """Generate representative BFCL-format test cases for a category.

    These are used when no external BFCL data files are available.
    Each test case has: prompt, tools (function definitions), expected_calls.

    Realistic function definitions: get_weather, search_web, send_email,
    calculate, file_operations, database_query.
    """

    # ── Tool definitions ──

    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }

    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
        },
    }

    search_tool = {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
    }

    email_tool = {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body content"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    }

    file_tool = {
        "type": "function",
        "function": {
            "name": "file_operations",
            "description": "Perform file system operations like read, write, delete, or list files",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "delete", "list"],
                        "description": "The file operation to perform",
                    },
                    "path": {"type": "string", "description": "File or directory path"},
                    "content": {"type": "string", "description": "Content for write operations"},
                },
                "required": ["operation", "path"],
            },
        },
    }

    db_tool = {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Execute a SQL query on the connected database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {
                        "type": "string",
                        "description": "Database name (default: 'main')",
                    },
                },
                "required": ["query"],
            },
        },
    }

    # ── Category-specific test cases ──

    stubs: dict[str, list[dict[str, Any]]] = {
        # ── simple: single function, single call ──
        "simple": [
            {
                "prompt": "What's the weather in San Francisco?",
                "tools": [weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "San Francisco"}},
                ],
            },
            {
                "prompt": "What's the weather like in Tokyo in Celsius?",
                "tools": [weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}},
                ],
            },
            {
                "prompt": "Calculate 2 + 2",
                "tools": [calculator_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "2 + 2"}},
                ],
            },
            {
                "prompt": "Compute 15 * 23 + 7",
                "tools": [calculator_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "15 * 23 + 7"}},
                ],
            },
            {
                "prompt": "Search the web for 'MLX framework Apple'",
                "tools": [search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "MLX framework Apple"}},
                ],
            },
            {
                "prompt": "Search for 'Python async tutorial' and return 5 results",
                "tools": [search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "Python async tutorial", "num_results": 5}},
                ],
            },
            {
                "prompt": "Send an email to alice@example.com with subject 'Meeting Tomorrow' and body 'Hi Alice, our meeting is at 3pm.'",
                "tools": [email_tool],
                "expected_calls": [
                    {"name": "send_email", "arguments": {
                        "to": "alice@example.com",
                        "subject": "Meeting Tomorrow",
                        "body": "Hi Alice, our meeting is at 3pm.",
                    }},
                ],
            },
            {
                "prompt": "Read the file at /tmp/data.json",
                "tools": [file_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "read", "path": "/tmp/data.json"}},
                ],
            },
            {
                "prompt": "List all files in the /var/log directory",
                "tools": [file_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "list", "path": "/var/log"}},
                ],
            },
            {
                "prompt": "Run a SQL query: SELECT * FROM users WHERE active = true",
                "tools": [db_tool],
                "expected_calls": [
                    {"name": "database_query", "arguments": {"query": "SELECT * FROM users WHERE active = true"}},
                ],
            },
            {
                "prompt": "Query the analytics database for SELECT COUNT(*) FROM events",
                "tools": [db_tool],
                "expected_calls": [
                    {"name": "database_query", "arguments": {"query": "SELECT COUNT(*) FROM events", "database": "analytics"}},
                ],
            },
            {
                "prompt": "Write 'Hello World' to the file /tmp/hello.txt",
                "tools": [file_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "write", "path": "/tmp/hello.txt", "content": "Hello World"}},
                ],
            },
        ],

        # ── parallel: multiple calls to the SAME function ──
        "parallel": [
            {
                "prompt": "What's the weather in SF and NYC?",
                "tools": [weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "San Francisco"}},
                    {"name": "get_weather", "arguments": {"city": "New York"}},
                ],
            },
            {
                "prompt": "Get the weather for London, Paris, and Berlin",
                "tools": [weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "London"}},
                    {"name": "get_weather", "arguments": {"city": "Paris"}},
                    {"name": "get_weather", "arguments": {"city": "Berlin"}},
                ],
            },
            {
                "prompt": "Calculate 2+2 and 3*3",
                "tools": [calculator_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "2+2"}},
                    {"name": "calculate", "arguments": {"expression": "3*3"}},
                ],
            },
            {
                "prompt": "Search for 'machine learning basics' and 'deep learning tutorial'",
                "tools": [search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "machine learning basics"}},
                    {"name": "search_web", "arguments": {"query": "deep learning tutorial"}},
                ],
            },
            {
                "prompt": "Send an email to bob@test.com about 'Report Ready' saying 'The report is done.' and also send to carol@test.com about 'Update' saying 'Project is on track.'",
                "tools": [email_tool],
                "expected_calls": [
                    {"name": "send_email", "arguments": {"to": "bob@test.com", "subject": "Report Ready", "body": "The report is done."}},
                    {"name": "send_email", "arguments": {"to": "carol@test.com", "subject": "Update", "body": "Project is on track."}},
                ],
            },
            {
                "prompt": "Delete /tmp/old_cache.dat and /tmp/old_logs.dat",
                "tools": [file_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "delete", "path": "/tmp/old_cache.dat"}},
                    {"name": "file_operations", "arguments": {"operation": "delete", "path": "/tmp/old_logs.dat"}},
                ],
            },
            {
                "prompt": "Query SELECT * FROM orders and SELECT COUNT(*) FROM products",
                "tools": [db_tool],
                "expected_calls": [
                    {"name": "database_query", "arguments": {"query": "SELECT * FROM orders"}},
                    {"name": "database_query", "arguments": {"query": "SELECT COUNT(*) FROM products"}},
                ],
            },
            {
                "prompt": "Compute the square root expressions: sqrt(144) and sqrt(256)",
                "tools": [calculator_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "sqrt(144)"}},
                    {"name": "calculate", "arguments": {"expression": "sqrt(256)"}},
                ],
            },
            {
                "prompt": "Get the weather for both Seattle and Portland in Fahrenheit",
                "tools": [weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "Seattle", "unit": "fahrenheit"}},
                    {"name": "get_weather", "arguments": {"city": "Portland", "unit": "fahrenheit"}},
                ],
            },
            {
                "prompt": "Search for 'Rust programming' and 'Go programming' with 3 results each",
                "tools": [search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "Rust programming", "num_results": 3}},
                    {"name": "search_web", "arguments": {"query": "Go programming", "num_results": 3}},
                ],
            },
            {
                "prompt": "Read both /etc/hosts and /etc/resolv.conf",
                "tools": [file_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "read", "path": "/etc/hosts"}},
                    {"name": "file_operations", "arguments": {"operation": "read", "path": "/etc/resolv.conf"}},
                ],
            },
        ],

        # ── multiple: single function from a list, single call ──
        "multiple": [
            {
                "prompt": "Search for 'MLX framework'",
                "tools": [weather_tool, calculator_tool, search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "MLX framework"}},
                ],
            },
            {
                "prompt": "What's the weather like in Miami?",
                "tools": [calculator_tool, email_tool, weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "Miami"}},
                ],
            },
            {
                "prompt": "Calculate 42 / 6",
                "tools": [search_tool, file_tool, calculator_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "42 / 6"}},
                ],
            },
            {
                "prompt": "Send an email to dev@team.io about 'Sprint Review' saying 'Please review the PRs.'",
                "tools": [weather_tool, calculator_tool, email_tool],
                "expected_calls": [
                    {"name": "send_email", "arguments": {"to": "dev@team.io", "subject": "Sprint Review", "body": "Please review the PRs."}},
                ],
            },
            {
                "prompt": "Read the file /home/user/.bashrc",
                "tools": [db_tool, email_tool, file_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "read", "path": "/home/user/.bashrc"}},
                ],
            },
            {
                "prompt": "Run SELECT version();",
                "tools": [search_tool, calculator_tool, db_tool],
                "expected_calls": [
                    {"name": "database_query", "arguments": {"query": "SELECT version();"}},
                ],
            },
            {
                "prompt": "Search for 'Apple Silicon MLX benchmark'",
                "tools": [weather_tool, email_tool, db_tool, search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "Apple Silicon MLX benchmark"}},
                ],
            },
            {
                "prompt": "Get the weather for Chicago",
                "tools": [search_tool, email_tool, file_tool, weather_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "Chicago"}},
                ],
            },
            {
                "prompt": "Calculate log(100)",
                "tools": [weather_tool, email_tool, db_tool, calculator_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "log(100)"}},
                ],
            },
            {
                "prompt": "Delete the temporary file /tmp/session_8273.cache",
                "tools": [weather_tool, search_tool, calculator_tool, file_tool, email_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "delete", "path": "/tmp/session_8273.cache"}},
                ],
            },
            {
                "prompt": "Send an email to support@company.com about 'Bug Report' saying 'Found a null pointer exception in module X.'",
                "tools": [weather_tool, db_tool, calculator_tool, file_tool, email_tool],
                "expected_calls": [
                    {"name": "send_email", "arguments": {"to": "support@company.com", "subject": "Bug Report", "body": "Found a null pointer exception in module X."}},
                ],
            },
        ],

        # ── parallel_multiple: multiple functions, multiple parallel calls ──
        "parallel_multiple": [
            {
                "prompt": "Search for 'MLX' and get weather in SF",
                "tools": [weather_tool, calculator_tool, search_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "MLX"}},
                    {"name": "get_weather", "arguments": {"city": "San Francisco"}},
                ],
            },
            {
                "prompt": "Get the weather in Boston and calculate 17 * 31",
                "tools": [weather_tool, calculator_tool, search_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "Boston"}},
                    {"name": "calculate", "arguments": {"expression": "17 * 31"}},
                ],
            },
            {
                "prompt": "Search for 'Apple M4 benchmarks', get weather in Cupertino, and send an email to john@apple.com about 'Benchmark Results' saying 'M4 shows 40% improvement.'",
                "tools": [search_tool, weather_tool, email_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "Apple M4 benchmarks"}},
                    {"name": "get_weather", "arguments": {"city": "Cupertino"}},
                    {"name": "send_email", "arguments": {"to": "john@apple.com", "subject": "Benchmark Results", "body": "M4 shows 40% improvement."}},
                ],
            },
            {
                "prompt": "Read the file /var/log/app.log and query the database for SELECT * FROM errors LIMIT 10",
                "tools": [file_tool, db_tool, search_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "read", "path": "/var/log/app.log"}},
                    {"name": "database_query", "arguments": {"query": "SELECT * FROM errors LIMIT 10"}},
                ],
            },
            {
                "prompt": "Write 'backup complete' to /tmp/status.txt and delete /tmp/old_backup.tar",
                "tools": [file_tool, weather_tool, search_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "write", "path": "/tmp/status.txt", "content": "backup complete"}},
                    {"name": "file_operations", "arguments": {"operation": "delete", "path": "/tmp/old_backup.tar"}},
                ],
            },
            {
                "prompt": "Calculate 100 / 3 and query SELECT AVG(price) FROM products",
                "tools": [calculator_tool, db_tool, email_tool],
                "expected_calls": [
                    {"name": "calculate", "arguments": {"expression": "100 / 3"}},
                    {"name": "database_query", "arguments": {"query": "SELECT AVG(price) FROM products"}},
                ],
            },
            {
                "prompt": "Search for 'Python 3.13 features' and send an email to team@dev.io about 'New Release' saying 'Python 3.13 is out with JIT compiler.'",
                "tools": [search_tool, email_tool, weather_tool],
                "expected_calls": [
                    {"name": "search_web", "arguments": {"query": "Python 3.13 features"}},
                    {"name": "send_email", "arguments": {"to": "team@dev.io", "subject": "New Release", "body": "Python 3.13 is out with JIT compiler."}},
                ],
            },
            {
                "prompt": "Get weather for Denver in Fahrenheit and calculate 2^10",
                "tools": [weather_tool, calculator_tool, file_tool],
                "expected_calls": [
                    {"name": "get_weather", "arguments": {"city": "Denver", "unit": "fahrenheit"}},
                    {"name": "calculate", "arguments": {"expression": "2^10"}},
                ],
            },
            {
                "prompt": "List files in /home/user/documents and search for 'quarterly report template'",
                "tools": [file_tool, search_tool, calculator_tool],
                "expected_calls": [
                    {"name": "file_operations", "arguments": {"operation": "list", "path": "/home/user/documents"}},
                    {"name": "search_web", "arguments": {"query": "quarterly report template"}},
                ],
            },
            {
                "prompt": "Send a quick email to hr@corp.com saying 'I will be late today' with subject 'Late Arrival' and get the weather in Austin",
                "tools": [email_tool, weather_tool, search_tool, db_tool],
                "expected_calls": [
                    {"name": "send_email", "arguments": {"to": "hr@corp.com", "subject": "Late Arrival", "body": "I will be late today"}},
                    {"name": "get_weather", "arguments": {"city": "Austin"}},
                ],
            },
            {
                "prompt": "Query the inventory database for SELECT * FROM stock WHERE quantity < 5 and search for 'restock supplier contacts'",
                "tools": [db_tool, search_tool, email_tool],
                "expected_calls": [
                    {"name": "database_query", "arguments": {"query": "SELECT * FROM stock WHERE quantity < 5", "database": "inventory"}},
                    {"name": "search_web", "arguments": {"query": "restock supplier contacts"}},
                ],
            },
        ],
    }

    return stubs.get(category, [])


# ── BFCL Evaluator ───────────────────────────────────────────────────────────


class BFCLEvaluator:
    """Orchestrates BFCL function calling evaluation.

    Usage:
        config = BFCLEvalConfig(model_name="qwen-0.5b")
        evaluator = BFCLEvaluator(config, engine=my_engine)
        results = evaluator.run_all()
        print(evaluator.format_results(results))
    """

    def __init__(self, config: BFCLEvalConfig, engine: Any = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            engine: Optional engine reference for live evaluation.
                    Must support generate(prompt, tools, ...) if provided.
        """
        self._config = config
        self._engine = engine

    @property
    def config(self) -> BFCLEvalConfig:
        return self._config

    def load_test_cases(self, category: str) -> list[dict[str, Any]]:
        """Load BFCL-format test cases for a category.

        Looks for JSON data files in output_dir first. If none found,
        generates representative stub test cases.

        Args:
            category: One of the BFCL categories.

        Returns:
            List of test case dicts with 'prompt', 'tools', 'expected_calls'.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'")

        # Try loading from file
        data_path = os.path.join(
            self._config.output_dir,
            f"bfcl_{category}.json",
        )
        if os.path.isfile(data_path):
            try:
                with open(data_path) as f:
                    cases = json.load(f)
                if isinstance(cases, list) and cases:
                    logger.info(
                        "Loaded %d test cases from %s", len(cases), data_path,
                    )
                    return cases
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", data_path, exc)

        # Fall back to stubs
        cases = _generate_stub_cases(category)
        logger.info(
            "Generated %d stub test cases for '%s'", len(cases), category,
        )
        return cases

    def evaluate_single(self, test_case: dict[str, Any]) -> bool:
        """Evaluate a single function calling test case.

        Formats the prompt with tool definitions, sends to the engine,
        parses function calls from the response, and compares against
        expected calls.

        Args:
            test_case: Dict with 'prompt', 'tools', 'expected_calls'.

        Returns:
            True if the parsed calls match expected calls.
        """
        prompt = test_case.get("prompt", "")
        tools = test_case.get("tools", [])
        expected = test_case.get("expected_calls", [])

        if not self._engine:
            # No engine: skip actual evaluation, return False
            logger.debug("No engine configured; cannot evaluate test case")
            return False

        # Build the full prompt with tool definitions
        tools_json = json.dumps(tools, indent=2)
        full_prompt = (
            f"You have access to the following tools:\n{tools_json}\n\n"
            f"User: {prompt}\n\n"
            f"Respond with a function call in JSON format."
        )

        try:
            # Call the engine — support both sync and async interfaces
            response = self._engine.generate(full_prompt)
            if hasattr(response, "__await__"):
                # Async engine — cannot await in sync context
                logger.warning("Async engine provided; use evaluate_category_async")
                return False

            response_text = _extract_text(response)
        except Exception as exc:
            logger.debug("Engine generation failed: %s", exc)
            return False

        # Parse function calls from response
        actual_calls = parse_function_calls(response_text)

        # Compare against expected
        return _calls_match(actual_calls, expected)

    def evaluate_category(self, category: str) -> BFCLEvalResult:
        """Run all test cases in a category.

        Args:
            category: BFCL category name.

        Returns:
            BFCLEvalResult with metrics for this category.
        """
        cases = self.load_test_cases(category)

        if self._config.max_samples > 0:
            cases = cases[: self._config.max_samples]

        total = len(cases)
        if total == 0:
            return BFCLEvalResult(category=category)

        correct = 0
        errors: list[str] = []
        total_latency = 0.0

        for i, case in enumerate(cases):
            start = time.perf_counter()
            try:
                is_correct = self.evaluate_single(case)
            except Exception as exc:
                is_correct = False
                errors.append(f"Case {i}: {exc}")

            latency_ms = (time.perf_counter() - start) * 1000.0
            total_latency += latency_ms

            if is_correct:
                correct += 1
            else:
                prompt_preview = case.get("prompt", "")[:80]
                if not any(f"Case {i}:" in e for e in errors):
                    errors.append(f"Case {i}: mismatch for '{prompt_preview}'")

        return BFCLEvalResult(
            category=category,
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0.0,
            avg_latency_ms=total_latency / total if total > 0 else 0.0,
            errors=errors,
        )

    def run_all(self) -> list[BFCLEvalResult]:
        """Run evaluation for all configured categories.

        Returns:
            List of BFCLEvalResult, one per category.
        """
        results: list[BFCLEvalResult] = []

        for category in self._config.test_categories:
            logger.info("Evaluating BFCL category: %s", category)
            result = self.evaluate_category(category)
            results.append(result)

        # Optionally save results
        if self._config.output_dir:
            self._save_results(results)

        return results

    def format_results(self, results: list[BFCLEvalResult]) -> str:
        """Format evaluation results as a human-readable table.

        Args:
            results: List of BFCLEvalResult to format.

        Returns:
            Formatted string table.
        """
        # Header
        lines: list[str] = []
        lines.append(f"BFCL Evaluation Results — {self._config.model_name}")
        lines.append("=" * 72)
        header = (
            f"{'Category':<22} {'Total':>6} {'Correct':>8} "
            f"{'Accuracy':>10} {'Avg ms':>8}"
        )
        lines.append(header)
        lines.append("-" * 72)

        total_all = 0
        correct_all = 0

        for r in results:
            lines.append(
                f"{r.category:<22} {r.total:>6} {r.correct:>8} "
                f"{r.accuracy:>9.1%} {r.avg_latency_ms:>7.1f}"
            )
            total_all += r.total
            correct_all += r.correct

        lines.append("-" * 72)
        overall_acc = correct_all / total_all if total_all > 0 else 0.0
        lines.append(
            f"{'OVERALL':<22} {total_all:>6} {correct_all:>8} "
            f"{overall_acc:>9.1%}"
        )
        lines.append("=" * 72)

        # Append error details
        all_errors: list[str] = []
        for r in results:
            for err in r.errors:
                all_errors.append(f"  [{r.category}] {err}")

        if all_errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(all_errors)

        return "\n".join(lines)

    def _save_results(self, results: list[BFCLEvalResult]) -> None:
        """Save results to a JSON file in output_dir."""
        output_dir = self._config.output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir,
                f"bfcl_results_{self._config.model_name}.json",
            )
            data = [
                {
                    "category": r.category,
                    "total": r.total,
                    "correct": r.correct,
                    "accuracy": r.accuracy,
                    "avg_latency_ms": r.avg_latency_ms,
                    "errors": r.errors,
                }
                for r in results
            ]
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved BFCL results to %s", out_path)
        except OSError as exc:
            logger.warning("Failed to save results: %s", exc)


def _extract_text(response: Any) -> str:
    """Extract text content from various engine response types."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("text") or response.get("content") or ""
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)
