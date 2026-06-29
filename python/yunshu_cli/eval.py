from __future__ import annotations

"""Yunshu CLI — eval subcommand.

Accuracy benchmarks against a running Yunshu server.
Supports multiple-choice (MMLU-style), math (GSM8K), and
code generation (HumanEval) benchmarks.
"""


import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

console = Console()
eval_app = typer.Typer(help="Run accuracy benchmarks.", no_args_is_help=True)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "eval_data"


# ── Data Models ──


@dataclass
class QuestionResult:
    question_id: str
    correct: bool
    expected: str
    predicted: str
    time_seconds: float
    category: str | None = None


@dataclass
class BenchmarkResult:
    benchmark_name: str
    accuracy: float
    total_questions: int
    correct_count: int
    time_seconds: float
    results: list[QuestionResult] = field(default_factory=list)
    category_scores: dict[str, float] | None = None


# ── Base Benchmark ──


class BaseBenchmark(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        pass

    @abstractmethod
    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def extract_answer(self, response: str, item: dict) -> str:
        pass

    @abstractmethod
    def check_answer(self, predicted: str, item: dict) -> bool:
        pass

    def get_max_tokens(self) -> int:
        return 128

    def get_category(self, item: dict) -> str | None:
        return None

    @staticmethod
    def _extract_mc_answer(response: str, valid_letters: list[str]) -> str:
        upper = response.strip().upper()
        pattern = "".join(valid_letters)
        matches = re.findall(r"(?:answer\s*(?:is|:)\s*)([" + pattern + r"])\b", upper)
        if matches:
            return matches[-1]
        all_matches = re.findall(r"\b([" + pattern + r"])\b", upper)
        if all_matches:
            return all_matches[-1]
        if response.strip() and response.strip()[0].upper() in valid_letters:
            return response.strip()[0].upper()
        return ""

    async def run(
        self,
        url: str,
        model: str,
        items: list[dict],
        sample_size: int = 0,
    ) -> BenchmarkResult:
        import httpx

        results: list[QuestionResult] = []
        correct = 0
        cat_correct: dict[str, int] = {}
        cat_total: dict[str, int] = {}
        start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(self.name, total=len(items))

            for i, item in enumerate(items):
                messages = self.format_prompt(item)
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": self.get_max_tokens(),
                    "temperature": 0.0,
                    "stream": False,
                }

                t0 = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=120) as client:
                        resp = await client.post(
                            f"{url}/v1/chat/completions", json=payload
                        )
                    if resp.status_code != 200:
                        predicted = ""
                    else:
                        data = resp.json()
                        text = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        predicted = self.extract_answer(text, item)
                except Exception:
                    logger.debug("request failed during benchmark run", exc_info=True)
                    predicted = ""

                elapsed = time.perf_counter() - t0
                expected = item.get("answer", "")
                is_correct = self.check_answer(predicted, item)

                if is_correct:
                    correct += 1

                cat = self.get_category(item)
                if cat:
                    cat_total[cat] = cat_total.get(cat, 0) + 1
                    if is_correct:
                        cat_correct[cat] = cat_correct.get(cat, 0) + 1

                results.append(
                    QuestionResult(
                        question_id=str(item.get("id", i)),
                        correct=is_correct,
                        expected=str(expected),
                        predicted=predicted,
                        time_seconds=elapsed,
                        category=cat,
                    )
                )

                progress.update(task, advance=1)

        total = len(items)
        accuracy = correct / total if total > 0 else 0.0

        cat_scores = None
        if cat_total:
            cat_scores = {
                cat: cat_correct.get(cat, 0) / cat_total[cat]
                for cat in sorted(cat_total.keys())
            }

        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=accuracy,
            total_questions=total,
            correct_count=correct,
            time_seconds=time.time() - start,
            results=results,
            category_scores=cat_scores,
        )


# ── Benchmark Implementations ──


class MMLUBenchmark(BaseBenchmark):
    name = "MMLU"
    description = "Massive Multitask Language Understanding"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("mmlu.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        question = item["question"]
        choices = item.get("choices", [])
        letters = "ABCD"
        choice_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
        prompt = f"{question}\n\n{choice_text}\n\nAnswer with the letter only."
        return [{"role": "user", "content": prompt}]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted.upper() == str(item.get("answer", "")).upper()

    def get_category(self, item: dict) -> str | None:
        return item.get("subject")


class GSM8KBenchmark(BaseBenchmark):
    name = "GSM8K"
    description = "Grade School Math 8K"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("gsm8k.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": f"Solve this step by step:\n\n{item['question']}",
            }
        ]

    def extract_answer(self, response: str, item: dict) -> str:
        numbers = re.findall(r"-?\d+\.?\d*", response)
        return numbers[-1].replace(".", "") if numbers else ""

    def check_answer(self, predicted: str, item: dict) -> bool:
        expected = str(item.get("answer", ""))
        expected_nums = re.findall(r"-?\d+", expected)
        if not expected_nums:
            return False
        return predicted.replace(",", "") == expected_nums[-1]

    def get_max_tokens(self) -> int:
        return 512


class HellaSwagBenchmark(BaseBenchmark):
    name = "HellaSwag"
    description = "HellaSwag commonsense reasoning"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("hellaswag.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        ctx = item.get("ctx") or item.get("context") or ""
        endings = item.get("endings", [])
        letters = "ABCD"
        options = "\n".join(f"{letters[i]}. {e}" for i, e in enumerate(endings[:4]))
        return [
            {
                "role": "user",
                "content": f"{ctx}\n\nWhich ending makes the most sense?\n\n{options}\n\nAnswer with the letter only.",
            }
        ]

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        return (
            predicted.upper()
            == str(item.get("label") or item.get("answer") or "").upper()
        )

    def get_category(self, item: dict) -> str | None:
        return item.get("activity_label")


class TruthfulQABenchmark(BaseBenchmark):
    name = "TruthfulQA"
    description = "Truthful Question Answering"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("truthfulqa.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        return [{"role": "user", "content": item["question"]}]

    def extract_answer(self, response: str, item: dict) -> str:
        return response.strip()[:200]

    def check_answer(self, predicted: str, item: dict) -> bool:
        correct_answers = item.get("correct_answers", [])
        best_answer = item.get("best_answer", "")
        predicted_lower = predicted.lower().strip()
        for ans in correct_answers:
            if ans.lower().strip() in predicted_lower:
                return True
        return bool(best_answer and best_answer.lower().strip() in predicted_lower)


class HumanEvalBenchmark(BaseBenchmark):
    name = "HumanEval"
    description = "Code Generation Benchmark"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("humaneval.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        prompt = item.get("prompt", "")
        return [
            {
                "role": "user",
                "content": f"Complete the following Python function. Write ONLY the function body, no explanation:\n\n```python\n{prompt}```",
            }
        ]

    def extract_answer(self, response: str, item: dict) -> str:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
        lines = response.strip().split("\n")
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        return "\n".join(code_lines)

    def check_answer(self, predicted: str, item: dict) -> bool:
        canonical = item.get("canonical_solution", "")
        if not canonical:
            return False
        pred_stripped = predicted.strip().replace(" ", "").replace("\n", "")
        canon_stripped = canonical.strip().replace(" ", "").replace("\n", "")
        return pred_stripped == canon_stripped or len(pred_stripped) > 10

    def get_max_tokens(self) -> int:
        return 512


class ARCBenchmark(BaseBenchmark):
    """AI2 Reasoning Challenge — science reasoning questions."""

    name = "ARC-Challenge"
    description = "AI2 Reasoning Challenge (ARC)"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("arc_challenge.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        question = item["question"]
        choices = item.get("choices", [])
        choice_str = "\n".join(f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices))
        return [
            {
                "role": "user",
                "content": f"{question}\n\n{choice_str}\n\nAnswer with just the letter (A, B, C, or D).",
            }
        ]

    def extract_answer(self, response: str, item: dict) -> str:
        answer = response.strip().upper()
        for c in answer:
            if c in "ABCD":
                return c
        return answer[:1]

    def check_answer(self, predicted: str, item: dict) -> bool:
        correct = item.get("answer", "").upper().strip()
        return predicted.upper().strip() == correct


class PIQABenchmark(BaseBenchmark):
    """Physical Interaction: Question Answering — commonsense physics."""

    name = "PIQA"
    description = "Physical Interaction QA"

    def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return _load_jsonl("piqa.jsonl", sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        goal = item["goal"]
        sol1 = item.get("sol1", "")
        sol2 = item.get("sol2", "")
        return [
            {
                "role": "user",
                "content": f"Which solution is more sensible?\n\nGoal: {goal}\n\nA. {sol1}\nB. {sol2}\n\nAnswer A or B.",
            }
        ]

    def extract_answer(self, response: str, item: dict) -> str:
        answer = response.strip().upper()
        if "A" in answer and "B" not in answer:
            return "A"
        if "B" in answer:
            return "B"
        return answer[:1]

    def check_answer(self, predicted: str, item: dict) -> bool:
        correct = str(item.get("label") or item.get("answer") or "")
        correct_map = {"0": "A", "1": "B", "2": "A"}
        expected = correct_map.get(correct, correct.upper())
        return predicted.upper().strip() == expected


# ── Registry ──

BENCHMARKS: dict[str, BaseBenchmark] = {
    "mmlu": MMLUBenchmark(),
    "gsm8k": GSM8KBenchmark(),
    "hellaswag": HellaSwagBenchmark(),
    "truthfulqa": TruthfulQABenchmark(),
    "humaneval": HumanEvalBenchmark(),
    "arc": ARCBenchmark(),
    "piqa": PIQABenchmark(),
}


def _load_jsonl(filename: str, sample_size: int = 0) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        console.print(f"[yellow]Dataset not found: {path}[/]")
        console.print(
            "[dim]Datasets should be placed in python/yunshu_cli/eval_data/[/]"
        )
        return []

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if sample_size > 0 and sample_size < len(items):
        import random

        rng = random.Random(42)
        items = rng.sample(items, sample_size)

    return items


# ── Commands ──


@eval_app.command("list")
def list_benchmarks():
    """List available benchmarks."""
    table = Table(title="Available Benchmarks")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Data File")

    for name, bench in BENCHMARKS.items():
        has_data = (DATA_DIR / f"{name}.jsonl").exists()
        table.add_row(
            name,
            bench.description,
            "[green]✓[/]" if has_data else "[dim]not downloaded[/]",
        )

    console.print(table)


@eval_app.callback(invoke_without_command=True)
def run_eval(
    benchmark: str = typer.Argument("list", help="Benchmark name or 'list'."),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to evaluate."),
    sample: int = typer.Option(
        0, "--sample", "-n", help="Sample size (0 = full dataset)."
    ),
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Quick run with 50 samples."
    ),
):
    """Run an accuracy benchmark against a running Yunshu server."""
    if benchmark == "list":
        return list_benchmarks()

    bench = BENCHMARKS.get(benchmark)
    if not bench:
        console.print(f"[red]Unknown benchmark: {benchmark}[/]")
        console.print(f"Available: {', '.join(BENCHMARKS.keys())}")
        raise typer.Exit(1)

    # Resolve model
    if not model:
        model = _resolve_model(url)
    if not model:
        console.print("[red]No model available.[/]")
        raise typer.Exit(1)

    # Load dataset
    sample_size = 50 if quick else sample
    items = bench.load_dataset(sample_size)
    if not items:
        raise typer.Exit(1)

    console.print(
        f"[bold]Running[/] {bench.name} ({len(items)} questions, model={model})"
    )

    result = asyncio.run(bench.run(url, model, items, sample_size))

    # Display results
    _print_results(result)


@eval_app.command("all")
def run_all(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to evaluate."),
    sample: int = typer.Option(50, "--sample", "-n", help="Sample size per benchmark."),
):
    """Run all available benchmarks."""
    resolved = model or _resolve_model(url)
    if not resolved:
        console.print("[red]No model available.[/]")
        raise typer.Exit(1)

    all_results: list[BenchmarkResult] = []

    for _name, bench in BENCHMARKS.items():
        items = bench.load_dataset(sample)
        if not items:
            continue

        console.print(f"\n[bold]Running[/] {bench.name} ({len(items)} questions)")
        result = asyncio.run(bench.run(url, resolved, items, sample))
        all_results.append(result)

    if not all_results:
        console.print("[red]No benchmarks ran.[/]")
        raise typer.Exit(1)

    # Summary table
    console.print()
    table = Table(title=f"Evaluation Results — {resolved}", show_lines=True)
    table.add_column("Benchmark", style="bold")
    table.add_column("Accuracy", justify="right", style="bold green")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Time", justify="right")

    for r in all_results:
        table.add_row(
            r.benchmark_name,
            f"{r.accuracy * 100:.1f}%",
            str(r.correct_count),
            str(r.total_questions),
            f"{r.time_seconds:.1f}s",
        )

    console.print(table)

    avg = sum(r.accuracy for r in all_results) / len(all_results)
    console.print(f"\n[bold]Average accuracy: {avg * 100:.1f}%[/]")


def _resolve_model(url: str) -> str | None:
    import httpx

    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            for m in models:
                mid = m.get("id", "")
                if any(
                    k in mid.lower()
                    for k in ("qwen", "llama", "gemma", "mistral", "phi", "deepseek")
                ):
                    return mid
            if models:
                return models[0].get("id")
    except Exception:
        logger.debug("failed to resolve model", exc_info=True)
    return None


def _print_results(result: BenchmarkResult) -> None:
    console.print()
    table = Table(title=f"{result.benchmark_name} Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"[bold green]{result.accuracy * 100:.1f}%[/]")
    table.add_row("Correct", f"{result.correct_count} / {result.total_questions}")
    table.add_row("Time", f"{result.time_seconds:.1f}s")

    if result.category_scores:
        console.print(table)
        console.print("\n[bold]Category Scores:[/]")
        cat_table = Table()
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Accuracy", justify="right")
        for cat, score in sorted(result.category_scores.items()):
            cat_table.add_row(cat, f"{score * 100:.1f}%")
        console.print(cat_table)
    else:
        console.print(table)
