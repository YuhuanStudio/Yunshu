"""Yunshu CLI — chat subcommand.

Interactive terminal chat with streaming, Markdown rendering,
thinking mode, and multi-turn conversation history.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()
chat_app = typer.Typer(help="Interactive chat.", no_args_is_help=True)

HISTORY: list[dict[str, str]] = []


@chat_app.callback(invoke_without_command=True)
def chat(
    model: str | None = typer.Option(None, "--model", "-m", help="Model name."),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt."),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Sampling temperature."
    ),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Max output tokens."),
    thinking: bool = typer.Option(
        False, "--thinking", help="Enable thinking/reasoning mode."
    ),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming."),
):
    """Interactive chat with a Yunshu model."""
    import httpx

    # Resolve model
    resolved_model = model
    if not resolved_model:
        try:
            resp = httpx.get(f"{url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                # Prefer LLM models
                for m in models:
                    mid = m.get("id", "")
                    if any(
                        k in mid.lower()
                        for k in (
                            "qwen",
                            "llama",
                            "gemma",
                            "mistral",
                            "phi",
                            "deepseek",
                        )
                    ):
                        resolved_model = mid
                        break
                if not resolved_model and models:
                    resolved_model = models[0].get("id")
        except httpx.ConnectError:
            console.print(
                "[red]Cannot connect to server.[/] Start with: [bold]yunshu serve[/]"
            )
            raise typer.Exit(1) from None

    if not resolved_model:
        console.print(
            "[red]No model available.[/] Specify with --model or start a server with a loaded model."
        )
        raise typer.Exit(1)

    if system:
        HISTORY.append({"role": "system", "content": system})

    _print_welcome(resolved_model, url, thinking)
    _repl(url, resolved_model, temperature, max_tokens, thinking, no_stream)


def _print_welcome(model: str, url: str, thinking: bool) -> None:
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"Model: [bold cyan]{model}[/]\n"
                f"Server: [dim]{url}[/]\n"
                f"Thinking: {'[green]on[/]' if thinking else '[dim]off[/]'}\n\n"
                "[dim]Type your message and press Enter. Ctrl+C or /quit to exit.[/]\n"
                "[dim]Commands: /clear, /thinking, /model, /help[/]"
            ),
            title="[bold]Yunshu Chat[/]",
            border_style="bright_blue",
        )
    )
    console.print()


def _repl(
    url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking: bool,
    no_stream: bool,
) -> None:
    import httpx

    while True:
        try:
            user_input = console.input("[bold green]You[/] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/]")
                break
            elif cmd == "/clear":
                HISTORY.clear()
                console.print("[dim]Conversation cleared.[/]")
                continue
            elif cmd == "/thinking":
                thinking = not thinking
                console.print(
                    f"Thinking: {'[green]on[/]' if thinking else '[dim]off[/]'}"
                )
                continue
            elif cmd == "/model":
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    model = parts[1]
                    console.print(f"Model → [cyan]{model}[/]")
                else:
                    console.print(f"Current model: [cyan]{model}[/]")
                continue
            elif cmd == "/help":
                console.print(
                    "[dim]/clear — clear history\n"
                    "/thinking — toggle thinking mode\n"
                    "/model <name> — switch model\n"
                    "/quit — exit[/]"
                )
                continue
            else:
                console.print(f"[yellow]Unknown command: {cmd}[/]")
                continue

        # Add user message
        HISTORY.append({"role": "user", "content": user_input})

        # Build request
        payload = {
            "model": model,
            "messages": HISTORY,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": not no_stream,
        }
        if thinking:
            payload["enable_thinking"] = True

        # Send and display response
        try:
            if no_stream:
                _send_non_stream(url, payload)
            else:
                _send_stream(url, payload)
        except httpx.ConnectError:
            console.print("[red]Connection lost.[/] Is the server running?")
            HISTORY.pop()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/]")

        console.print()


def _send_non_stream(url: str, payload: dict) -> None:
    import httpx

    with console.status("[bold cyan]Generating..."):
        resp = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=120)

    if resp.status_code != 200:
        console.print(f"[red]Error {resp.status_code}:[/] {resp.text[:200]}")
        HISTORY.pop()
        return

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    content = msg.get("content", "")

    if reasoning:
        console.print(
            Panel(
                reasoning,
                title="[bold]Thinking[/]",
                border_style="dim",
            )
        )

    if content:
        console.print(Markdown(content))
        HISTORY.append({"role": "assistant", "content": content})

    # Show usage
    usage = data.get("usage", {})
    if usage:
        parts = []
        if usage.get("prompt_tokens"):
            parts.append(f"prompt: {usage['prompt_tokens']}")
        if usage.get("completion_tokens"):
            parts.append(f"completion: {usage['completion_tokens']}")
        if parts:
            console.print(f"[dim]{', '.join(parts)}[/]")


def _send_stream(url: str, payload: dict) -> None:
    import json

    import httpx

    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json=payload, timeout=120
    ) as resp:
        if resp.status_code != 200:
            error_body = "".join(resp.iter_text())
            console.print(f"[red]Error {resp.status_code}:[/] {error_body[:200]}")
            HISTORY.pop()
            return

        content_buf = ""
        reasoning_buf = ""
        in_thinking = False

        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if not delta:
                continue

            # Reasoning
            rc = delta.get("reasoning_content")
            if rc:
                if not in_thinking:
                    in_thinking = True
                    console.print("[bold dim]Thinking...[/]", end="")
                reasoning_buf += rc
                # Show dots for thinking progress
                console.print("[dim]·[/]", end="")

            # Content
            c = delta.get("content")
            if c:
                if in_thinking:
                    # End thinking, show reasoning panel
                    console.print()
                    if reasoning_buf:
                        console.print(
                            Panel(
                                reasoning_buf,
                                title="[bold]Thinking[/]",
                                border_style="dim",
                            )
                        )
                    in_thinking = False
                content_buf += c

        # Final render
        if content_buf:
            console.print()
            console.print(Markdown(content_buf))
            HISTORY.append({"role": "assistant", "content": content_buf})
        elif reasoning_buf and not content_buf:
            console.print()
            console.print(
                Panel(
                    reasoning_buf,
                    title="[bold]Thinking[/]",
                    border_style="dim",
                )
            )
