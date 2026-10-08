"""Generate completions from the actual command tree, without editing shell files."""

import typer

from ._output import emit, fail


def completion(shell: str = typer.Argument(help="Shell: zsh, bash or fish.")) -> None:
    """Print a shell completion script; redirect it to your shell's completion directory."""
    if shell not in {"zsh", "bash", "fish"}:
        fail("Choose zsh, bash or fish: yunshu completion zsh", code=2)
    from typer._completion_shared import get_completion_script

    # Click's completion protocol derives its private variable from the executable
    # name. It is not an engine setting.
    prog_name = "yunshu"
    script = get_completion_script(
        prog_name=prog_name, complete_var=f"_{prog_name.upper()}_COMPLETE", shell=shell
    )
    emit({"shell": shell, "script": script}, human=lambda: typer.echo(script))
