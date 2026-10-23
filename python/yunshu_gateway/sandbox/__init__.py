"""Local code execution under Anthropic's sandbox-runtime (srt).

``srt`` (optional, Node >= 22.12) gives file and network isolation; this package adds what it does
not: wall / CPU / memory / output limits (``supervisor``), per-container workspaces with expiry
(``containers``) and the glue that turns a tool call into a sandboxed process (``executor``).
Nothing here ever runs a command outside srt.
"""

from .containers import Container, ContainerError, ContainerManager, get_manager
from .executor import CodeExecutor, ExecResult, SandboxUnavailableError, get_executor
from .srt import SrtInfo, build_settings, find_srt
from .supervisor import RunResult, run_supervised

__all__ = [
    "CodeExecutor",
    "Container",
    "ContainerError",
    "ContainerManager",
    "ExecResult",
    "RunResult",
    "SandboxUnavailableError",
    "SrtInfo",
    "build_settings",
    "find_srt",
    "get_executor",
    "get_manager",
    "run_supervised",
]
