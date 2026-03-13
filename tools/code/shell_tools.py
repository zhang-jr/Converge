"""Shell execution tools for Code Agent.

BashTool and KillShellTool are high-risk operations that trigger
HumanInterventionHandler before execution.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from core.config import BASH_TOOL_DEFAULT_TIMEOUT
from tools.base import ToolBase, ToolDryRunResult, ToolResult

# Process registry for KillShellTool: pid -> asyncio.subprocess.Process
_active_processes: dict[int, asyncio.subprocess.Process] = {}


class BashTool(ToolBase):
    """Execute a shell command in a subprocess.

    HIGH RISK: Can run arbitrary commands. Always triggers human-in-the-loop.
    """

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Captures stdout, stderr, and exit code. "
            "Supports timeout. HIGH RISK: triggers human approval."
        )

    @property
    def side_effects(self) -> list[str]:
        return ["executes_shell_command", "may_modify_filesystem", "may_make_network_requests"]

    @property
    def reversible(self) -> bool:
        return False

    @property
    def risk_level(self) -> Literal["low", "medium", "high"]:
        return "high"

    @property
    def idempotent(self) -> bool:
        return False

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {BASH_TOOL_DEFAULT_TIMEOUT})",
                    "default": BASH_TOOL_DEFAULT_TIMEOUT,
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: current directory)",
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["command"],
        }

    async def dry_run(self, params: dict[str, Any]) -> ToolDryRunResult:
        """Preview the command that would be executed.

        Args:
            params: Command parameters.

        Returns:
            ToolDryRunResult with command preview.
        """
        command = params.get("command", "")
        cwd = params.get("cwd", ".")
        timeout = params.get("timeout", BASH_TOOL_DEFAULT_TIMEOUT)
        return ToolDryRunResult(
            would_succeed=True,
            preview=f"Would execute: {command!r}\n  cwd={cwd}, timeout={timeout}s",
            affected_resources=["filesystem", "processes"],
            warnings=["This command may have irreversible side effects"],
        )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute a shell command.

        Args:
            params: Must contain 'command'; optionally 'timeout', 'cwd', 'env'.

        Returns:
            ToolResult with stdout, stderr, and returncode.
        """
        command = params.get("command", "")
        timeout = params.get("timeout", BASH_TOOL_DEFAULT_TIMEOUT)
        cwd = params.get("cwd") or None
        extra_env = params.get("env") or {}

        if not command:
            return ToolResult(success=False, error="'command' parameter is required")

        env = {**os.environ, **extra_env} if extra_env else None

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            _active_processes[proc.pid] = proc
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout}s: {command!r}",
                    metadata={"command": command, "timeout": timeout},
                )
            finally:
                _active_processes.pop(proc.pid, None)

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            returncode = proc.returncode

            success = returncode == 0
            return ToolResult(
                success=success,
                output={"stdout": stdout, "stderr": stderr, "returncode": returncode},
                error=None if success else f"Command exited with code {returncode}: {stderr[:500]}",
                metadata={"command": command, "returncode": returncode},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"Shell not found for command: {command!r}")
        except OSError as e:
            return ToolResult(success=False, error=f"OS error executing command: {e}")


class KillShellTool(ToolBase):
    """Kill a running subprocess by PID.

    HIGH RISK: Terminates processes.
    """

    @property
    def name(self) -> str:
        return "kill_shell"

    @property
    def description(self) -> str:
        return "Kill a running shell process by PID. Only processes started by BashTool can be killed."

    @property
    def side_effects(self) -> list[str]:
        return ["terminates_process"]

    @property
    def reversible(self) -> bool:
        return False

    @property
    def risk_level(self) -> Literal["low", "medium", "high"]:
        return "high"

    @property
    def idempotent(self) -> bool:
        return True  # killing an already-dead process is a no-op

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pid": {"type": "integer", "description": "Process ID to kill"},
            },
            "required": ["pid"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Kill a running process by PID.

        Args:
            params: Must contain 'pid' (integer).

        Returns:
            ToolResult indicating success or failure.
        """
        pid = params.get("pid")
        if pid is None:
            return ToolResult(success=False, error="'pid' parameter is required")
        if not isinstance(pid, int):
            return ToolResult(success=False, error="'pid' must be an integer")

        proc = _active_processes.get(pid)
        if proc is None:
            return ToolResult(
                success=False,
                error=(
                    f"Process {pid} not found in active processes "
                    "(only BashTool-managed processes can be killed)"
                ),
            )

        try:
            proc.kill()
            _active_processes.pop(pid, None)
            return ToolResult(
                success=True,
                output=f"Process {pid} killed",
                metadata={"pid": pid},
            )
        except ProcessLookupError:
            _active_processes.pop(pid, None)
            return ToolResult(
                success=True,
                output=f"Process {pid} was already terminated",
                metadata={"pid": pid},
            )
        except OSError as e:
            return ToolResult(success=False, error=f"Failed to kill process {pid}: {e}")
