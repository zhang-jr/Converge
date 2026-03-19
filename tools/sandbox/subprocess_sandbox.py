"""SubprocessSandbox - subprocess isolation with timeout and resource limits.

Uses asyncio.create_subprocess_shell for execution.
On Unix, applies rlimit constraints via os.setrlimit in a preexec_fn.
On Windows, only timeout isolation is available (resource module absent).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Callable

from errors.exceptions import SandboxError
from tools.sandbox.base import ResourceLimits, SandboxBase, SandboxResult

# Unix-only resource limiting
_HAS_RESOURCE = False
if sys.platform != "win32":
    try:
        import resource as _resource_module
        _HAS_RESOURCE = True
    except ImportError:
        pass


def _build_preexec(limits: ResourceLimits) -> Callable[[], None] | None:
    """Build a preexec_fn that applies rlimits before exec.

    Only meaningful on Unix.  Returns None on Windows or if no limits set.

    Args:
        limits: Resource limits to apply.

    Returns:
        A zero-argument callable, or None.
    """
    if not _HAS_RESOURCE:
        return None

    import resource as res

    def _apply() -> None:
        if limits.max_cpu_seconds is not None:
            cpu_sec = int(limits.max_cpu_seconds)
            res.setrlimit(res.RLIMIT_CPU, (cpu_sec, cpu_sec))
        if limits.max_memory_mb is not None:
            mem_bytes = limits.max_memory_mb * 1024 * 1024
            res.setrlimit(res.RLIMIT_AS, (mem_bytes, mem_bytes))
        if limits.max_file_size_mb is not None:
            fsize_bytes = limits.max_file_size_mb * 1024 * 1024
            res.setrlimit(res.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        if limits.max_processes is not None:
            nproc = limits.max_processes
            res.setrlimit(res.RLIMIT_NPROC, (nproc, nproc))

    return _apply


class SubprocessSandbox(SandboxBase):
    """Sandbox implementation using a child subprocess.

    Provides:
    - Timeout isolation (cross-platform)
    - Resource limits via Unix rlimit (Unix only)
    - Optional environment variable control

    This is the recommended sandbox for testing/pre-production use.
    Zero external dependencies beyond Python stdlib.

    Args:
        base_env: If True (default), inherit the current process environment
            and merge with any caller-supplied env dict.
            If False, only the caller-supplied env is used (minimal env).
    """

    def __init__(self, *, base_env: bool = True) -> None:
        self._base_env = base_env

    @property
    def sandbox_type(self) -> str:
        return "subprocess"

    async def run(
        self,
        cmd: str | list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
        resource_limits: ResourceLimits | None = None,
        stdin: str | None = None,
    ) -> SandboxResult:
        """Execute cmd in an isolated subprocess.

        Args:
            cmd: Shell command string or argv list.
            env: Extra environment variables (merged with base if base_env=True).
            cwd: Working directory for the subprocess.
            timeout: Seconds before SIGKILL is sent.  0 = no limit.
            resource_limits: Optional rlimit constraints (Unix only).
            stdin: Optional stdin string (encoded UTF-8).

        Returns:
            SandboxResult with execution details.

        Raises:
            SandboxError: If the shell cannot be found or an OS error occurs.
        """
        # Build environment
        if self._base_env:
            merged_env: dict[str, str] | None = {**os.environ, **(env or {})}
        else:
            merged_env = env or {}

        # Build rlimit preexec_fn (Unix only)
        preexec_fn: Callable[[], None] | None = None
        if resource_limits is not None and _HAS_RESOURCE:
            preexec_fn = _build_preexec(resource_limits)

        # Encode stdin
        stdin_bytes: bytes | None = None
        if stdin is not None:
            stdin_bytes = stdin.encode("utf-8", errors="replace")

        stdin_pipe = asyncio.subprocess.PIPE if stdin_bytes is not None else None

        # Choose shell vs exec depending on cmd type
        start_time = time.monotonic()
        timed_out = False
        resource_exceeded = False

        try:
            if isinstance(cmd, list):
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=stdin_pipe,
                    cwd=cwd,
                    env=merged_env,
                    **({"preexec_fn": preexec_fn} if preexec_fn and sys.platform != "win32" else {}),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=stdin_pipe,
                    cwd=cwd,
                    env=merged_env,
                    **({"preexec_fn": preexec_fn} if preexec_fn and sys.platform != "win32" else {}),
                )

            try:
                effective_timeout: float | None = float(timeout) if timeout > 0 else None
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    stdout_bytes, stderr_bytes = await proc.communicate()
                except Exception:
                    stdout_bytes, stderr_bytes = b"", b""

        except FileNotFoundError as e:
            raise SandboxError(
                f"Shell/executable not found: {e}",
                sandbox_type=self.sandbox_type,
                cmd=str(cmd),
            ) from e
        except OSError as e:
            # Non-zero exit due to rlimit typically shows as signal
            # SIGXCPU (24) or SIGKILL from RLIMIT_AS
            raise SandboxError(
                f"OS error in subprocess sandbox: {e}",
                sandbox_type=self.sandbox_type,
                cmd=str(cmd),
            ) from e

        duration_ms = (time.monotonic() - start_time) * 1000
        returncode = proc.returncode if proc.returncode is not None else -1

        # Detect resource exceeded: SIGXCPU=24 on Linux, or specific exit codes
        if sys.platform != "win32" and returncode in (-24, -9):
            # SIGXCPU (-24) or SIGKILL (-9, from RLIMIT_AS)
            resource_exceeded = True

        return SandboxResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=returncode,
            duration_ms=duration_ms,
            timed_out=timed_out,
            resource_exceeded=resource_exceeded,
        )
