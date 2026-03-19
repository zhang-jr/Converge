"""Abstract base for execution sandboxes.

Sandboxes provide isolated, resource-limited environments for running
shell commands. The three-tier strategy (ADR-014):
  None            → direct host execution (development default)
  SubprocessSandbox → subprocess + timeout + ulimit (test/pre-prod)
  DockerSandbox   → container isolation (production)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ResourceLimits(BaseModel):
    """Resource constraints applied to sandbox execution.

    Attributes:
        max_cpu_seconds: Maximum CPU time in seconds (Unix only).
        max_memory_mb: Maximum virtual memory in megabytes (Unix only).
        max_file_size_mb: Maximum size for any single file written (Unix only).
        max_processes: Maximum number of child processes (Unix only).
    """

    max_cpu_seconds: float | None = None
    max_memory_mb: int | None = None
    max_file_size_mb: int | None = None
    max_processes: int | None = None


class SandboxResult(BaseModel):
    """Result of a sandboxed command execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        returncode: Process exit code.
        duration_ms: Wall-clock time in milliseconds.
        timed_out: True if the command was killed due to timeout.
        resource_exceeded: True if a resource limit was hit.
    """

    stdout: str
    stderr: str
    returncode: int
    duration_ms: float = Field(default=0.0)
    timed_out: bool = False
    resource_exceeded: bool = False


class SandboxBase(ABC):
    """Abstract base class for execution sandboxes.

    All implementations must be async-compatible.
    High-risk commands (risk_level='high') are routed through a sandbox
    by BashTool when one is configured.
    """

    @property
    @abstractmethod
    def sandbox_type(self) -> str:
        """Return a string identifier for this sandbox type."""
        ...

    @abstractmethod
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
        """Run a command inside the sandbox.

        Args:
            cmd: Shell command string or argv list.
            env: Environment variables to pass (merges with minimal base env).
            cwd: Working directory inside the sandbox.
            timeout: Wall-clock timeout in seconds. 0 means no limit.
            resource_limits: Optional resource caps (CPU/memory/file size).
            stdin: Optional string to pipe to stdin.

        Returns:
            SandboxResult with stdout, stderr, returncode, and metadata.

        Raises:
            SandboxError: If the sandbox itself fails to start or communicate.
        """
        ...
