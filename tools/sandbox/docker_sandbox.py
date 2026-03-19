"""DockerSandbox - full container isolation via Docker CLI.

Requires Docker daemon to be running and `docker` CLI on PATH.
Enabled via the [docker] optional dependency group (see pyproject.toml).

This sandbox is recommended for production use where untrusted commands
may be executed.  Each invocation runs `docker run --rm` in a fresh
container, providing:
- Filesystem isolation (no host filesystem access by default)
- Network isolation (--network none by default)
- CPU and memory limits via Docker flags
- Non-root user execution

Usage:
    sandbox = DockerSandbox(image="python:3.11-slim")
    result = await sandbox.run("python -c 'print(1+1)'")
"""

from __future__ import annotations

import asyncio
import shutil
import time

from errors.exceptions import SandboxError
from tools.sandbox.base import ResourceLimits, SandboxBase, SandboxResult


def _docker_available() -> bool:
    """Check if docker CLI is available on PATH."""
    return shutil.which("docker") is not None


class DockerSandbox(SandboxBase):
    """Sandbox implementation using Docker container isolation.

    Each `run()` call spawns a fresh container with `docker run --rm`.
    The container is always removed after execution.

    Args:
        image: Docker image to use.  Defaults to "python:3.11-slim".
        network: Docker network mode.  Defaults to "none" (no networking).
        user: Run as this user inside the container.  Defaults to "nobody".
        extra_flags: Additional docker run flags (e.g. ["-v", "/data:/data:ro"]).
        allow_network: If True, uses the default bridge network instead of "none".
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        *,
        network: str = "none",
        user: str = "nobody",
        extra_flags: list[str] | None = None,
        allow_network: bool = False,
    ) -> None:
        self._image = image
        self._network = "bridge" if allow_network else network
        self._user = user
        self._extra_flags = extra_flags or []

        if not _docker_available():
            raise SandboxError(
                "Docker CLI not found on PATH. Install Docker or use SubprocessSandbox.",
                sandbox_type=self.sandbox_type,
            )

    @property
    def sandbox_type(self) -> str:
        return "docker"

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
        """Execute cmd inside a fresh Docker container.

        Args:
            cmd: Shell command string or argv list.
            env: Environment variables passed via --env flags.
            cwd: Working directory inside the container (--workdir).
            timeout: Seconds before the container is killed.  0 = no limit.
            resource_limits: Resource constraints mapped to Docker flags.
            stdin: Optional stdin string.

        Returns:
            SandboxResult with execution details.

        Raises:
            SandboxError: If docker CLI fails to start or produces an OS error.
        """
        docker_argv = ["docker", "run", "--rm", "--network", self._network, "--user", self._user]

        # Resource limits → docker flags
        if resource_limits:
            if resource_limits.max_memory_mb is not None:
                docker_argv += ["--memory", f"{resource_limits.max_memory_mb}m"]
            if resource_limits.max_cpu_seconds is not None:
                # docker uses --cpus as a fraction, not seconds; map roughly
                docker_argv += ["--cpus", "1"]
            if resource_limits.max_processes is not None:
                docker_argv += ["--pids-limit", str(resource_limits.max_processes)]

        # Environment variables
        for k, v in (env or {}).items():
            docker_argv += ["--env", f"{k}={v}"]

        # Working directory
        if cwd:
            docker_argv += ["--workdir", cwd]

        # Extra caller-supplied flags
        docker_argv.extend(self._extra_flags)

        # Image
        docker_argv.append(self._image)

        # Command: wrap string in sh -c for shell features
        if isinstance(cmd, list):
            docker_argv.extend(cmd)
        else:
            docker_argv += ["sh", "-c", cmd]

        stdin_bytes: bytes | None = None
        if stdin is not None:
            stdin_bytes = stdin.encode("utf-8", errors="replace")

        stdin_pipe = asyncio.subprocess.PIPE if stdin_bytes is not None else None

        start_time = time.monotonic()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=stdin_pipe,
            )

            try:
                effective_timeout: float | None = float(timeout) if timeout > 0 else None
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                # Kill the container — find the container ID from stderr if possible
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
                "docker executable not found",
                sandbox_type=self.sandbox_type,
                cmd=str(cmd),
            ) from e
        except OSError as e:
            raise SandboxError(
                f"OS error running docker: {e}",
                sandbox_type=self.sandbox_type,
                cmd=str(cmd),
            ) from e

        duration_ms = (time.monotonic() - start_time) * 1000
        returncode = proc.returncode if proc.returncode is not None else -1

        return SandboxResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=returncode,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
