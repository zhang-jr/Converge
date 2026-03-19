"""Unit tests for sandbox implementations.

Tests SubprocessSandbox isolation, timeout, resource limits, and BashTool
sandbox integration.  DockerSandbox tests are marked with a custom marker
and skipped when docker CLI is not available.
"""

from __future__ import annotations

import shutil
import sys

import pytest

from tools.sandbox.base import ResourceLimits, SandboxResult
from tools.sandbox.subprocess_sandbox import SubprocessSandbox


# ---------------------------------------------------------------------------
# SubprocessSandbox — basic execution
# ---------------------------------------------------------------------------

class TestSubprocessSandboxBasic:
    """Basic command execution through SubprocessSandbox."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        return SubprocessSandbox()

    @pytest.mark.asyncio
    async def test_echo_command(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("echo hello")
        assert isinstance(result, SandboxResult)
        assert result.stdout.strip() == "hello"
        assert result.returncode == 0
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("exit 42", timeout=5.0)
        assert result.returncode == 42
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_stderr_captured(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("echo error_msg >&2", timeout=5.0)
        assert "error_msg" in result.stderr

    @pytest.mark.asyncio
    async def test_duration_ms_positive(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("echo ok")
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_env_var_passed(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("echo $MY_VAR", env={"MY_VAR": "sandbox_value"})
        assert "sandbox_value" in result.stdout

    @pytest.mark.asyncio
    async def test_cwd_respected(self, sandbox: SubprocessSandbox, tmp_path) -> None:
        result = await sandbox.run("pwd", cwd=str(tmp_path))
        assert str(tmp_path).replace("\\", "/") in result.stdout.replace("\\", "/")

    @pytest.mark.asyncio
    async def test_argv_list_cmd(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run(["echo", "list_cmd"])
        assert "list_cmd" in result.stdout

    @pytest.mark.asyncio
    async def test_stdin_pipe(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("cat", stdin="hello_stdin")
        assert "hello_stdin" in result.stdout

    @pytest.mark.asyncio
    async def test_minimal_env(self) -> None:
        """With base_env=False, only caller-supplied env is set."""
        sandbox = SubprocessSandbox(base_env=False)
        result = await sandbox.run("echo $NONEXISTENT_VAR_XYZ", env={})
        # Should not raise; empty variable expands to empty string
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# SubprocessSandbox — timeout
# ---------------------------------------------------------------------------

class TestSubprocessSandboxTimeout:
    """Timeout enforcement tests."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        return SubprocessSandbox()

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, sandbox: SubprocessSandbox) -> None:
        # Use Python sleep for cross-platform compatibility
        result = await sandbox.run(
            sys.executable + " -c \"import time; time.sleep(60)\"",
            timeout=0.2,
        )
        assert result.timed_out
        assert result.returncode != 0

    @pytest.mark.asyncio
    async def test_no_timeout_completes(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("echo fast", timeout=30.0)
        assert not result.timed_out
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_zero_timeout_means_no_limit(self, sandbox: SubprocessSandbox) -> None:
        result = await sandbox.run("echo quick", timeout=0)
        assert result.returncode == 0
        assert not result.timed_out


# ---------------------------------------------------------------------------
# SubprocessSandbox — resource limits (Unix only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="rlimit not available on Windows")
class TestSubprocessSandboxResourceLimits:
    """Resource limit tests — Unix only."""

    @pytest.fixture
    def sandbox(self) -> SubprocessSandbox:
        return SubprocessSandbox()

    @pytest.mark.asyncio
    async def test_cpu_limit_kills_process(self, sandbox: SubprocessSandbox) -> None:
        # Spin CPU for 10 seconds; limit to 1 CPU second
        limits = ResourceLimits(max_cpu_seconds=1.0)
        result = await sandbox.run(
            sys.executable + " -c \"while True: pass\"",
            timeout=10.0,
            resource_limits=limits,
        )
        # Process should be killed by SIGXCPU or similar
        assert result.returncode != 0 or result.timed_out

    @pytest.mark.asyncio
    async def test_resource_limits_with_normal_cmd(self, sandbox: SubprocessSandbox) -> None:
        """Normal command should succeed even with limits set."""
        limits = ResourceLimits(max_cpu_seconds=10.0, max_memory_mb=512)
        result = await sandbox.run("echo ok", resource_limits=limits)
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# BashTool with SubprocessSandbox
# ---------------------------------------------------------------------------

class TestBashToolWithSandbox:
    """Integration: BashTool routes through SubprocessSandbox."""

    @pytest.mark.asyncio
    async def test_basshtool_sandbox_basic(self) -> None:
        from tools.code.shell_tools import BashTool
        from tools.sandbox.subprocess_sandbox import SubprocessSandbox

        sandbox = SubprocessSandbox()
        tool = BashTool(sandbox=sandbox)
        result = await tool.execute({"command": "echo sandboxed"})
        assert result.success
        assert "sandboxed" in result.output["stdout"]
        assert result.metadata.get("sandbox") == "subprocess"

    @pytest.mark.asyncio
    async def test_bashtool_sandbox_timeout(self) -> None:
        from tools.code.shell_tools import BashTool
        from tools.sandbox.subprocess_sandbox import SubprocessSandbox

        sandbox = SubprocessSandbox()
        tool = BashTool(sandbox=sandbox)
        result = await tool.execute({
            "command": sys.executable + " -c \"import time; time.sleep(60)\"",
            "timeout": 0.2,
        })
        assert not result.success
        assert result.metadata.get("timed_out") is True

    @pytest.mark.asyncio
    async def test_bashtool_no_sandbox_still_works(self) -> None:
        """BashTool without sandbox falls back to direct subprocess."""
        from tools.code.shell_tools import BashTool

        tool = BashTool()  # no sandbox
        result = await tool.execute({"command": "echo direct"})
        assert result.success
        assert "direct" in result.output["stdout"]

    @pytest.mark.asyncio
    async def test_bashtool_sandbox_env(self) -> None:
        from tools.code.shell_tools import BashTool
        from tools.sandbox.subprocess_sandbox import SubprocessSandbox

        sandbox = SubprocessSandbox()
        tool = BashTool(sandbox=sandbox)
        result = await tool.execute({
            "command": "echo $TOOL_VAR",
            "env": {"TOOL_VAR": "from_tool"},
        })
        assert result.success
        assert "from_tool" in result.output["stdout"]


# ---------------------------------------------------------------------------
# DockerSandbox — skipped unless docker is available AND functional
# ---------------------------------------------------------------------------

_docker_cli_available = shutil.which("docker") is not None


def _docker_functional() -> bool:
    """Return True only if the Docker daemon is reachable AND a busybox/alpine
    image is already present locally (no network pull required).

    Runs `docker images -q busybox` synchronously at collection time.
    Skips if docker CLI is absent, daemon is unreachable, or no local image.
    """
    if not _docker_cli_available:
        return False
    import subprocess
    # Check daemon reachable
    try:
        ret = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        if ret.returncode != 0:
            return False
    except Exception:
        return False
    # Check if busybox image is already local (avoids network pull)
    try:
        ret = subprocess.run(
            ["docker", "images", "-q", "busybox:latest"],
            capture_output=True, timeout=5,
        )
        return bool(ret.stdout.strip())
    except Exception:
        return False


_docker_ready = _docker_functional()
_docker_skip_reason = (
    "Docker daemon not reachable or 'busybox:latest' not present locally "
    "(run: docker pull busybox)"
)


@pytest.mark.skipif(not _docker_ready, reason=_docker_skip_reason)
class TestDockerSandbox:
    """DockerSandbox integration tests.

    Requires:
    - Docker daemon running and reachable
    - `busybox:latest` image already pulled locally (`docker pull busybox`)

    Using busybox (not alpine) avoids any network-dependent pulls.
    """

    @pytest.fixture
    def sandbox(self):
        from tools.sandbox.docker_sandbox import DockerSandbox
        return DockerSandbox(image="busybox:latest")

    @pytest.mark.asyncio
    async def test_echo_in_container(self, sandbox) -> None:
        result = await sandbox.run("echo docker_ok")
        assert result.returncode == 0
        assert "docker_ok" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout_kills_container(self, sandbox) -> None:
        result = await sandbox.run("sleep 60", timeout=0.5)
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_env_var_in_container(self, sandbox) -> None:
        result = await sandbox.run("echo $MY_KEY", env={"MY_KEY": "docker_val"})
        assert "docker_val" in result.stdout

    @pytest.mark.asyncio
    async def test_sandbox_type(self, sandbox) -> None:
        assert sandbox.sandbox_type == "docker"


# ---------------------------------------------------------------------------
# SandboxBase contract tests
# ---------------------------------------------------------------------------

class TestSandboxBaseContract:
    """Verify SandboxResult and ResourceLimits Pydantic models."""

    def test_sandbox_result_defaults(self) -> None:
        r = SandboxResult(stdout="", stderr="", returncode=0)
        assert r.timed_out is False
        assert r.resource_exceeded is False
        assert r.duration_ms == 0.0

    def test_resource_limits_all_none(self) -> None:
        lim = ResourceLimits()
        assert lim.max_cpu_seconds is None
        assert lim.max_memory_mb is None

    def test_resource_limits_set_values(self) -> None:
        lim = ResourceLimits(max_cpu_seconds=5.0, max_memory_mb=128)
        assert lim.max_cpu_seconds == 5.0
        assert lim.max_memory_mb == 128
