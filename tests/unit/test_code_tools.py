"""Tests for Code Agent tools (Phase 4).

Tests file_tools, search_tools, and shell_tools.
Uses tmp_path fixture (pytest built-in) for file system operations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.code.file_tools import EditFileTool, ReadFileTool, WriteFileTool
from tools.code.search_tools import GlobTool, GrepTool
from tools.code.shell_tools import BashTool, KillShellTool


# =============================================================================
# ReadFileTool
# =============================================================================


class TestReadFileTool:
    @pytest.fixture
    def tool(self) -> ReadFileTool:
        return ReadFileTool()

    async def test_read_existing_file(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert result.output == "hello world"

    async def test_file_not_found(self, tool: ReadFileTool, tmp_path: Path) -> None:
        result = await tool.execute({"path": str(tmp_path / "missing.txt")})
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_missing_path_param(self, tool: ReadFileTool) -> None:
        result = await tool.execute({})
        assert not result.success
        assert "path" in result.error

    def test_risk_level_is_low(self, tool: ReadFileTool) -> None:
        assert tool.risk_level == "low"
        assert tool.idempotent is True

    async def test_metadata_contains_size(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "sized.txt"
        f.write_text("abc", encoding="utf-8")
        result = await tool.execute({"path": str(f)})
        assert result.success
        assert result.metadata["size_bytes"] == 3


# =============================================================================
# WriteFileTool
# =============================================================================


class TestWriteFileTool:
    @pytest.fixture
    def tool(self) -> WriteFileTool:
        return WriteFileTool()

    async def test_write_new_file(self, tool: WriteFileTool, tmp_path: Path) -> None:
        f = tmp_path / "output.txt"
        result = await tool.execute({"path": str(f), "content": "test content"})
        assert result.success
        assert f.read_text() == "test content"

    async def test_overwrite_existing_file(self, tool: WriteFileTool, tmp_path: Path) -> None:
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        result = await tool.execute({"path": str(f), "content": "new content"})
        assert result.success
        assert f.read_text() == "new content"

    async def test_creates_parent_dirs(self, tool: WriteFileTool, tmp_path: Path) -> None:
        f = tmp_path / "a" / "b" / "c.txt"
        result = await tool.execute({"path": str(f), "content": "deep"})
        assert result.success
        assert f.read_text() == "deep"

    async def test_missing_content_param(self, tool: WriteFileTool, tmp_path: Path) -> None:
        result = await tool.execute({"path": str(tmp_path / "x.txt")})
        assert not result.success

    def test_risk_level_is_medium(self, tool: WriteFileTool) -> None:
        assert tool.risk_level == "medium"

    def test_side_effects_declared(self, tool: WriteFileTool) -> None:
        assert "writes_file" in tool.side_effects


# =============================================================================
# EditFileTool
# =============================================================================


class TestEditFileTool:
    @pytest.fixture
    def tool(self) -> EditFileTool:
        return EditFileTool()

    async def test_unique_edit_succeeds(self, tool: EditFileTool, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        result = await tool.execute({
            "path": str(f),
            "old_string": "return 1",
            "new_string": "return 42",
        })
        assert result.success
        assert "42" in f.read_text()

    async def test_old_string_not_found(self, tool: EditFileTool, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def foo(): pass")
        result = await tool.execute({"path": str(f), "old_string": "MISSING", "new_string": "x"})
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_ambiguous_edit_rejected(self, tool: EditFileTool, tmp_path: Path) -> None:
        f = tmp_path / "dup.py"
        f.write_text("x = 1\nx = 1\n")
        result = await tool.execute({"path": str(f), "old_string": "x = 1", "new_string": "x = 2"})
        assert not result.success
        assert "2 times" in result.error or "found" in result.error

    async def test_file_not_found(self, tool: EditFileTool, tmp_path: Path) -> None:
        result = await tool.execute({
            "path": str(tmp_path / "nope.py"),
            "old_string": "x",
            "new_string": "y",
        })
        assert not result.success

    def test_not_idempotent(self, tool: EditFileTool) -> None:
        assert tool.idempotent is False


# =============================================================================
# GlobTool
# =============================================================================


class TestGlobTool:
    @pytest.fixture
    def tool(self) -> GlobTool:
        return GlobTool()

    async def test_finds_files(self, tool: GlobTool, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await tool.execute({"pattern": "*.py", "path": str(tmp_path)})
        assert result.success
        assert result.metadata["count"] == 2

    async def test_no_matches(self, tool: GlobTool, tmp_path: Path) -> None:
        result = await tool.execute({"pattern": "*.xyz", "path": str(tmp_path)})
        assert result.success
        assert result.output == []

    async def test_invalid_path(self, tool: GlobTool, tmp_path: Path) -> None:
        result = await tool.execute({"pattern": "*.py", "path": str(tmp_path / "nonexistent")})
        assert not result.success

    async def test_recursive_glob(self, tool: GlobTool, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("")
        (tmp_path / "top.py").write_text("")
        result = await tool.execute({"pattern": "**/*.py", "path": str(tmp_path)})
        assert result.success
        assert result.metadata["count"] == 2


# =============================================================================
# GrepTool
# =============================================================================


class TestGrepTool:
    @pytest.fixture
    def tool(self) -> GrepTool:
        return GrepTool()

    async def test_finds_matching_lines(self, tool: GrepTool, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    pass\ndef world():\n    pass\n")
        result = await tool.execute({"pattern": r"def \w+", "path": str(f)})
        assert result.success
        assert result.metadata["count"] == 2

    async def test_case_insensitive_search(self, tool: GrepTool, tmp_path: Path) -> None:
        f = tmp_path / "text.txt"
        f.write_text("Hello World\nhello world\nHELLO WORLD\n")
        result = await tool.execute({"pattern": "hello", "path": str(f), "case_insensitive": True})
        assert result.success
        assert result.metadata["count"] == 3

    async def test_invalid_regex(self, tool: GrepTool, tmp_path: Path) -> None:
        result = await tool.execute({"pattern": "[invalid", "path": str(tmp_path)})
        assert not result.success
        assert "regex" in result.error.lower() or "pattern" in result.error.lower()

    async def test_file_pattern_filter(self, tool: GrepTool, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("import os\n")
        (tmp_path / "b.txt").write_text("import os\n")
        result = await tool.execute({
            "pattern": "import",
            "path": str(tmp_path),
            "file_pattern": "*.py",
        })
        assert result.success
        assert result.metadata["count"] == 1

    async def test_max_results_limit(self, tool: GrepTool, tmp_path: Path) -> None:
        f = tmp_path / "many.txt"
        f.write_text("\n".join(f"line {i}" for i in range(200)))
        result = await tool.execute({"pattern": "line", "path": str(f), "max_results": 5})
        assert result.success
        assert result.metadata["count"] == 5


# =============================================================================
# BashTool
# =============================================================================


class TestBashTool:
    @pytest.fixture
    def tool(self) -> BashTool:
        return BashTool()

    async def test_simple_command(self, tool: BashTool) -> None:
        result = await tool.execute({"command": "echo hello"})
        assert result.success
        assert "hello" in result.output["stdout"]

    async def test_command_failure(self, tool: BashTool) -> None:
        result = await tool.execute({"command": "exit 1"})
        assert not result.success
        assert result.output["returncode"] == 1

    async def test_timeout(self, tool: BashTool) -> None:
        result = await tool.execute({"command": "sleep 10", "timeout": 1})
        assert not result.success
        assert "timed out" in result.error.lower()

    async def test_missing_command_param(self, tool: BashTool) -> None:
        result = await tool.execute({})
        assert not result.success

    def test_risk_level_is_high(self, tool: BashTool) -> None:
        assert tool.risk_level == "high"
        assert tool.reversible is False

    async def test_dry_run_shows_command(self, tool: BashTool) -> None:
        preview = await tool.dry_run({"command": "rm -rf /", "timeout": 30})
        assert "rm -rf /" in preview.preview
        assert len(preview.warnings) > 0

    async def test_captures_stderr(self, tool: BashTool) -> None:
        result = await tool.execute({"command": "echo error >&2"})
        assert result.success
        assert "error" in result.output["stderr"]

    def test_side_effects_declared(self, tool: BashTool) -> None:
        assert "executes_shell_command" in tool.side_effects
        assert tool.idempotent is False


# =============================================================================
# KillShellTool
# =============================================================================


class TestKillShellTool:
    @pytest.fixture
    def tool(self) -> KillShellTool:
        return KillShellTool()

    async def test_kill_unknown_pid(self, tool: KillShellTool) -> None:
        result = await tool.execute({"pid": 999999})
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_missing_pid_param(self, tool: KillShellTool) -> None:
        result = await tool.execute({})
        assert not result.success

    def test_risk_level_is_high(self, tool: KillShellTool) -> None:
        assert tool.risk_level == "high"

    async def test_non_integer_pid_rejected(self, tool: KillShellTool) -> None:
        result = await tool.execute({"pid": "abc"})
        assert not result.success
        assert "integer" in result.error.lower()
