"""File operation tools for Code Agent.

Provides read, write, and precise edit capabilities following
the ToolBase contract with proper side_effects declarations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tools.base import ReadOnlyTool, StateMutatingTool, ToolResult


class ReadFileTool(ReadOnlyTool):
    """Read the contents of a file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"},
            },
            "required": ["path"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Read file contents.

        Args:
            params: Must contain 'path'; optionally 'encoding'.

        Returns:
            ToolResult with file contents as output.
        """
        path_str = params.get("path", "")
        encoding = params.get("encoding", "utf-8")
        if not path_str:
            return ToolResult(success=False, error="'path' parameter is required")
        try:
            path = Path(path_str)
            content = await asyncio.to_thread(path.read_text, encoding=encoding)
            return ToolResult(
                success=True,
                output=content,
                metadata={"path": str(path.resolve()), "size_bytes": len(content.encode(encoding))},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path_str}")
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {path_str}")
        except UnicodeDecodeError as e:
            return ToolResult(success=False, error=f"Encoding error reading {path_str}: {e}")
        except OSError as e:
            return ToolResult(success=False, error=f"OS error reading {path_str}: {e}")


class WriteFileTool(StateMutatingTool):
    """Write content to a file (creates or overwrites)."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file if it doesn't exist, or overwrites it."

    @property
    def side_effects(self) -> list[str]:
        return ["writes_file"]

    @property
    def idempotent(self) -> bool:
        return True  # same content produces same file state

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
                "encoding": {"type": "string", "default": "utf-8"},
                "create_dirs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Create parent directories if missing",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Write content to a file.

        Args:
            params: Must contain 'path' and 'content'; optionally 'encoding', 'create_dirs'.

        Returns:
            ToolResult indicating success or failure.
        """
        path_str = params.get("path", "")
        content = params.get("content")
        encoding = params.get("encoding", "utf-8")
        create_dirs = params.get("create_dirs", True)

        if not path_str:
            return ToolResult(success=False, error="'path' parameter is required")
        if content is None:
            return ToolResult(success=False, error="'content' parameter is required")

        def _write() -> None:
            p = Path(path_str)
            if create_dirs:
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=encoding)

        try:
            await asyncio.to_thread(_write)
            return ToolResult(
                success=True,
                output=f"Written {len(content)} characters to {path_str}",
                metadata={
                    "path": str(Path(path_str).resolve()),
                    "bytes_written": len(content.encode(encoding)),
                },
            )
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {path_str}")
        except OSError as e:
            return ToolResult(success=False, error=f"OS error writing {path_str}: {e}")


class EditFileTool(StateMutatingTool):
    """Precisely edit a file by replacing an exact string."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing an exact occurrence of old_string with new_string. "
            "The old_string must be unique in the file for a safe edit."
        )

    @property
    def side_effects(self) -> list[str]:
        return ["writes_file"]

    @property
    def idempotent(self) -> bool:
        return False  # applying same edit twice changes or fails

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Edit a file by replacing an exact unique string.

        Args:
            params: Must contain 'path', 'old_string', 'new_string'; optionally 'encoding'.

        Returns:
            ToolResult indicating success or failure.
        """
        path_str = params.get("path", "")
        old_string = params.get("old_string")
        new_string = params.get("new_string")
        encoding = params.get("encoding", "utf-8")

        if not path_str:
            return ToolResult(success=False, error="'path' parameter is required")
        if old_string is None:
            return ToolResult(success=False, error="'old_string' parameter is required")
        if new_string is None:
            return ToolResult(success=False, error="'new_string' parameter is required")

        def _edit() -> str:
            p = Path(path_str)
            content = p.read_text(encoding=encoding)
            count = content.count(old_string)
            if count == 0:
                raise ValueError(f"old_string not found in {path_str}")
            if count > 1:
                raise ValueError(
                    f"old_string found {count} times in {path_str}; "
                    "provide more context for a unique match"
                )
            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content, encoding=encoding)
            return new_content

        try:
            await asyncio.to_thread(_edit)
            return ToolResult(
                success=True,
                output=f"Successfully edited {path_str}",
                metadata={"path": str(Path(path_str).resolve())},
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path_str}")
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {path_str}")
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except OSError as e:
            return ToolResult(success=False, error=f"OS error editing {path_str}: {e}")
