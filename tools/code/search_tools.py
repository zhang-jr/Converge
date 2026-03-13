"""Search tools for Code Agent: file pattern matching and content search."""

from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path
from typing import Any

from tools.base import ReadOnlyTool, ToolResult


class GlobTool(ReadOnlyTool):
    """Find files matching a glob pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts')."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern like '**/*.py'"},
                "path": {"type": "string", "description": "Root directory to search (default: cwd)"},
            },
            "required": ["pattern"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Find files matching the glob pattern.

        Args:
            params: Must contain 'pattern'; optionally 'path'.

        Returns:
            ToolResult with list of matching file paths.
        """
        pattern = params.get("pattern", "")
        root_str = params.get("path", ".")

        if not pattern:
            return ToolResult(success=False, error="'pattern' parameter is required")

        def _glob() -> list[str]:
            root = Path(root_str)
            if not root.is_dir():
                raise ValueError(f"Path is not a directory: {root_str}")
            matches = sorted(str(p) for p in root.glob(pattern))
            return matches

        try:
            matches = await asyncio.to_thread(_glob)
            return ToolResult(
                success=True,
                output=matches,
                metadata={"count": len(matches), "pattern": pattern, "root": root_str},
            )
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except OSError as e:
            return ToolResult(success=False, error=f"OS error during glob: {e}")


class GrepTool(ReadOnlyTool):
    """Search file contents using a regex pattern."""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents for a regex pattern. "
            "Returns matching lines with file paths and line numbers."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: cwd)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob filter for files (e.g., '*.py')",
                },
                "case_insensitive": {"type": "boolean", "default": False},
                "max_results": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum number of matches to return",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Search file contents for a regex pattern.

        Args:
            params: Must contain 'pattern'; optionally 'path', 'file_pattern',
                    'case_insensitive', 'max_results'.

        Returns:
            ToolResult with list of matching lines (file, line number, content).
        """
        pattern = params.get("pattern", "")
        root_str = params.get("path", ".")
        file_pattern = params.get("file_pattern")
        case_insensitive = params.get("case_insensitive", False)
        max_results = params.get("max_results", 100)

        if not pattern:
            return ToolResult(success=False, error="'pattern' parameter is required")

        def _grep() -> list[dict[str, Any]]:
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e

            root = Path(root_str)
            results: list[dict[str, Any]] = []

            # Collect candidate files
            if root.is_file():
                candidates = [root]
            else:
                candidates = [p for p in root.rglob("*") if p.is_file()]
                if file_pattern:
                    candidates = [p for p in candidates if fnmatch.fnmatch(p.name, file_pattern)]

            for file_path in candidates:
                if len(results) >= max_results:
                    break
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for lineno, line in enumerate(lines, 1):
                        if compiled.search(line):
                            results.append({
                                "file": str(file_path),
                                "line": lineno,
                                "content": line,
                            })
                            if len(results) >= max_results:
                                break
                except OSError:
                    continue  # skip unreadable files silently

            return results

        try:
            matches = await asyncio.to_thread(_grep)
            return ToolResult(
                success=True,
                output=matches,
                metadata={"count": len(matches), "pattern": pattern},
            )
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except OSError as e:
            return ToolResult(success=False, error=f"OS error during grep: {e}")
