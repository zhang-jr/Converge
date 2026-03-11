"""Base classes for the Tool system.

Every Tool must declare its side effects, reversibility, risk level,
and idempotency. The framework uses these declarations to enforce
safety policies like human-in-the-loop for high-risk operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of a tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        output: The output data from the tool.
        error: Error message if the tool failed.
        metadata: Additional metadata about the execution.
    """

    success: bool = True
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDryRunResult(BaseModel):
    """Result of a tool dry-run (preview without execution).

    Attributes:
        would_succeed: Prediction of whether execution would succeed.
        preview: Description of what would happen.
        affected_resources: List of resources that would be affected.
        warnings: Any warnings about potential issues.
    """

    would_succeed: bool = True
    preview: str = ""
    affected_resources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolBase(ABC):
    """Abstract base class for all Tools.

    Every Tool must declare:
    - side_effects: What external changes the tool makes
    - reversible: Whether the action can be undone
    - risk_level: low/medium/high (high triggers human-in-the-loop)
    - idempotent: Whether repeated calls have the same effect

    Tools must implement execute() and optionally dry_run() for high-risk tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    @abstractmethod
    def side_effects(self) -> list[str]:
        """List of side effects this tool can cause.

        Examples: ["writes_file", "sends_email", "modifies_database"]
        """
        ...

    @property
    @abstractmethod
    def reversible(self) -> bool:
        """Whether the action can be undone."""
        ...

    @property
    @abstractmethod
    def risk_level(self) -> Literal["low", "medium", "high"]:
        """Risk level of this tool.

        - low: Safe operations, no confirmation needed
        - medium: May need logging/audit, but auto-proceed
        - high: Requires human approval before execution
        """
        ...

    @property
    @abstractmethod
    def idempotent(self) -> bool:
        """Whether repeated calls with same params have same effect."""
        ...

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters.

        Override this to define expected parameters.
        """
        return {}

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            params: Tool-specific parameters.

        Returns:
            ToolResult with success status and output.

        Raises:
            ToolExecutionError: If the tool execution fails.
        """
        ...

    async def dry_run(self, params: dict[str, Any]) -> ToolDryRunResult:
        """Preview what would happen without executing.

        High-risk tools should implement this to allow users to
        review the action before approval.

        Args:
            params: Tool-specific parameters.

        Returns:
            ToolDryRunResult with preview information.
        """
        return ToolDryRunResult(
            would_succeed=True,
            preview=f"Would execute {self.name} with params: {params}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert tool metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "side_effects": self.side_effects,
            "reversible": self.reversible,
            "risk_level": self.risk_level,
            "idempotent": self.idempotent,
            "parameters_schema": self.parameters_schema,
        }


class ReadOnlyTool(ToolBase):
    """Base class for read-only tools.

    Provides sensible defaults for tools that don't modify anything.
    """

    @property
    def side_effects(self) -> list[str]:
        return []

    @property
    def reversible(self) -> bool:
        return True

    @property
    def risk_level(self) -> Literal["low", "medium", "high"]:
        return "low"

    @property
    def idempotent(self) -> bool:
        return True


class StateMutatingTool(ToolBase):
    """Base class for tools that mutate state.

    Sets default risk_level to medium for state-mutating operations.
    """

    @property
    def reversible(self) -> bool:
        return True

    @property
    def risk_level(self) -> Literal["low", "medium", "high"]:
        return "medium"
