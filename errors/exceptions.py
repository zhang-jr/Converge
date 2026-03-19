"""Agent Framework exception hierarchy.

All exceptions inherit from AgentFrameworkError and carry context information
for debugging and observability.
"""

from __future__ import annotations

from typing import Any


class AgentFrameworkError(Exception):
    """Base exception for all Agent Framework errors.

    All exceptions carry context information including agent_id, step number,
    and other relevant debugging information.

    Args:
        message: Human-readable error description.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.step = step
        self.context = context or {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.agent_id:
            parts.append(f"agent_id={self.agent_id}")
        if self.step is not None:
            parts.append(f"step={self.step}")
        if self.context:
            parts.append(f"context={self.context}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "agent_id": self.agent_id,
            "step": self.step,
            "context": self.context,
        }


# =============================================================================
# State Store Errors
# =============================================================================


class StateStoreError(AgentFrameworkError):
    """Base exception for state store operations.

    Args:
        message: Human-readable error description.
        key: The state key involved in the error.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        key: str | None = None,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if key:
            ctx["key"] = key
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.key = key


class VersionConflictError(StateStoreError):
    """Raised when optimistic lock version does not match.

    This indicates a concurrent modification conflict. The ReconcileLoop
    should retry with exponential backoff.

    Args:
        message: Human-readable error description.
        key: The state key with version conflict.
        expected_version: The version the caller expected.
        actual_version: The actual version in the store.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        key: str,
        expected_version: int,
        actual_version: int,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["expected_version"] = expected_version
        ctx["actual_version"] = actual_version
        super().__init__(message, key=key, agent_id=agent_id, step=step, context=ctx)
        self.expected_version = expected_version
        self.actual_version = actual_version


# =============================================================================
# Reconcile Errors
# =============================================================================


class ReconcileError(AgentFrameworkError):
    """Base exception for reconcile loop errors.

    Args:
        message: Human-readable error description.
        steps_completed: Number of steps completed before the error.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        steps_completed: int = 0,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["steps_completed"] = steps_completed
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.steps_completed = steps_completed


class ConvergenceTimeoutError(ReconcileError):
    """Raised when reconcile loop exceeds safety_max_steps without converging.

    This is a safety mechanism to prevent infinite loops. The task should be
    reviewed and possibly split into smaller subtasks.

    Args:
        message: Human-readable error description.
        max_steps: The maximum steps limit that was exceeded.
        steps_completed: Number of steps completed before timeout.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        max_steps: int,
        steps_completed: int,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["max_steps"] = max_steps
        super().__init__(
            message,
            steps_completed=steps_completed,
            agent_id=agent_id,
            step=step,
            context=ctx,
        )
        self.max_steps = max_steps


class LoopDetectedError(ReconcileError):
    """Raised when a reasoning loop is detected.

    The agent is repeating similar actions without making progress.
    Human intervention may be required.

    Args:
        message: Human-readable error description.
        loop_pattern: Description of the detected loop pattern.
        steps_completed: Number of steps completed before detection.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        loop_pattern: str,
        steps_completed: int,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["loop_pattern"] = loop_pattern
        super().__init__(
            message,
            steps_completed=steps_completed,
            agent_id=agent_id,
            step=step,
            context=ctx,
        )
        self.loop_pattern = loop_pattern


# =============================================================================
# Tool Errors
# =============================================================================


class ToolExecutionError(AgentFrameworkError):
    """Raised when a tool execution fails.

    Args:
        message: Human-readable error description.
        tool_name: Name of the tool that failed.
        tool_params: Parameters passed to the tool.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        tool_params: dict[str, Any] | None = None,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["tool_name"] = tool_name
        if tool_params:
            ctx["tool_params"] = tool_params
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.tool_name = tool_name
        self.tool_params = tool_params or {}


class ToolPermissionError(AgentFrameworkError):
    """Raised when an agent lacks permission to use a tool.

    Args:
        message: Human-readable error description.
        tool_name: Name of the tool access was denied for.
        required_permission: The permission that was required.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        required_permission: str | None = None,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["tool_name"] = tool_name
        if required_permission:
            ctx["required_permission"] = required_permission
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.tool_name = tool_name
        self.required_permission = required_permission


# =============================================================================
# Flow Control Signals
# =============================================================================


class HumanInterventionRequired(AgentFrameworkError):
    """Signal that human intervention is needed.

    This is a flow control signal, not an error. It's raised when:
    - Tool risk_level is "high" and requires approval
    - Confidence score falls below threshold
    - Workflow declares a require_approval checkpoint

    Args:
        message: Human-readable explanation of why intervention is needed.
        reason: Categorized reason for intervention.
        pending_action: Description of the action awaiting approval.
        agent_id: ID of the agent that raised the signal.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        pending_action: str | None = None,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["reason"] = reason
        if pending_action:
            ctx["pending_action"] = pending_action
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.reason = reason
        self.pending_action = pending_action


class WorkflowError(AgentFrameworkError):
    """Base exception for workflow execution errors.

    Args:
        message: Human-readable error description.
        workflow_id: ID of the workflow that raised the error.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        workflow_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if workflow_id:
            ctx["workflow_id"] = workflow_id
        super().__init__(message, context=ctx)
        self.workflow_id = workflow_id


class WorkflowStepError(WorkflowError):
    """Raised when a workflow step fails and its on_failure policy is 'fail'.

    Args:
        message: Human-readable error description.
        step_id: The step that failed.
        step_name: The step's human-readable name.
        workflow_id: ID of the parent workflow.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        step_id: str,
        step_name: str,
        workflow_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["step_id"] = step_id
        ctx["step_name"] = step_name
        super().__init__(message, workflow_id=workflow_id, context=ctx)
        self.step_id = step_id
        self.step_name = step_name


class WorkflowCycleError(WorkflowError):
    """Raised when a workflow DAG contains a dependency cycle.

    Args:
        message: Human-readable error description.
        cycle_path: The step IDs that form the cycle.
        workflow_id: ID of the workflow with the cycle.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        cycle_path: list[str],
        workflow_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["cycle_path"] = cycle_path
        super().__init__(message, workflow_id=workflow_id, context=ctx)
        self.cycle_path = cycle_path


class SandboxError(AgentFrameworkError):
    """Raised when sandbox execution encounters an error.

    Args:
        message: Human-readable error description.
        sandbox_type: The type of sandbox that failed.
        cmd: The command that was attempted.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        sandbox_type: str,
        cmd: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["sandbox_type"] = sandbox_type
        if cmd:
            ctx["cmd"] = cmd
        super().__init__(message, context=ctx)
        self.sandbox_type = sandbox_type
        self.cmd = cmd


class RollbackError(AgentFrameworkError):
    """Raised when a state rollback operation fails.

    Args:
        message: Human-readable error description.
        snapshot_id: The snapshot ID that failed to restore.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        snapshot_id: str | None = None,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if snapshot_id:
            ctx["snapshot_id"] = snapshot_id
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.snapshot_id = snapshot_id


class QualityProbeFailure(AgentFrameworkError):
    """Raised when a quality probe fails with hard_fail verdict.

    Contains the full ProbeResult for analysis.

    Args:
        message: Human-readable error description.
        verdict: The probe verdict (typically "hard_fail").
        confidence: The confidence score from the probe.
        probe_reason: The reason provided by the probe.
        agent_id: ID of the agent that raised the error.
        step: Current step number in the reconcile loop.
        context: Additional context information.
    """

    def __init__(
        self,
        message: str,
        *,
        verdict: str,
        confidence: float,
        probe_reason: str,
        agent_id: str | None = None,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["verdict"] = verdict
        ctx["confidence"] = confidence
        ctx["probe_reason"] = probe_reason
        super().__init__(message, agent_id=agent_id, step=step, context=ctx)
        self.verdict = verdict
        self.confidence = confidence
        self.probe_reason = probe_reason
