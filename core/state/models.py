"""Data models for the Agent Framework state management.

All models use Pydantic v2 BaseModel for validation and serialization.
Uses Literal for finite enumerations instead of plain strings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# State Store Models
# =============================================================================


class StateEntry(BaseModel):
    """A single entry in the state store.

    Represents the atomic unit of state with optimistic locking support.

    Attributes:
        key: Unique identifier for this state entry.
        value: The actual state data as a dictionary.
        version: Optimistic lock version number, incremented on each update.
        updated_at: Timestamp of the last update.
        updated_by: ID of the agent or "system" that made the update.
    """

    key: str
    value: dict[str, Any]
    version: int = Field(default=1, ge=1)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: str = Field(default="system")

    model_config = {"frozen": False}


class StateChangeEvent(BaseModel):
    """Event emitted when state changes.

    Used by StateStore.watch() to notify observers of state changes.

    Attributes:
        key: The state key that changed.
        old_value: Previous value (None if created).
        new_value: New value (None if deleted).
        change_type: Type of change that occurred.
        version: New version number after the change.
        timestamp: When the change occurred.
        changed_by: ID of the agent or "system" that made the change.
    """

    key: str
    old_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    change_type: Literal["created", "updated", "deleted"]
    version: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    changed_by: str = Field(default="system")


# =============================================================================
# Reconcile Loop Models
# =============================================================================


class ConvergenceCriterion(BaseModel):
    """A single convergence criterion for structured termination.

    Declares a condition that must be satisfied for the reconcile loop
    to consider the goal achieved.

    Attributes:
        criterion_type: Type of criterion to check.
        description: Human-readable description of this criterion.
        params: Type-specific parameters (e.g., path for file_exists).
    """

    criterion_type: Literal["all_tests_pass", "lint_clean", "file_exists", "custom_probe"]
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class DesiredState(BaseModel):
    """Declaration of the desired state/goal for the reconcile loop.

    Users describe what they want, the framework figures out how to get there.

    Attributes:
        goal: Natural language description of the desired outcome.
        constraints: List of constraints that must be satisfied.
        context: Additional context for the agent (e.g., prior conversation).
        metadata: Arbitrary metadata for tracking and observability.
    """

    goal: str = Field(..., min_length=1)
    constraints: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    convergence_criteria: list[ConvergenceCriterion] = Field(default_factory=list)


class ToolCall(BaseModel):
    """Record of a tool invocation.

    Attributes:
        tool_name: Name of the tool that was called.
        params: Parameters passed to the tool.
        result: Result returned by the tool.
        duration_ms: Execution time in milliseconds.
        success: Whether the tool execution succeeded.
        error: Error message if the tool failed.
    """

    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


class StepOutput(BaseModel):
    """Output from a single reconcile loop step.

    Captures everything that happened in one iteration of the control loop.

    Attributes:
        step_number: Sequential step number (1-indexed).
        action: Description of the action taken in this step.
        reasoning: The agent's reasoning for taking this action.
        tool_calls: List of tools invoked in this step.
        state_changes: Keys that were modified in the state store.
        result: The output/result of this step.
        timestamp: When this step completed.
        llm_tokens_used: Total tokens consumed by the LLM call in this step.
        llm_latency_ms: Wall-clock latency of the LLM call in milliseconds.
    """

    step_number: int = Field(..., ge=1)
    action: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    state_changes: list[str] = Field(default_factory=list)
    result: Any = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    llm_tokens_used: int = 0
    llm_latency_ms: float = 0.0
    reflection: str = ""


class LoopContext(BaseModel):
    """Context passed to each step of the reconcile loop.

    Provides the full context needed for the agent to make decisions.

    Attributes:
        desired_state: The goal we're trying to achieve.
        current_step: Current step number.
        history: All previous step outputs.
        state_snapshot: Current state store snapshot (relevant keys).
        agent_id: ID of the agent running the loop.
        trace_id: Unique ID for tracing this reconcile run.
    """

    desired_state: DesiredState
    current_step: int = Field(default=1, ge=1)
    history: list[StepOutput] = Field(default_factory=list)
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    agent_id: str = ""
    trace_id: str = ""
    execution_plan: ExecutionPlan | None = None


class ReconcileResult(BaseModel):
    """Final result of a reconcile loop run.

    Attributes:
        status: Final status of the reconcile operation.
        steps: All steps executed during the reconcile.
        final_state: State snapshot after reconciliation.
        total_steps: Total number of steps executed.
        converged: Whether the loop converged successfully.
        error: Error information if the loop failed.
        duration_ms: Total execution time in milliseconds.
        trace_id: Unique ID for tracing this reconcile run.
    """

    status: Literal["converged", "failed", "timeout", "human_intervention"]
    steps: list[StepOutput] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    total_steps: int = Field(default=0, ge=0)
    converged: bool = False
    error: str | None = None
    duration_ms: float = 0.0
    trace_id: str = ""


# =============================================================================
# Human Intervention Models
# =============================================================================


class HumanDecision(BaseModel):
    """Response from a human intervention request.

    Attributes:
        approved: Whether the pending action was approved.
        feedback: Optional feedback or instructions from the human.
        modified_action: If the human modified the proposed action.
        timestamp: When the decision was made.
        decision_by: Identifier of who made the decision.
    """

    approved: bool
    feedback: str = ""
    modified_action: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_by: str = ""


# =============================================================================
# Agent Models
# =============================================================================


class AgentConfig(BaseModel):
    """Configuration for an Agent.

    Attributes:
        agent_id: Unique identifier for the agent.
        name: Human-readable name.
        description: Description of the agent's purpose.
        model: LLM model to use (via LiteLLM).
        system_prompt: Base system prompt for the agent.
        temperature: LLM temperature setting.
        max_tokens: Maximum tokens for LLM response.
        tools: List of tool names the agent can use.
        safety_max_steps: Maximum reconcile loop steps before timeout.
        confidence_threshold: Minimum confidence for auto-proceed.
    """

    agent_id: str = Field(..., min_length=1)
    name: str = ""
    description: str = ""
    model: str | None = Field(default=None)
    system_prompt: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    tools: list[str] = Field(default_factory=list)
    safety_max_steps: int = Field(default=50, ge=1)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


# =============================================================================
# Planning Models
# =============================================================================


class PlannedStep(BaseModel):
    """A single step in an execution plan.

    Attributes:
        step_index: Zero-based index of this step in the plan.
        goal: What this step should accomplish.
        tool_hint: Suggested tool name (may be empty).
        expected_output: Description of what success looks like.
    """

    step_index: int = Field(ge=0)
    goal: str
    tool_hint: str = ""
    expected_output: str = ""


class ExecutionPlan(BaseModel):
    """An execution plan generated before the reconcile loop runs.

    Contains a sequence of PlannedSteps that the agent intends to follow.

    Attributes:
        plan_id: Unique identifier for this plan.
        agent_id: ID of the agent that generated the plan.
        goal: The goal this plan addresses.
        steps: Ordered list of planned steps.
        created_at: When the plan was created.
        reasoning: Why the plan was structured this way.
    """

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    goal: str = ""
    steps: list[PlannedStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reasoning: str = ""


class SubTask(BaseModel):
    """A subtask that can be delegated to an agent.

    Supports parent-child relationships for task decomposition.

    Attributes:
        task_id: Unique identifier for this subtask.
        parent_task_id: ID of the parent task (None if top-level).
        agent_id: Which agent owns this subtask.
        goal: What this subtask should accomplish.
        status: Current execution status.
        output: Result data after completion.
        error: Error message if failed.
        created_at: When the subtask was created.
        completed_at: When the subtask finished.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: str | None = None
    agent_id: str = ""
    goal: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    output: Any = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


# LoopContext references ExecutionPlan which is defined after it.
# Pydantic v2 requires model_rebuild() to resolve forward references.
LoopContext.model_rebuild()
