"""Declarative Workflow models.

A Workflow declares a desired multi-step outcome; the WorkflowController
is responsible for orchestrating its execution. This follows the same
"declare what, not how" philosophy as Kubernetes Deployments.

A Workflow is composed of WorkflowSteps, each of which maps to an Agent
with a specific goal. Steps can depend on each other (DAG execution) or
run sequentially/in parallel.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.state.models import AgentConfig, ReconcileResult


class WorkflowStep(BaseModel):
    """Declaration of a single step in a workflow.

    A step is the unit of work: one Agent pursuing one goal.

    Attributes:
        step_id: Unique identifier within the workflow.
        name: Human-readable name.
        agent_config: Configuration for the agent that executes this step.
        goal: Natural language goal for this step's agent.
        constraints: Constraints passed to the agent.
        depends_on: IDs of steps that must complete before this step starts.
        require_approval: If True, a human must approve before this step runs.
        on_failure: What to do if this step fails:
            - "fail": Stop the entire workflow (default).
            - "skip": Mark step as skipped and continue.
            - "retry": Retry up to max_retries times before applying on_failure.
        max_retries: Number of retries before on_failure applies (only with "retry").
        context_from: Step IDs whose outputs are injected as context into this step.
    """

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    agent_config: AgentConfig
    goal: str
    constraints: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    require_approval: bool = False
    on_failure: Literal["fail", "skip", "retry"] = "fail"
    max_retries: int = Field(default=0, ge=0)
    context_from: list[str] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    """Declarative specification of a multi-step workflow.

    Analogous to a Kubernetes Deployment spec: users declare what they want,
    the WorkflowController handles execution.

    Attributes:
        workflow_id: Unique identifier.
        name: Human-readable workflow name.
        description: What this workflow does.
        steps: Ordered list of workflow steps.
        execution_mode: How steps are executed:
            - "sequential": Steps run one after another in declaration order.
            - "parallel": All steps run concurrently (ignores depends_on).
            - "dag": Respect depends_on to build an execution DAG.
        metadata: Arbitrary metadata for tracking.
    """

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    steps: list[WorkflowStep]
    execution_mode: Literal["sequential", "parallel", "dag"] = "sequential"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_step(self, step_id: str) -> WorkflowStep | None:
        """Look up a step by ID.

        Args:
            step_id: The step identifier.

        Returns:
            The WorkflowStep if found, None otherwise.
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def step_ids(self) -> list[str]:
        """Return all step IDs in declaration order."""
        return [s.step_id for s in self.steps]


# =============================================================================
# Execution Status Models
# =============================================================================


class WorkflowStepStatus(BaseModel):
    """Runtime status of a single workflow step.

    Attributes:
        step_id: The step identifier.
        step_name: Human-readable name.
        status: Current execution status.
        result: ReconcileResult if the step completed.
        retries: Number of retries attempted.
        error: Error message if the step failed.
        started_at: When execution began.
        completed_at: When execution finished.
    """

    step_id: str
    step_name: str
    status: Literal["pending", "running", "completed", "failed", "skipped", "awaiting_approval"]
    result: ReconcileResult | None = None
    retries: int = 0
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration of this step in milliseconds."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000


class WorkflowExecutionResult(BaseModel):
    """Final result of a workflow execution.

    Attributes:
        execution_id: Unique ID for this execution run.
        workflow_id: ID of the workflow spec that was executed.
        workflow_name: Human-readable workflow name.
        status: Overall execution status.
        step_statuses: Per-step status keyed by step_id.
        started_at: When the workflow started.
        completed_at: When the workflow finished (None if still running).
        error: Top-level error message if the workflow failed.
        total_steps: Total number of steps in the workflow.
        completed_steps: Number of successfully completed steps.
        failed_steps: Number of failed steps.
        skipped_steps: Number of skipped steps.
    """

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    workflow_name: str
    status: Literal["running", "completed", "failed", "cancelled"]
    step_statuses: dict[str, WorkflowStepStatus] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0

    @property
    def duration_ms(self) -> float | None:
        """Total workflow duration in milliseconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def get_step_output(self, step_id: str) -> dict[str, Any] | None:
        """Get the final state from a completed step's result.

        Used for passing outputs between steps (context_from).

        Args:
            step_id: The step to get output from.

        Returns:
            The step's final_state dict, or None if step is not completed.
        """
        status = self.step_statuses.get(step_id)
        if status is None or status.result is None:
            return None
        return status.result.final_state
