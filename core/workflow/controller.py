"""WorkflowController - orchestrates declarative workflow execution.

The controller takes a WorkflowSpec and drives its execution according to
the declared execution_mode. It manages:
- DAG validation (cycle detection)
- Step scheduling based on dependency completion
- Failure handling per step policy (fail / skip / retry)
- Output passing between steps (context_from)
- Human-in-the-loop checkpoints (require_approval)

Follows the control loop philosophy: observe the step graph state,
find ready steps, execute them, update status, repeat.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from core.state.models import AgentConfig, DesiredState
from core.workflow.workflow import (
    WorkflowExecutionResult,
    WorkflowSpec,
    WorkflowStep,
    WorkflowStepStatus,
)
from errors.exceptions import (
    AgentFrameworkError,
    HumanInterventionRequired,
    WorkflowCycleError,
    WorkflowStepError,
)

if TYPE_CHECKING:
    from core.runtime.agent_runtime import AgentRuntime


class WorkflowController:
    """Executes a declarative WorkflowSpec.

    Supports three execution modes:
    - sequential: Steps run in declaration order, one at a time.
    - parallel: All steps run concurrently (dependencies ignored).
    - dag: Steps run as soon as all their depends_on steps complete.

    For human approval checkpoints, the controller calls on_approval_required()
    which subclasses can override. The default implementation auto-approves
    (suitable for testing).

    Usage:
        runtime = AgentRuntime(...)
        async with runtime:
            controller = WorkflowController(runtime)
            result = await controller.run(spec)
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        """Initialize the workflow controller.

        Args:
            runtime: The AgentRuntime used to execute individual agent steps.
        """
        self._runtime = runtime

    async def run(self, spec: WorkflowSpec) -> WorkflowExecutionResult:
        """Execute a workflow spec.

        Args:
            spec: The workflow to execute.

        Returns:
            WorkflowExecutionResult with per-step statuses and overall outcome.

        Raises:
            WorkflowCycleError: If the DAG has a cycle.
            WorkflowStepError: If a step fails with on_failure="fail".
        """
        if spec.execution_mode == "dag":
            _validate_dag(spec)

        execution = WorkflowExecutionResult(
            workflow_id=spec.workflow_id,
            workflow_name=spec.name,
            status="running",
            total_steps=len(spec.steps),
        )

        # Initialize all steps as pending
        for step in spec.steps:
            execution.step_statuses[step.step_id] = WorkflowStepStatus(
                step_id=step.step_id,
                step_name=step.name,
                status="pending",
            )

        try:
            if spec.execution_mode == "sequential":
                await self._run_sequential(spec, execution)
            elif spec.execution_mode == "parallel":
                await self._run_parallel(spec, execution)
            else:  # dag
                await self._run_dag(spec, execution)

            # Determine final status
            if execution.failed_steps > 0:
                execution.status = "failed"
            else:
                execution.status = "completed"

        except (WorkflowStepError, WorkflowCycleError):
            execution.status = "failed"
            raise
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            raise
        finally:
            execution.completed_at = datetime.utcnow()

        return execution

    async def _run_sequential(
        self,
        spec: WorkflowSpec,
        execution: WorkflowExecutionResult,
    ) -> None:
        """Run steps in declaration order, one at a time."""
        for step in spec.steps:
            should_stop = await self._execute_step(step, spec, execution)
            if should_stop:
                break

    async def _run_parallel(
        self,
        spec: WorkflowSpec,
        execution: WorkflowExecutionResult,
    ) -> None:
        """Run all steps concurrently."""
        tasks = [
            asyncio.create_task(self._execute_step(step, spec, execution))
            for step in spec.steps
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_dag(
        self,
        spec: WorkflowSpec,
        execution: WorkflowExecutionResult,
    ) -> None:
        """Run steps respecting depends_on relationships.

        A step becomes eligible when all its dependencies have completed
        (or been skipped). Uses an event-driven approach with asyncio events.
        """
        # One asyncio.Event per step: set when the step is done (any terminal state)
        step_done: dict[str, asyncio.Event] = {
            s.step_id: asyncio.Event() for s in spec.steps
        }

        async def run_step_when_ready(step: WorkflowStep) -> None:
            # Wait for all dependencies to finish
            for dep_id in step.depends_on:
                await step_done[dep_id].wait()

            # If any required dependency failed (and is not skipped), skip this step
            for dep_id in step.depends_on:
                dep_status = execution.step_statuses[dep_id]
                if dep_status.status == "failed":
                    execution.step_statuses[step.step_id].status = "skipped"
                    execution.skipped_steps += 1
                    step_done[step.step_id].set()
                    return

            await self._execute_step(step, spec, execution)
            step_done[step.step_id].set()

        tasks = [
            asyncio.create_task(run_step_when_ready(step))
            for step in spec.steps
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_step(
        self,
        step: WorkflowStep,
        spec: WorkflowSpec,
        execution: WorkflowExecutionResult,
    ) -> bool:
        """Execute a single workflow step.

        Handles approval gates, retries, and failure policies.

        Args:
            step: The step to execute.
            spec: The parent workflow spec.
            execution: The execution result being built.

        Returns:
            True if the workflow should stop after this step (hard failure).
        """
        step_status = execution.step_statuses[step.step_id]

        # Human approval gate
        if step.require_approval:
            step_status.status = "awaiting_approval"
            approved = await self.on_approval_required(step, execution)
            if not approved:
                step_status.status = "skipped"
                execution.skipped_steps += 1
                return False

        step_status.status = "running"
        step_status.started_at = datetime.utcnow()

        # Build context from dependency outputs
        context = self._gather_context(step, execution)

        retries = 0
        while True:
            try:
                result = await self._runtime.run(
                    goal=step.goal,
                    agent_config=step.agent_config,
                    constraints=step.constraints,
                    context=context,
                )

                step_status.result = result
                step_status.completed_at = datetime.utcnow()

                if result.converged:
                    step_status.status = "completed"
                    execution.completed_steps += 1
                    return False
                else:
                    raise WorkflowStepError(
                        f"Step '{step.name}' agent did not converge: {result.error}",
                        step_id=step.step_id,
                        step_name=step.name,
                    )

            except WorkflowStepError:
                retries += 1
                step_status.retries = retries

                if step.on_failure == "retry" and retries <= step.max_retries:
                    continue  # Retry the loop

                step_status.status = "failed"
                step_status.error = (
                    f"Step did not converge after {retries} attempt(s)"
                )
                step_status.completed_at = datetime.utcnow()
                execution.failed_steps += 1

                if step.on_failure == "skip":
                    step_status.status = "skipped"
                    execution.failed_steps -= 1
                    execution.skipped_steps += 1
                    return False

                # on_failure == "fail" or retry exhausted
                execution.error = step_status.error
                return True  # Stop the workflow

            except Exception as e:
                step_status.status = "failed"
                step_status.error = str(e)
                step_status.completed_at = datetime.utcnow()
                execution.failed_steps += 1

                if step.on_failure == "skip":
                    step_status.status = "skipped"
                    execution.failed_steps -= 1
                    execution.skipped_steps += 1
                    return False

                execution.error = str(e)
                return True

    def _gather_context(
        self,
        step: WorkflowStep,
        execution: WorkflowExecutionResult,
    ) -> dict[str, Any]:
        """Build the context dict by collecting outputs from context_from steps.

        Args:
            step: The step that needs context.
            execution: Current execution result with completed step outputs.

        Returns:
            Context dict to pass to the agent.
        """
        context: dict[str, Any] = {}
        for src_step_id in step.context_from:
            output = execution.get_step_output(src_step_id)
            if output is not None:
                context[f"step_{src_step_id}_output"] = output
        return context

    async def on_approval_required(
        self,
        step: WorkflowStep,
        execution: WorkflowExecutionResult,
    ) -> bool:
        """Called when a step requires human approval before executing.

        Override this in subclasses to implement interactive approval UX.
        Default implementation auto-approves (suitable for testing).

        Args:
            step: The step awaiting approval.
            execution: Current workflow execution context.

        Returns:
            True to approve (proceed), False to skip the step.
        """
        return True


# =============================================================================
# DAG Validation
# =============================================================================


def _validate_dag(spec: WorkflowSpec) -> None:
    """Validate that the workflow DAG has no cycles.

    Args:
        spec: The workflow to validate.

    Raises:
        WorkflowCycleError: If a cycle is detected.
        AgentFrameworkError: If a depends_on references an unknown step.
    """
    step_ids = {s.step_id for s in spec.steps}

    # Verify all depends_on references are valid
    for step in spec.steps:
        for dep_id in step.depends_on:
            if dep_id not in step_ids:
                raise AgentFrameworkError(
                    f"Step '{step.step_id}' depends on unknown step '{dep_id}'",
                    context={"workflow_id": spec.workflow_id},
                )

    # Kahn's algorithm: detect cycles via topological sort
    in_degree: dict[str, int] = {s.step_id: 0 for s in spec.steps}
    adjacency: dict[str, list[str]] = {s.step_id: [] for s in spec.steps}

    for step in spec.steps:
        for dep_id in step.depends_on:
            adjacency[dep_id].append(step.step_id)
            in_degree[step.step_id] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    visited_count = 0

    while queue:
        node = queue.pop(0)
        visited_count += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited_count != len(spec.steps):
        # Some nodes were never reached — they are part of a cycle
        cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
        raise WorkflowCycleError(
            f"Workflow '{spec.name}' contains a dependency cycle",
            cycle_path=cycle_nodes,
        )
