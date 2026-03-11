"""Unit tests for core/workflow/workflow.py and core/workflow/controller.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.state.models import AgentConfig, ReconcileResult
from core.workflow.controller import WorkflowController, _validate_dag
from core.workflow.workflow import (
    WorkflowExecutionResult,
    WorkflowSpec,
    WorkflowStep,
    WorkflowStepStatus,
)
from errors.exceptions import (
    AgentFrameworkError,
    WorkflowCycleError,
    WorkflowStepError,
)


# =============================================================================
# Helpers
# =============================================================================

def make_agent_config(agent_id: str = "agent") -> AgentConfig:
    return AgentConfig(agent_id=agent_id)


def make_step(
    step_id: str,
    name: str,
    depends_on: list[str] | None = None,
    context_from: list[str] | None = None,
    on_failure: str = "fail",
    max_retries: int = 0,
    require_approval: bool = False,
) -> WorkflowStep:
    return WorkflowStep(
        step_id=step_id,
        name=name,
        agent_config=make_agent_config(agent_id=f"agent-{step_id}"),
        goal=f"Goal for {name}",
        depends_on=depends_on or [],
        context_from=context_from or [],
        on_failure=on_failure,
        max_retries=max_retries,
        require_approval=require_approval,
    )


def converged_result(final_state: dict | None = None) -> ReconcileResult:
    """A ReconcileResult that represents successful convergence."""
    return ReconcileResult(
        status="converged",
        converged=True,
        final_state=final_state or {"status": "done"},
        trace_id="test-trace",
    )


def failed_result() -> ReconcileResult:
    """A ReconcileResult that represents a failed run."""
    return ReconcileResult(
        status="failed",
        converged=False,
        error="Agent did not converge",
        trace_id="test-trace",
    )


def make_runtime(side_effects: list) -> MagicMock:
    """Create a mock AgentRuntime whose run() returns values from side_effects."""
    runtime = MagicMock()
    runtime.run = AsyncMock(side_effect=side_effects)
    runtime.initialize = AsyncMock()
    runtime.shutdown = AsyncMock()
    return runtime


# =============================================================================
# WorkflowSpec Model
# =============================================================================


class TestWorkflowSpec:
    """Tests for WorkflowSpec and WorkflowStep models."""

    def test_get_step_found(self):
        """get_step() returns the step with the matching ID."""
        step = make_step("s1", "Step 1")
        spec = WorkflowSpec(name="W", steps=[step])
        assert spec.get_step("s1") is step

    def test_get_step_not_found(self):
        """get_step() returns None for an unknown step_id."""
        spec = WorkflowSpec(name="W", steps=[make_step("s1", "S1")])
        assert spec.get_step("ghost") is None

    def test_step_ids(self):
        """step_ids() returns all step IDs in declaration order."""
        spec = WorkflowSpec(
            name="W",
            steps=[make_step("a", "A"), make_step("b", "B"), make_step("c", "C")],
        )
        assert spec.step_ids() == ["a", "b", "c"]

    def test_step_defaults(self):
        """WorkflowStep defaults: depends_on=[], on_failure='fail', max_retries=0."""
        step = make_step("s1", "Step 1")
        assert step.depends_on == []
        assert step.on_failure == "fail"
        assert step.max_retries == 0
        assert step.require_approval is False

    def test_workflow_execution_result_get_step_output(self):
        """get_step_output() returns final_state for completed steps."""
        result = WorkflowExecutionResult(
            workflow_id="w",
            workflow_name="W",
            status="completed",
        )
        result.step_statuses["s1"] = WorkflowStepStatus(
            step_id="s1",
            step_name="S1",
            status="completed",
            result=converged_result(final_state={"key": "value"}),
        )
        output = result.get_step_output("s1")
        assert output == {"key": "value"}

    def test_workflow_execution_result_get_step_output_pending(self):
        """get_step_output() returns None for pending steps."""
        result = WorkflowExecutionResult(
            workflow_id="w",
            workflow_name="W",
            status="running",
        )
        result.step_statuses["s1"] = WorkflowStepStatus(
            step_id="s1",
            step_name="S1",
            status="pending",
        )
        assert result.get_step_output("s1") is None


# =============================================================================
# DAG Validation
# =============================================================================


class TestDagValidation:
    """Tests for _validate_dag()."""

    def test_valid_linear_dag(self):
        """A simple linear DAG passes validation."""
        spec = WorkflowSpec(
            name="Linear",
            execution_mode="dag",
            steps=[
                make_step("a", "A"),
                make_step("b", "B", depends_on=["a"]),
                make_step("c", "C", depends_on=["b"]),
            ],
        )
        _validate_dag(spec)  # Should not raise

    def test_valid_diamond_dag(self):
        """A diamond-shaped DAG passes validation."""
        spec = WorkflowSpec(
            name="Diamond",
            execution_mode="dag",
            steps=[
                make_step("root", "Root"),
                make_step("left", "Left", depends_on=["root"]),
                make_step("right", "Right", depends_on=["root"]),
                make_step("tip", "Tip", depends_on=["left", "right"]),
            ],
        )
        _validate_dag(spec)  # Should not raise

    def test_cycle_detected(self):
        """A cycle in depends_on raises WorkflowCycleError."""
        spec = WorkflowSpec(
            name="Cyclic",
            execution_mode="dag",
            steps=[
                make_step("a", "A", depends_on=["b"]),
                make_step("b", "B", depends_on=["a"]),
            ],
        )
        with pytest.raises(WorkflowCycleError) as exc_info:
            _validate_dag(spec)
        assert len(exc_info.value.cycle_path) > 0

    def test_self_loop_detected(self):
        """A step depending on itself raises WorkflowCycleError."""
        spec = WorkflowSpec(
            name="SelfLoop",
            execution_mode="dag",
            steps=[make_step("a", "A", depends_on=["a"])],
        )
        with pytest.raises(WorkflowCycleError):
            _validate_dag(spec)

    def test_unknown_dependency_raises(self):
        """A depends_on reference to an unknown step raises AgentFrameworkError."""
        spec = WorkflowSpec(
            name="BadDep",
            execution_mode="dag",
            steps=[make_step("a", "A", depends_on=["nonexistent"])],
        )
        with pytest.raises(AgentFrameworkError):
            _validate_dag(spec)


# =============================================================================
# WorkflowController — Sequential
# =============================================================================


class TestSequentialWorkflow:
    """Tests for sequential execution mode."""

    async def test_all_steps_run_in_order(self):
        """All steps execute in declaration order."""
        call_order = []

        async def run_side_effect(goal, agent_config, **kwargs):
            call_order.append(agent_config.agent_id)
            return converged_result()

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        spec = WorkflowSpec(
            name="Seq",
            execution_mode="sequential",
            steps=[
                make_step("s1", "Step 1"),
                make_step("s2", "Step 2"),
                make_step("s3", "Step 3"),
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "completed"
        assert result.completed_steps == 3
        assert call_order == ["agent-s1", "agent-s2", "agent-s3"]

    async def test_sequential_stops_on_hard_failure(self):
        """A step with on_failure='fail' stops the workflow."""
        runtime = make_runtime([converged_result(), failed_result()])

        spec = WorkflowSpec(
            name="Seq",
            execution_mode="sequential",
            steps=[
                make_step("s1", "Step 1"),
                make_step("s2", "Step 2"),  # fails
                make_step("s3", "Step 3"),  # should not run
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "failed"
        assert result.step_statuses["s2"].status == "failed"
        assert result.step_statuses["s3"].status == "pending"  # Never ran

    async def test_on_failure_skip_continues(self):
        """on_failure='skip' marks the step skipped and continues."""
        runtime = make_runtime([converged_result(), failed_result(), converged_result()])

        spec = WorkflowSpec(
            name="Seq",
            execution_mode="sequential",
            steps=[
                make_step("s1", "Step 1"),
                make_step("s2", "Step 2", on_failure="skip"),
                make_step("s3", "Step 3"),
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "completed"
        assert result.step_statuses["s2"].status == "skipped"
        assert result.step_statuses["s3"].status == "completed"

    async def test_retry_exhaustion_then_fail(self):
        """on_failure='retry' retries max_retries times then fails."""
        # All attempts fail
        runtime = make_runtime([failed_result()] * 5)

        spec = WorkflowSpec(
            name="Retry",
            execution_mode="sequential",
            steps=[make_step("s1", "Step 1", on_failure="retry", max_retries=2)],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "failed"
        assert result.step_statuses["s1"].retries == 3  # 1 initial + 2 retries

    async def test_context_passed_between_steps(self):
        """context_from causes prior step output to be injected as context."""
        received_contexts = []

        async def run_side_effect(goal, agent_config, context=None, **kwargs):
            received_contexts.append(context or {})
            return converged_result(final_state={"from": agent_config.agent_id})

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        spec = WorkflowSpec(
            name="ContextPass",
            execution_mode="sequential",
            steps=[
                make_step("s1", "Step 1"),
                make_step("s2", "Step 2", context_from=["s1"]),
            ],
        )

        controller = WorkflowController(runtime)
        await controller.run(spec)

        # Step 2 should have received step 1's output as context
        s2_context = received_contexts[1]
        assert "step_s1_output" in s2_context
        assert s2_context["step_s1_output"]["from"] == "agent-s1"


# =============================================================================
# WorkflowController — Parallel
# =============================================================================


class TestParallelWorkflow:
    """Tests for parallel execution mode."""

    async def test_all_steps_run(self):
        """All steps run and complete in parallel mode."""
        runtime = make_runtime([converged_result()] * 3)

        spec = WorkflowSpec(
            name="Parallel",
            execution_mode="parallel",
            steps=[
                make_step("s1", "Step 1"),
                make_step("s2", "Step 2"),
                make_step("s3", "Step 3"),
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "completed"
        assert result.completed_steps == 3

    async def test_partial_success(self):
        """When some steps fail, result reflects both completed and failed."""
        runtime = make_runtime([converged_result(), failed_result(), converged_result()])

        spec = WorkflowSpec(
            name="PartialParallel",
            execution_mode="parallel",
            steps=[
                make_step("s1", "Step 1", on_failure="skip"),
                make_step("s2", "Step 2", on_failure="skip"),
                make_step("s3", "Step 3", on_failure="skip"),
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.skipped_steps == 1
        assert result.completed_steps == 2


# =============================================================================
# WorkflowController — DAG
# =============================================================================


class TestDagWorkflow:
    """Tests for DAG execution mode."""

    async def test_respects_dependencies(self):
        """Steps with depends_on wait for their dependencies."""
        call_order = []

        async def run_side_effect(goal, agent_config, **kwargs):
            call_order.append(agent_config.agent_id)
            return converged_result()

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        spec = WorkflowSpec(
            name="DAG",
            execution_mode="dag",
            steps=[
                make_step("root", "Root"),
                make_step("child", "Child", depends_on=["root"]),
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "completed"
        assert call_order.index("agent-root") < call_order.index("agent-child")

    async def test_skips_child_when_parent_fails(self):
        """When a depended-upon step fails, dependent steps are skipped."""
        runtime = make_runtime([failed_result()])

        spec = WorkflowSpec(
            name="DAG-fail",
            execution_mode="dag",
            steps=[
                make_step("root", "Root"),
                make_step("child", "Child", depends_on=["root"]),
            ],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.step_statuses["root"].status == "failed"
        assert result.step_statuses["child"].status == "skipped"

    async def test_cycle_raises_before_execution(self):
        """WorkflowCycleError is raised before any step executes."""
        runtime = make_runtime([])

        spec = WorkflowSpec(
            name="Cyclic",
            execution_mode="dag",
            steps=[
                make_step("a", "A", depends_on=["b"]),
                make_step("b", "B", depends_on=["a"]),
            ],
        )

        controller = WorkflowController(runtime)
        with pytest.raises(WorkflowCycleError):
            await controller.run(spec)

        # runtime.run should never have been called
        runtime.run.assert_not_called()


# =============================================================================
# Human Approval Gate
# =============================================================================


class TestApprovalGate:
    """Tests for require_approval=True behavior."""

    async def test_approval_auto_approves_by_default(self):
        """Default on_approval_required() returns True (auto-approve)."""
        runtime = make_runtime([converged_result()])

        spec = WorkflowSpec(
            name="Approval",
            execution_mode="sequential",
            steps=[make_step("s1", "Sensitive Step", require_approval=True)],
        )

        controller = WorkflowController(runtime)
        result = await controller.run(spec)

        assert result.status == "completed"
        assert result.step_statuses["s1"].status == "completed"

    async def test_rejected_step_is_skipped(self):
        """When on_approval_required() returns False, the step is skipped."""

        class RejectingController(WorkflowController):
            async def on_approval_required(self, step, execution):
                return False

        runtime = make_runtime([])  # run() should not be called

        spec = WorkflowSpec(
            name="Approval",
            execution_mode="sequential",
            steps=[make_step("s1", "Sensitive Step", require_approval=True)],
        )

        controller = RejectingController(runtime)
        result = await controller.run(spec)

        assert result.step_statuses["s1"].status == "skipped"
        assert result.skipped_steps == 1
        runtime.run.assert_not_called()
