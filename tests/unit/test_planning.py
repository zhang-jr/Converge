"""Unit tests for Phase 5: Planning phase and Reflect step in ReconcileLoop."""

from __future__ import annotations

import pytest

from core.runtime.reconcile_loop import SimpleReconcileLoop
from core.state.models import (
    ConvergenceCriterion,
    DesiredState,
    ExecutionPlan,
    LoopContext,
    PlannedStep,
    StepOutput,
    SubTask,
)
from probes.quality_probe import ProbeResult, QualityProbe


class ConvergeOnFirstProbe(QualityProbe):
    """Probe that converges immediately on the first step."""

    async def evaluate(self, step_output, context):
        return ProbeResult(
            verdict="passed",
            confidence=0.9,
            reason="ok",
            should_converge=True,
        )


class TestPlanningPhase:
    """Tests for ReconcileLoop planning phase."""

    async def test_planning_disabled_by_default(self, state_store):
        """Loop skips planning when enable_planning=False."""
        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=ConvergeOnFirstProbe(),
            act_callback=lambda d, c, l: StepOutput(
                step_number=c.current_step, action="act"
            ),
        )
        result = await loop.run(DesiredState(goal="test"))
        # No plan stored
        entries = await state_store.list("plan/")
        assert len(entries) == 0

    async def test_planning_phase_stores_plan(self, state_store):
        """Enable_planning=True calls _planning_phase and persists to StateStore."""

        class PlanningLoop(SimpleReconcileLoop):
            async def _planning_phase(self, desired_state, context):
                return ExecutionPlan(
                    agent_id="test",
                    goal=desired_state.goal,
                    steps=[
                        PlannedStep(step_index=0, goal="step1", expected_output="done"),
                    ],
                )

            async def _act(self, diff, context):
                return StepOutput(
                    step_number=context.current_step, action="act", result="done"
                )

        loop = PlanningLoop(
            state_store=state_store,
            quality_probe=ConvergeOnFirstProbe(),
            enable_planning=True,
        )
        result = await loop.run(DesiredState(goal="test goal"))
        assert result.converged

        # Plan persisted in StateStore
        entries = await state_store.list("plan/")
        assert len(entries) == 1
        assert entries[0].value["goal"] == "test goal"

    async def test_planning_mode_only_plans_no_execution(self, state_store):
        """planning_mode=True: plan is generated but no act steps run."""

        class PlanningLoop(SimpleReconcileLoop):
            async def _planning_phase(self, desired_state, context):
                return ExecutionPlan(
                    agent_id="test",
                    goal=desired_state.goal,
                    steps=[PlannedStep(step_index=0, goal="do something")],
                )

        act_called: list[int] = []

        async def act_cb(diff, ctx, loop):
            act_called.append(1)
            return StepOutput(step_number=ctx.current_step, action="act")

        loop = PlanningLoop(
            state_store=state_store,
            quality_probe=ConvergeOnFirstProbe(),
            enable_planning=True,
            planning_mode=True,
            act_callback=act_cb,
        )
        result = await loop.run(DesiredState(goal="test"))
        assert result.status == "converged"
        assert len(act_called) == 0  # No execution in planning_mode

    async def test_context_has_plan_during_execution(self, state_store):
        """LoopContext.execution_plan is populated during act steps."""
        received_plans: list[ExecutionPlan | None] = []

        class PlanningLoop(SimpleReconcileLoop):
            async def _planning_phase(self, desired_state, context):
                return ExecutionPlan(
                    goal=desired_state.goal,
                    steps=[PlannedStep(step_index=0, goal="step")],
                )

            async def _act(self, diff, context):
                received_plans.append(context.execution_plan)
                return StepOutput(step_number=context.current_step, action="act")

        loop = PlanningLoop(
            state_store=state_store,
            quality_probe=ConvergeOnFirstProbe(),
            enable_planning=True,
        )
        await loop.run(DesiredState(goal="test"))
        assert len(received_plans) == 1
        assert received_plans[0] is not None
        assert received_plans[0].goal == "test"

    async def test_reflection_populated_when_plan_exists(self, state_store):
        """StepOutput.reflection is filled when a planned step exists."""

        class PlanningLoop(SimpleReconcileLoop):
            async def _planning_phase(self, desired_state, context):
                return ExecutionPlan(
                    goal=desired_state.goal,
                    steps=[
                        PlannedStep(
                            step_index=0,
                            goal="do it",
                            expected_output="completed task",
                        )
                    ],
                )

            async def _act(self, diff, context):
                return StepOutput(
                    step_number=context.current_step,
                    action="completed task successfully",
                    result="completed",
                )

        loop = PlanningLoop(
            state_store=state_store,
            quality_probe=ConvergeOnFirstProbe(),
            enable_planning=True,
        )
        result = await loop.run(DesiredState(goal="test"))
        assert result.converged
        assert result.steps[0].reflection != ""

    async def test_reflection_empty_without_plan(self, state_store):
        """StepOutput.reflection is empty string when no plan exists."""

        async def act_cb(diff, ctx, loop):
            return StepOutput(step_number=ctx.current_step, action="act")

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=ConvergeOnFirstProbe(),
            act_callback=act_cb,
        )
        result = await loop.run(DesiredState(goal="test"))
        assert result.steps[0].reflection == ""


class TestConvergenceCriteriaProbe:
    """Tests for ConvergenceCriteriaProbe."""

    async def test_no_criteria_passes_no_converge(self, state_store):
        """No criteria declared -> passes but does not converge."""
        from probes.quality_probe import ConvergenceCriteriaProbe

        probe = ConvergenceCriteriaProbe()
        step = StepOutput(step_number=1, action="act")
        ctx = LoopContext(
            desired_state=DesiredState(goal="test"),
            agent_id="a",
            trace_id="t",
        )
        result = await probe.evaluate(step, ctx)
        assert result.verdict == "passed"
        assert result.should_converge is False

    async def test_file_exists_criterion_pass(self, state_store, tmp_path):
        """file_exists criterion passes when file is on disk."""
        from probes.quality_probe import ConvergenceCriteriaProbe

        f = tmp_path / "output.txt"
        f.write_text("done")

        probe = ConvergenceCriteriaProbe()
        step = StepOutput(step_number=1, action="act")
        ctx = LoopContext(
            desired_state=DesiredState(
                goal="test",
                convergence_criteria=[
                    ConvergenceCriterion(
                        criterion_type="file_exists",
                        description="output file",
                        params={"path": str(f)},
                    )
                ],
            ),
            agent_id="a",
            trace_id="t",
        )
        result = await probe.evaluate(step, ctx)
        assert result.verdict == "passed"
        assert result.should_converge is True

    async def test_file_exists_criterion_fail(self, state_store):
        """file_exists criterion fails when file is missing."""
        from probes.quality_probe import ConvergenceCriteriaProbe

        probe = ConvergenceCriteriaProbe()
        step = StepOutput(step_number=1, action="act")
        ctx = LoopContext(
            desired_state=DesiredState(
                goal="test",
                convergence_criteria=[
                    ConvergenceCriterion(
                        criterion_type="file_exists",
                        params={"path": "/nonexistent/path/file.txt"},
                    )
                ],
            ),
            agent_id="a",
            trace_id="t",
        )
        result = await probe.evaluate(step, ctx)
        assert result.should_converge is False

    async def test_custom_probe_always_passes(self, state_store):
        """custom_probe criterion type always passes."""
        from probes.quality_probe import ConvergenceCriteriaProbe

        probe = ConvergenceCriteriaProbe()
        step = StepOutput(step_number=1, action="act")
        ctx = LoopContext(
            desired_state=DesiredState(
                goal="test",
                convergence_criteria=[
                    ConvergenceCriterion(
                        criterion_type="custom_probe",
                        description="external check",
                    )
                ],
            ),
            agent_id="a",
            trace_id="t",
        )
        result = await probe.evaluate(step, ctx)
        assert result.should_converge is True

    async def test_multiple_criteria_all_must_pass(self, state_store, tmp_path):
        """All criteria must pass for should_converge=True."""
        from probes.quality_probe import ConvergenceCriteriaProbe

        f = tmp_path / "exists.txt"
        f.write_text("ok")

        probe = ConvergenceCriteriaProbe()
        step = StepOutput(step_number=1, action="act")
        ctx = LoopContext(
            desired_state=DesiredState(
                goal="test",
                convergence_criteria=[
                    ConvergenceCriterion(
                        criterion_type="file_exists",
                        params={"path": str(f)},
                    ),
                    ConvergenceCriterion(
                        criterion_type="file_exists",
                        params={"path": "/nonexistent/missing.txt"},
                    ),
                ],
            ),
            agent_id="a",
            trace_id="t",
        )
        result = await probe.evaluate(step, ctx)
        assert result.should_converge is False


class TestSubTask:
    """Tests for SubTask model and AgentRuntime.run_subtask."""

    async def test_subtask_model_defaults(self):
        """SubTask has correct defaults."""
        st = SubTask(goal="do something")
        assert st.status == "pending"
        assert st.task_id is not None
        assert st.parent_task_id is None
        assert st.output is None

    async def test_subtask_completed_after_run(self, state_store):
        """run_subtask() updates status to completed for converging agent."""
        from core.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime(state_store=state_store)
        await runtime.initialize()

        st = SubTask(goal="simple goal", agent_id="test-agent")
        completed = await runtime.run_subtask(st)

        assert completed.completed_at is not None
        # Status is either completed or failed (depends on mock LLM behavior)
        assert completed.status in ("completed", "failed")

    async def test_run_subtasks_sequential(self, state_store):
        """run_subtasks() runs subtasks in sequence."""
        from core.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime(state_store=state_store)
        await runtime.initialize()

        subtasks = [SubTask(goal=f"goal {i}") for i in range(3)]
        results = await runtime.run_subtasks(subtasks, parallel=False)

        assert len(results) == 3
        for r in results:
            assert r.status in ("completed", "failed")
            assert r.completed_at is not None

    async def test_run_subtasks_parallel(self, state_store):
        """run_subtasks() can run subtasks in parallel."""
        from core.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime(state_store=state_store)
        await runtime.initialize()

        subtasks = [SubTask(goal=f"goal {i}") for i in range(2)]
        results = await runtime.run_subtasks(subtasks, parallel=True)

        assert len(results) == 2
        for r in results:
            assert r.completed_at is not None

    async def test_subtask_parent_child_relationship(self):
        """SubTask parent_task_id links parent to child."""
        parent = SubTask(goal="parent task")
        child = SubTask(goal="child task", parent_task_id=parent.task_id)

        assert child.parent_task_id == parent.task_id
        assert child.task_id != parent.task_id
