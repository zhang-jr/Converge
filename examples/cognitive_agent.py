"""Phase 5 example: Cognitive loop enhancement.

Demonstrates:
1. Planning phase -- Agent generates ExecutionPlan before acting
2. Reflect step -- Each step reflects against the plan
3. SubTask decomposition -- Complex task broken into SubTasks
4. ConvergenceCriteriaProbe -- Structured termination conditions
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from core.runtime.agent_runtime import AgentRuntime
from core.runtime.reconcile_loop import SimpleReconcileLoop
from core.state.models import (
    AgentConfig,
    ConvergenceCriterion,
    DesiredState,
    ExecutionPlan,
    LoopContext,
    PlannedStep,
    StepOutput,
    SubTask,
)
from core.state.sqlite_store import SQLiteStateStore
from probes.quality_probe import ConvergenceCriteriaProbe, ProbeResult, QualityProbe


# ---------------------------------------------------------------------------
# Scenario 1: Planning + Reflect
# ---------------------------------------------------------------------------


class PlanningAgent(SimpleReconcileLoop):
    """Loop that generates a plan before executing."""

    async def _planning_phase(
        self, desired_state: DesiredState, context: LoopContext
    ) -> ExecutionPlan:
        """Generate a three-step execution plan."""
        return ExecutionPlan(
            agent_id=self._agent_id,
            goal=desired_state.goal,
            reasoning="Breaking the goal into sequential steps",
            steps=[
                PlannedStep(
                    step_index=0,
                    goal="Analyze the problem",
                    tool_hint="read_file",
                    expected_output="problem analysis completed",
                ),
                PlannedStep(
                    step_index=1,
                    goal="Implement solution",
                    tool_hint="write_file",
                    expected_output="solution implemented",
                ),
                PlannedStep(
                    step_index=2,
                    goal="Verify result",
                    tool_hint="bash",
                    expected_output="verification passed",
                ),
            ],
        )

    async def _act(self, diff: dict, context: LoopContext) -> StepOutput:
        """Execute the current planned step."""
        step_num = context.current_step
        plan = context.execution_plan
        if plan and step_num <= len(plan.steps):
            planned = plan.steps[step_num - 1]
            action = planned.goal
            result_text = planned.expected_output
        else:
            action = "Continuing toward goal"
            result_text = "completed"

        return StepOutput(
            step_number=step_num,
            action=action,
            reasoning=f"Following plan step {step_num}",
            result={"status": result_text},
        )


class ConvergeAfterPlanProbe(QualityProbe):
    """Converges after 3 steps (matching the plan)."""

    async def evaluate(
        self, step_output: StepOutput, context: LoopContext
    ) -> ProbeResult:
        should = context.current_step >= 3
        return ProbeResult(
            verdict="passed",
            confidence=0.9,
            reason=f"Step {context.current_step} done",
            should_converge=should,
        )


async def scenario_planning_with_reflection() -> None:
    """Scenario 1: Planning + Reflection."""
    print("\n=== Scenario 1: Planning + Reflection ===")  # noqa: T201 (example only)
    store = SQLiteStateStore(":memory:")
    loop = PlanningAgent(
        state_store=store,
        quality_probe=ConvergeAfterPlanProbe(),
        enable_planning=True,
        agent_id="planning-agent",
    )
    result = await loop.run(DesiredState(goal="Build a feature"))
    print(f"Status: {result.status}, Steps: {result.total_steps}")  # noqa: T201
    for step in result.steps:
        print(f"  Step {step.step_number}: {step.action}")  # noqa: T201
        if step.reflection:
            print(f"    Reflection: {step.reflection}")  # noqa: T201
    await store.close()


# ---------------------------------------------------------------------------
# Scenario 2: Planning mode (plan only, no execution)
# ---------------------------------------------------------------------------


async def scenario_planning_mode_only() -> None:
    """Scenario 2: Plan-only mode."""
    print("\n=== Scenario 2: Plan-only mode ===")  # noqa: T201
    store = SQLiteStateStore(":memory:")
    loop = PlanningAgent(
        state_store=store,
        quality_probe=ConvergeAfterPlanProbe(),
        enable_planning=True,
        planning_mode=True,
        agent_id="plan-only-agent",
    )
    result = await loop.run(DesiredState(goal="Refactor module"))
    print(f"Status: {result.status} (no steps executed -- plan only)")  # noqa: T201
    print(f"Steps executed: {result.total_steps}")  # noqa: T201

    # Retrieve persisted plan
    plans = await store.list("plan/")
    if plans:
        plan_data = plans[0].value
        print(f"Plan stored: {len(plan_data.get('steps', []))} steps planned")  # noqa: T201
    await store.close()


# ---------------------------------------------------------------------------
# Scenario 3: SubTask decomposition
# ---------------------------------------------------------------------------


async def scenario_subtask_decomposition() -> None:
    """Scenario 3: SubTask decomposition."""
    print("\n=== Scenario 3: SubTask decomposition ===")  # noqa: T201
    store = SQLiteStateStore(":memory:")
    runtime = AgentRuntime(state_store=store)
    await runtime.initialize()

    subtasks = [
        SubTask(goal="Read and analyze codebase structure", agent_id="sub-1"),
        SubTask(goal="Write unit tests for core module", agent_id="sub-2"),
        SubTask(goal="Run lint and fix issues", agent_id="sub-3"),
    ]

    print("Running subtasks in parallel...")  # noqa: T201
    results = await runtime.run_subtasks(subtasks, parallel=True)
    for r in results:
        print(f"  SubTask '{r.goal[:40]}': {r.status}")  # noqa: T201
    await runtime.shutdown()


# ---------------------------------------------------------------------------
# Scenario 4: ConvergenceCriteriaProbe with file_exists
# ---------------------------------------------------------------------------


async def scenario_convergence_criteria(tmp_dir: str) -> None:
    """Scenario 4: ConvergenceCriteriaProbe."""
    print("\n=== Scenario 4: ConvergenceCriteriaProbe ===")  # noqa: T201

    output_file = os.path.join(tmp_dir, "output.txt")
    desired = DesiredState(
        goal=f"Generate output file at {output_file}",
        convergence_criteria=[
            ConvergenceCriterion(
                criterion_type="file_exists",
                description="output file generated",
                params={"path": output_file},
            ),
        ],
    )

    call_count = [0]

    async def act_cb(diff, ctx, loop):
        call_count[0] += 1
        # On second step, create the file to satisfy criterion
        if call_count[0] >= 2:
            with open(output_file, "w") as f:
                f.write("generated")
        return StepOutput(
            step_number=ctx.current_step,
            action=f"Working on generating file (attempt {call_count[0]})",
        )

    store = SQLiteStateStore(":memory:")
    loop = SimpleReconcileLoop(
        state_store=store,
        quality_probe=ConvergenceCriteriaProbe(),
        act_callback=act_cb,
        safety_max_steps=10,
    )
    result = await loop.run(desired)
    print(f"Status: {result.status}, Steps: {result.total_steps}")  # noqa: T201
    print(f"File created: {os.path.exists(output_file)}")  # noqa: T201
    await store.close()


async def main() -> None:
    """Run all Phase 5 example scenarios."""
    await scenario_planning_with_reflection()
    await scenario_planning_mode_only()
    await scenario_subtask_decomposition()
    with tempfile.TemporaryDirectory() as tmp:
        await scenario_convergence_criteria(tmp)


if __name__ == "__main__":
    asyncio.run(main())
