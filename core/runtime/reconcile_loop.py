"""ReconcileLoop - the control loop that drives agent execution.

Implements the observe → diff → act → verify → repeat pattern.
Terminates when QualityProbe.should_converge=True or safety_max_steps is reached.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from core.state.models import (
    DesiredState,
    ExecutionPlan,
    HumanDecision,
    LoopContext,
    PlannedStep,
    ReconcileResult,
    StepOutput,
    ToolCall,
)
from errors.exceptions import (
    ConvergenceTimeoutError,
    HumanInterventionRequired,
    LoopDetectedError,
    QualityProbeFailure,
    RollbackError,
)
from observability.tracer import Tracer
from probes.quality_probe import DefaultQualityProbe, ProbeResult, QualityProbe

if TYPE_CHECKING:
    from core.runtime.human_intervention import HumanInterventionHandler
    from core.state.state_store import StateStore
    from tools.base import ToolBase
    from tools.registry import ToolRegistry


class ReconcileLoop(ABC):
    """Control loop for agent execution.

    Drives the observe → diff → act → verify → repeat cycle until
    the goal is achieved or an error occurs.

    Subclasses must implement:
    - _observe: Gather current state
    - _diff: Compute difference from desired state
    - _act: Take action to move toward desired state

    Lifecycle hooks can be overridden:
    - on_loop_start: Called before the loop begins
    - on_step_complete: Called after each step
    - on_convergence: Called when goal is achieved
    - on_failure: Called on unrecoverable error
    - on_human_intervention_needed: Called when human input is required

    Args:
        state_store: StateStore for reading/writing state.
        tool_registry: Registry for accessing tools.
        quality_probe: Probe for evaluating step quality.
        tracer: Tracer for observability.
        safety_max_steps: Maximum steps before timeout (default 50).
        confidence_threshold: Minimum confidence for auto-proceed.
        retry_on_conflict: Whether to retry on version conflicts.
        max_conflict_retries: Maximum retries on version conflict.
        step_callback: Optional async callable invoked after each completed step.
            Signature: ``async def callback(step: StepOutput) -> None``.
            Useful for streaming step events to external consumers (e.g. API server).
        enable_rollback: If True, a StateStore snapshot is taken before each
            ``_act`` call.  On unrecoverable failure the loop attempts to
            restore state to the pre-act snapshot via :meth:`StateStore.restore`.
            Requires the StateStore implementation to support snapshots
            (e.g. SQLiteStateStore).  Silently skips if not supported.
    """

    def __init__(
        self,
        state_store: StateStore,
        tool_registry: ToolRegistry | None = None,
        quality_probe: QualityProbe | None = None,
        tracer: Tracer | None = None,
        safety_max_steps: int = 50,
        confidence_threshold: float = 0.7,
        retry_on_conflict: bool = True,
        max_conflict_retries: int = 3,
        agent_id: str = "",
        human_intervention_handler: HumanInterventionHandler | None = None,
        enable_planning: bool = False,
        planning_mode: bool = False,
        step_callback: Callable[[StepOutput], Awaitable[None]] | None = None,
        enable_rollback: bool = False,
    ) -> None:
        self._state_store = state_store
        self._tool_registry = tool_registry
        self._quality_probe = quality_probe or DefaultQualityProbe()
        self._tracer = tracer or Tracer(agent_id=agent_id)
        self._safety_max_steps = safety_max_steps
        self._confidence_threshold = confidence_threshold
        self._retry_on_conflict = retry_on_conflict
        self._max_conflict_retries = max_conflict_retries
        self._agent_id = agent_id
        self._human_intervention_handler = human_intervention_handler
        self._enable_planning = enable_planning
        self._planning_mode = planning_mode
        self._step_callback = step_callback
        self._enable_rollback = enable_rollback

    async def _planning_phase(
        self,
        desired_state: DesiredState,
        context: LoopContext,
    ) -> ExecutionPlan | None:
        """Generate an ExecutionPlan before the loop starts.

        Subclasses override this to use LLM for real planning.
        Default implementation returns None (no planning).

        The plan is persisted to StateStore under key ``f"plan/{context.trace_id}"``.

        Args:
            desired_state: The goal to plan for.
            context: Current loop context.

        Returns:
            ExecutionPlan or None if planning is not implemented.
        """
        return None

    async def _reflect_step(
        self,
        step_output: StepOutput,
        planned_step: PlannedStep | None,
        context: LoopContext,
    ) -> str:
        """Reflect on whether this step met the planned expectation.

        Subclasses override this for LLM-based reflection.
        Default: compare step result against expected_output using keyword matching.

        Args:
            step_output: Output from the current step.
            planned_step: The corresponding planned step (may be None).
            context: Current loop context.

        Returns:
            A reflection string stored in step_output.reflection.
        """
        if planned_step is None:
            return ""
        expected = planned_step.expected_output.lower()
        result_str = str(step_output.result).lower() if step_output.result else ""
        action_str = step_output.action.lower()
        combined = result_str + " " + action_str
        if expected and any(kw in combined for kw in expected.split()):
            return f"Step met expectation: {planned_step.expected_output}"
        return f"Step may not have met expectation: {planned_step.expected_output}"

    async def run(self, desired_state: DesiredState) -> ReconcileResult:
        """Execute the reconcile loop until convergence or failure.

        Args:
            desired_state: The goal to achieve.

        Returns:
            ReconcileResult with final status and all steps.

        Raises:
            ConvergenceTimeoutError: If safety_max_steps is exceeded.
            LoopDetectedError: If a reasoning loop is detected.
            QualityProbeFailure: If probe returns hard_fail.
        """
        start_time = time.monotonic()
        trace_id = self._tracer.start_trace()

        context = LoopContext(
            desired_state=desired_state,
            current_step=1,
            history=[],
            state_snapshot={},
            agent_id=self._agent_id,
            trace_id=trace_id,
        )

        await self.on_loop_start(context)

        # Planning phase (if enabled)
        if self._enable_planning:
            plan = await self._planning_phase(desired_state, context)
            if plan is not None:
                context.execution_plan = plan
                # Persist plan to StateStore
                plan_key = f"plan/{trace_id}"
                await self._state_store.put(
                    plan_key,
                    plan.model_dump(mode="json"),
                    updated_by=self._agent_id or "system",
                )
                self._tracer.log_state_change(
                    plan_key,
                    old_value=None,
                    new_value={"plan_id": plan.plan_id, "steps": len(plan.steps)},
                    change_type="created",
                )

        steps: list[StepOutput] = []
        final_status = "failed"
        error_message: str | None = None

        # In planning_mode, only generate the plan — do not execute
        if self._planning_mode:
            if context.execution_plan is not None:
                final_status = "converged"
            else:
                error_message = "Planning mode enabled but _planning_phase returned None"
        else:
            # Track the most recent pre-act snapshot for rollback
            _last_snapshot_id: str | None = None

            try:
                for step_num in range(1, self._safety_max_steps + 1):
                    context.current_step = step_num
                    context.state_snapshot = await self._get_state_snapshot()

                    observed = await self._observe(context)
                    diff = await self._diff(observed, desired_state)

                    if not diff:
                        final_status = "converged"
                        break

                    # Take a snapshot before acting (rollback support)
                    if self._enable_rollback:
                        try:
                            _last_snapshot_id = await self._state_store.snapshot()
                        except NotImplementedError:
                            pass  # StateStore doesn't support snapshots — skip silently

                    step_output = await self._act(diff, context)
                    step_output.step_number = step_num

                    # Reflect on step vs plan
                    planned_step: PlannedStep | None = None
                    if context.execution_plan:
                        plan_steps = context.execution_plan.steps
                        step_idx = step_num - 1
                        if step_idx < len(plan_steps):
                            planned_step = plan_steps[step_idx]
                    reflection = await self._reflect_step(step_output, planned_step, context)
                    step_output.reflection = reflection

                    steps.append(step_output)
                    context.history = steps.copy()

                    self._tracer.log_step(step_output)

                    probe_result = await self._quality_probe.evaluate(step_output, context)
                    self._tracer.log_probe_result(probe_result, self._quality_probe.name)

                    if probe_result.verdict == "hard_fail":
                        raise QualityProbeFailure(
                            "Quality probe hard fail",
                            verdict=probe_result.verdict,
                            confidence=probe_result.confidence,
                            probe_reason=probe_result.reason,
                            agent_id=self._agent_id,
                            step=step_num,
                        )

                    if probe_result.confidence < self._confidence_threshold:
                        decision = await self.on_human_intervention_needed(
                            f"Confidence {probe_result.confidence:.2f} below threshold",
                            context,
                        )
                        if not decision.approved:
                            final_status = "human_intervention"
                            error_message = "Human rejected continuation"
                            break

                    await self.on_step_complete(step_output)

                    # Notify external consumer (e.g. API server WebSocket stream)
                    if self._step_callback is not None:
                        try:
                            await self._step_callback(step_output)
                        except Exception:
                            pass  # Never let callback failure interrupt the loop

                    if probe_result.should_converge:
                        final_status = "converged"
                        break

                else:
                    raise ConvergenceTimeoutError(
                        f"Exceeded safety_max_steps ({self._safety_max_steps})",
                        max_steps=self._safety_max_steps,
                        steps_completed=len(steps),
                        agent_id=self._agent_id,
                    )

            except ConvergenceTimeoutError:
                final_status = "timeout"
                error_message = f"Exceeded {self._safety_max_steps} steps"
                await self._attempt_rollback(_last_snapshot_id, len(steps))
                raise

            except LoopDetectedError as e:
                final_status = "failed"
                error_message = str(e)
                await self._attempt_rollback(_last_snapshot_id, len(steps))
                await self.on_failure(e, len(steps))
                raise

            except QualityProbeFailure as e:
                final_status = "failed"
                error_message = e.probe_reason
                await self._attempt_rollback(_last_snapshot_id, len(steps))
                await self.on_failure(e, len(steps))
                raise

            except HumanInterventionRequired:
                final_status = "human_intervention"

            except Exception as e:
                final_status = "failed"
                error_message = str(e)
                self._tracer.log_error(e)
                await self._attempt_rollback(_last_snapshot_id, len(steps))
                await self.on_failure(e, len(steps))
                raise

        end_time = time.monotonic()
        duration_ms = (end_time - start_time) * 1000

        result = ReconcileResult(
            status=final_status,
            steps=steps,
            final_state=await self._get_state_snapshot(),
            total_steps=len(steps),
            converged=(final_status == "converged"),
            error=error_message,
            duration_ms=duration_ms,
            trace_id=trace_id,
        )

        if final_status == "converged":
            await self.on_convergence(result)

        self._tracer.end_trace(final_status)

        return result

    async def _attempt_rollback(
        self,
        snapshot_id: str | None,
        steps_completed: int,
    ) -> None:
        """Attempt to restore state to the last pre-act snapshot.

        Silently skips if rollback is disabled, snapshot_id is None,
        or the StateStore doesn't support snapshots.

        Args:
            snapshot_id: The snapshot ID to restore, or None.
            steps_completed: Used only for logging context.
        """
        if not self._enable_rollback or snapshot_id is None:
            return
        try:
            await self._state_store.restore(snapshot_id)
            self._tracer.log_state_change(
                key="__rollback__",
                old_value=None,
                new_value={"snapshot_id": snapshot_id, "steps_completed": steps_completed},
                change_type="updated",
            )
        except (NotImplementedError, RollbackError) as e:
            self._tracer.log_error(e)

    async def _get_state_snapshot(self) -> dict[str, Any]:
        """Get current state snapshot."""
        entries = await self._state_store.list()
        return {entry.key: entry.value for entry in entries}

    async def _observe(self, context: LoopContext) -> dict[str, Any]:
        """Observe the current state.

        Override this to implement custom observation logic.

        Args:
            context: Current loop context.

        Returns:
            Dictionary representing observed state.
        """
        return context.state_snapshot

    async def _diff(
        self,
        observed: dict[str, Any],
        desired: DesiredState,
    ) -> dict[str, Any] | None:
        """Compute difference between observed and desired state.

        Override this to implement custom diff logic.

        Args:
            observed: Current observed state.
            desired: The desired state/goal.

        Returns:
            Dictionary describing what needs to change, or None if converged.
        """
        return {"goal": desired.goal, "observed": observed}

    async def _act(
        self,
        diff: dict[str, Any],
        context: LoopContext,
    ) -> StepOutput:
        """Take action to move toward desired state.

        Must be overridden by subclasses to implement the actual
        agent logic (LLM calls, tool execution, etc.).

        Args:
            diff: What needs to change.
            context: Current loop context.

        Returns:
            StepOutput describing the action taken.
        """
        return StepOutput(
            step_number=context.current_step,
            action="No action (base implementation)",
            reasoning="ReconcileLoop base class does not implement _act",
        )

    async def _execute_tool(
        self,
        tool: ToolBase,
        params: dict[str, Any],
        context: LoopContext | None = None,
    ) -> ToolCall:
        """Execute a tool and return the result.

        High-risk tools trigger human intervention before execution.
        Handles error wrapping and timing.

        Args:
            tool: The tool to execute.
            params: Parameters for the tool.
            context: Current loop context (required for high-risk approval).

        Returns:
            ToolCall record with results.
        """
        # High-risk gate: require human approval before execution
        if tool.risk_level == "high" and context is not None:
            decision = await self.on_human_intervention_needed(
                reason=f"High-risk tool '{tool.name}' requires approval",
                context=context,
                pending_action={"tool": tool.name, "params": params},
            )
            if not decision.approved:
                return ToolCall(
                    tool_name=tool.name,
                    params=params,
                    success=False,
                    error=f"Human rejected high-risk tool '{tool.name}': {decision.feedback}",
                )

        start = time.monotonic()
        try:
            result = await tool.execute(params)
            duration = (time.monotonic() - start) * 1000

            tool_call = ToolCall(
                tool_name=tool.name,
                params=params,
                result=result.output,
                duration_ms=duration,
                success=result.success,
                error=result.error,
            )

            self._tracer.log_tool_call(
                tool.name,
                params,
                result.output,
                result.success,
                result.error,
                duration,
            )

            return tool_call

        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            tool_call = ToolCall(
                tool_name=tool.name,
                params=params,
                duration_ms=duration,
                success=False,
                error=str(e),
            )

            self._tracer.log_tool_call(
                tool.name,
                params,
                None,
                False,
                str(e),
                duration,
            )

            return tool_call

    async def on_loop_start(self, context: LoopContext) -> None:
        """Called before the reconcile loop begins.

        Override to add custom initialization logic.

        Args:
            context: Initial loop context.
        """
        pass

    async def on_step_complete(self, step: StepOutput) -> None:
        """Called after each step completes successfully.

        Override to add custom post-step logic.

        Args:
            step: The completed step output.
        """
        pass

    async def on_convergence(self, result: ReconcileResult) -> None:
        """Called when the loop converges (goal achieved).

        Override to add custom completion logic.

        Args:
            result: The final reconcile result.
        """
        pass

    async def on_failure(self, error: Exception, steps_completed: int) -> None:
        """Called when the loop fails with an unrecoverable error.

        Override to add custom error handling.

        Args:
            error: The exception that caused the failure.
            steps_completed: Number of steps completed before failure.
        """
        pass

    async def on_human_intervention_needed(
        self,
        reason: str,
        context: LoopContext,
        pending_action: dict[str, Any] | None = None,
    ) -> HumanDecision:
        """Called when human intervention is required.

        If a HumanInterventionHandler was provided at construction time,
        delegates to it. Otherwise auto-approves (suitable for testing).

        Args:
            reason: Why intervention is needed.
            context: Current loop context.
            pending_action: Optional description of the pending action.

        Returns:
            HumanDecision with approval and feedback.
        """
        self._tracer.log_human_intervention(reason)
        if self._human_intervention_handler is not None:
            return await self._human_intervention_handler.request_approval(
                reason=reason,
                context=context,
                pending_action=pending_action,
            )
        return HumanDecision(
            approved=True,
            feedback="Auto-approved (default implementation)",
        )


class SimpleReconcileLoop(ReconcileLoop):
    """Simple reconcile loop for basic use cases.

    Provides a straightforward implementation that can be configured
    with callback functions for observe/diff/act.
    """

    def __init__(
        self,
        state_store: StateStore,
        act_callback: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize simple reconcile loop.

        Args:
            state_store: StateStore for state management.
            act_callback: Async function called for each action step.
            **kwargs: Additional arguments for ReconcileLoop.
        """
        super().__init__(state_store, **kwargs)
        self._act_callback = act_callback

    async def _act(
        self,
        diff: dict[str, Any],
        context: LoopContext,
    ) -> StepOutput:
        """Execute the act callback if provided.

        Handles both async and sync callbacks transparently.
        """
        if self._act_callback is not None:
            result = self._act_callback(diff, context, self)
            if asyncio.iscoroutine(result):
                return await result
            return result

        return StepOutput(
            step_number=context.current_step,
            action="No action callback provided",
            reasoning="SimpleReconcileLoop requires an act_callback",
        )
