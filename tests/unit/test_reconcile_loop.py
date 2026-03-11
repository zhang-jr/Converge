"""Unit tests for ReconcileLoop."""

import pytest

from core.runtime.reconcile_loop import ReconcileLoop, SimpleReconcileLoop
from core.state.models import (
    DesiredState,
    LoopContext,
    ReconcileResult,
    StepOutput,
)
from core.state.sqlite_store import SQLiteStateStore
from errors.exceptions import ConvergenceTimeoutError, QualityProbeFailure
from probes.quality_probe import ProbeResult, QualityProbe


@pytest.fixture
async def state_store():
    """Create a fresh in-memory SQLite store for each test."""
    store = SQLiteStateStore(":memory:")
    yield store
    await store.close()


class MockProbe(QualityProbe):
    """Mock probe for testing."""

    def __init__(self, should_converge_at: int = 3):
        self.call_count = 0
        self.should_converge_at = should_converge_at

    async def evaluate(self, step_output, context):
        self.call_count += 1
        return ProbeResult(
            verdict="passed",
            confidence=0.8,
            reason="Mock evaluation",
            should_converge=self.call_count >= self.should_converge_at,
        )


class FailingProbe(QualityProbe):
    """Probe that always fails."""

    def __init__(self, verdict: str = "hard_fail"):
        self.verdict = verdict

    async def evaluate(self, step_output, context):
        return ProbeResult(
            verdict=self.verdict,
            confidence=0.1,
            reason="Always fails",
            should_converge=False,
        )


class TestSimpleReconcileLoop:
    """Tests for SimpleReconcileLoop."""

    async def test_converges_when_probe_says_so(
        self, state_store: SQLiteStateStore
    ):
        """Test loop converges when QualityProbe.should_converge=True."""
        probe = MockProbe(should_converge_at=2)

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action="Test action",
                result="Done",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=probe,
            act_callback=act_callback,
            safety_max_steps=10,
        )

        result = await loop.run(DesiredState(goal="Test goal"))

        assert result.status == "converged"
        assert result.converged is True
        assert result.total_steps == 2
        assert probe.call_count == 2

    async def test_timeout_on_max_steps(
        self, state_store: SQLiteStateStore
    ):
        """Test loop times out when max steps exceeded."""

        class NeverConvergeProbe(QualityProbe):
            async def evaluate(self, step_output, context):
                return ProbeResult(
                    verdict="passed",
                    confidence=0.8,
                    reason="Keep going",
                    should_converge=False,
                )

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action="Test action",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=NeverConvergeProbe(),
            act_callback=act_callback,
            safety_max_steps=5,
        )

        with pytest.raises(ConvergenceTimeoutError) as exc_info:
            await loop.run(DesiredState(goal="Test goal"))

        assert exc_info.value.max_steps == 5
        assert exc_info.value.steps_completed == 5

    async def test_fails_on_hard_fail(
        self, state_store: SQLiteStateStore
    ):
        """Test loop fails when probe returns hard_fail."""

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action="Test action",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=FailingProbe("hard_fail"),
            act_callback=act_callback,
        )

        with pytest.raises(QualityProbeFailure):
            await loop.run(DesiredState(goal="Test goal"))

    async def test_records_all_steps(
        self, state_store: SQLiteStateStore
    ):
        """Test that all steps are recorded in result."""
        step_count = 0

        async def act_callback(diff, context, loop):
            nonlocal step_count
            step_count += 1
            return StepOutput(
                step_number=context.current_step,
                action=f"Action {step_count}",
                result=f"Result {step_count}",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=MockProbe(should_converge_at=3),
            act_callback=act_callback,
        )

        result = await loop.run(DesiredState(goal="Test goal"))

        assert len(result.steps) == 3
        assert result.steps[0].action == "Action 1"
        assert result.steps[1].action == "Action 2"
        assert result.steps[2].action == "Action 3"

    async def test_provides_context_to_callback(
        self, state_store: SQLiteStateStore
    ):
        """Test that context is correctly passed to callback."""
        received_contexts = []

        async def act_callback(diff, context, loop):
            # Use model_copy() to capture the context at this point in time
            received_contexts.append(context.model_copy(deep=True))
            return StepOutput(
                step_number=context.current_step,
                action="Test",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=MockProbe(should_converge_at=2),
            act_callback=act_callback,
            agent_id="test-agent",
        )

        await loop.run(
            DesiredState(
                goal="Test goal",
                constraints=["constraint1"],
            )
        )

        assert len(received_contexts) == 2
        assert received_contexts[0].current_step == 1
        assert received_contexts[1].current_step == 2
        assert received_contexts[0].agent_id == "test-agent"
        assert received_contexts[0].desired_state.goal == "Test goal"

    async def test_history_accumulates(
        self, state_store: SQLiteStateStore
    ):
        """Test that history accumulates across steps."""
        histories = []

        async def act_callback(diff, context, loop):
            histories.append(len(context.history))
            return StepOutput(
                step_number=context.current_step,
                action="Test",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=MockProbe(should_converge_at=3),
            act_callback=act_callback,
        )

        await loop.run(DesiredState(goal="Test goal"))

        assert histories == [0, 1, 2]

    async def test_lifecycle_hooks_called(
        self, state_store: SQLiteStateStore
    ):
        """Test that lifecycle hooks are called."""
        events = []

        class HookedLoop(SimpleReconcileLoop):
            async def on_loop_start(self, context):
                events.append("start")

            async def on_step_complete(self, step):
                events.append(f"step_{step.step_number}")

            async def on_convergence(self, result):
                events.append("converge")

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action="Test",
            )

        loop = HookedLoop(
            state_store=state_store,
            quality_probe=MockProbe(should_converge_at=2),
            act_callback=act_callback,
        )

        await loop.run(DesiredState(goal="Test goal"))

        assert events == ["start", "step_1", "step_2", "converge"]

    async def test_on_failure_called(
        self, state_store: SQLiteStateStore
    ):
        """Test that on_failure is called on error."""
        failure_info = {}

        class HookedLoop(SimpleReconcileLoop):
            async def on_failure(self, error, steps_completed):
                failure_info["error"] = error
                failure_info["steps"] = steps_completed

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action="Test",
            )

        loop = HookedLoop(
            state_store=state_store,
            quality_probe=FailingProbe("hard_fail"),
            act_callback=act_callback,
        )

        with pytest.raises(QualityProbeFailure):
            await loop.run(DesiredState(goal="Test goal"))

        assert "error" in failure_info
        assert isinstance(failure_info["error"], QualityProbeFailure)

    async def test_trace_id_in_result(
        self, state_store: SQLiteStateStore
    ):
        """Test that trace_id is included in result."""

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action="Test",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=MockProbe(should_converge_at=1),
            act_callback=act_callback,
        )

        result = await loop.run(DesiredState(goal="Test goal"))

        assert result.trace_id is not None
        assert len(result.trace_id) > 0

    async def test_duration_tracked(
        self, state_store: SQLiteStateStore
    ):
        """Test that duration is tracked."""
        import asyncio

        async def act_callback(diff, context, loop):
            await asyncio.sleep(0.01)
            return StepOutput(
                step_number=context.current_step,
                action="Test",
            )

        loop = SimpleReconcileLoop(
            state_store=state_store,
            quality_probe=MockProbe(should_converge_at=1),
            act_callback=act_callback,
        )

        result = await loop.run(DesiredState(goal="Test goal"))

        assert result.duration_ms > 0
