"""Unit tests for ConfidenceProbe."""

from __future__ import annotations

import pytest

from core.state.models import DesiredState, LoopContext, StepOutput, ToolCall
from probes.confidence_probe import ConfidenceProbe


def _ctx() -> LoopContext:
    return LoopContext(
        desired_state=DesiredState(goal="test"),
        agent_id="test",
    )


def _step(
    action: str = "do something",
    reasoning: str = "",
    tool_calls: list[ToolCall] | None = None,
    state_changes: list[str] | None = None,
    result: object = None,
    step_number: int = 1,
) -> StepOutput:
    return StepOutput(
        step_number=step_number,
        action=action,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
        state_changes=state_changes or [],
        result=result,
    )


class TestConfidenceProbe:
    @pytest.fixture()
    def probe(self) -> ConfidenceProbe:
        return ConfidenceProbe(hard_threshold=0.3, soft_threshold=0.6, min_reasoning_length=50)

    # ------------------------------------------------------------------
    # Tool success score
    # ------------------------------------------------------------------

    async def test_all_tools_succeed_boosts_score(self, probe: ConfidenceProbe) -> None:
        """All successful tools → high confidence."""
        tools = [
            ToolCall(tool_name="t1", success=True),
            ToolCall(tool_name="t2", success=True),
        ]
        step = _step(
            reasoning="therefore I conclude because of the result analysis",
            tool_calls=tools,
            state_changes=["key1"],
        )
        result = await probe.evaluate(step, _ctx())
        assert result.verdict == "passed"
        assert result.confidence >= 0.6

    async def test_all_tools_fail_lowers_confidence(self, probe: ConfidenceProbe) -> None:
        """All failed tools → low tool score."""
        tools = [
            ToolCall(tool_name="t1", success=False, error="err"),
            ToolCall(tool_name="t2", success=False, error="err"),
        ]
        step = _step(tool_calls=tools)  # no reasoning, no state changes
        result = await probe.evaluate(step, _ctx())
        # tool_score=0.0 → weighted 0.0*0.4 = 0.0; reasoning=0, progress ~0.3
        assert result.confidence < 0.4

    async def test_no_tool_calls_full_tool_score(self, probe: ConfidenceProbe) -> None:
        """No tool calls → tool_score defaults to 1.0."""
        step = _step(reasoning="because I found the result")
        result = await probe.evaluate(step, _ctx())
        # tool_score=1.0, some reasoning, no state changes
        assert result.confidence > 0.4

    # ------------------------------------------------------------------
    # Reasoning quality score
    # ------------------------------------------------------------------

    async def test_rich_reasoning_boosts_score(self, probe: ConfidenceProbe) -> None:
        """Long reasoning with keywords boosts score."""
        reasoning = "therefore the answer is X because the analysis determined Y, result confirmed"
        step = _step(reasoning=reasoning, state_changes=["k"])
        result = await probe.evaluate(step, _ctx())
        assert result.confidence >= 0.6

    async def test_empty_reasoning_lowers_score(self, probe: ConfidenceProbe) -> None:
        """Empty reasoning → lowest reasoning score."""
        step = _step(reasoning="")
        result = await probe.evaluate(step, _ctx())
        assert result.confidence < 0.7

    # ------------------------------------------------------------------
    # Progress score
    # ------------------------------------------------------------------

    async def test_state_changes_full_progress(self, probe: ConfidenceProbe) -> None:
        """State changes present → full progress score (1.0)."""
        step = _step(
            reasoning="therefore because result",
            state_changes=["key1", "key2"],
        )
        result = await probe.evaluate(step, _ctx())
        assert result.confidence > 0.5

    async def test_no_state_changes_no_result_low_progress(
        self, probe: ConfidenceProbe
    ) -> None:
        """No state changes, no result → lowest progress score (0.3)."""
        step = _step(reasoning="")
        result = await probe.evaluate(step, _ctx())
        assert result.confidence < 0.6

    # ------------------------------------------------------------------
    # Hard / soft / pass thresholds
    # ------------------------------------------------------------------

    async def test_hard_fail_when_confidence_very_low(self) -> None:
        """Very low confidence (< 0.3) → hard_fail."""
        probe = ConfidenceProbe(hard_threshold=0.9, soft_threshold=0.95)
        step = _step(reasoning="")
        result = await probe.evaluate(step, _ctx())
        assert result.verdict == "hard_fail"

    async def test_soft_fail_mid_confidence(self) -> None:
        """Confidence between hard and soft threshold → soft_fail."""
        probe = ConfidenceProbe(hard_threshold=0.1, soft_threshold=0.9)
        step = _step(reasoning="therefore", state_changes=["k"])
        result = await probe.evaluate(step, _ctx())
        assert result.verdict == "soft_fail"

    async def test_passed_above_soft_threshold(self, probe: ConfidenceProbe) -> None:
        """Confidence above soft threshold → passed."""
        step = _step(
            reasoning="therefore the analysis confirmed because the result is verified",
            state_changes=["key"],
            tool_calls=[ToolCall(tool_name="t", success=True)],
        )
        result = await probe.evaluate(step, _ctx())
        assert result.verdict == "passed"

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    async def test_completion_keyword_in_result_not_convergence(
        self, probe: ConfidenceProbe
    ) -> None:
        """Completion keyword in result dict must NOT signal convergence.

        Internal status fields like ``status: completed`` are unrelated
        to goal completion.
        """
        step = _step(result={"status": "completed successfully"})
        result = await probe.evaluate(step, _ctx())
        assert result.should_converge is False

    async def test_completion_keyword_in_reasoning(self, probe: ConfidenceProbe) -> None:
        """Completion keyword in reasoning signals convergence."""
        step = _step(reasoning="The task is now completed and the file has been written.")
        result = await probe.evaluate(step, _ctx())
        assert result.should_converge is True

    async def test_completion_keyword_in_action(self, probe: ConfidenceProbe) -> None:
        """Completion keyword in action also signals convergence."""
        step = _step(action="Task is done and finished")
        result = await probe.evaluate(step, _ctx())
        assert result.should_converge is True

    async def test_no_completion_keyword(self, probe: ConfidenceProbe) -> None:
        """No completion keyword → should_converge is False."""
        step = _step(action="Still working", result={"progress": 50})
        result = await probe.evaluate(step, _ctx())
        assert result.should_converge is False

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_partial_tool_failure(self, probe: ConfidenceProbe) -> None:
        """50% tool success → mid tool score."""
        tools = [
            ToolCall(tool_name="t1", success=True),
            ToolCall(tool_name="t2", success=False, error="e"),
        ]
        step = _step(tool_calls=tools)
        result = await probe.evaluate(step, _ctx())
        # tool_score=0.5; mid confidence expected
        assert 0.0 < result.confidence < 1.0
