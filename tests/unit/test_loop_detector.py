"""Unit tests for LoopDetectorProbe."""

from __future__ import annotations

import pytest

from core.state.models import DesiredState, LoopContext, StepOutput, ToolCall
from probes.loop_detector import LoopDetectorProbe


def _ctx(history: list[StepOutput]) -> LoopContext:
    return LoopContext(
        desired_state=DesiredState(goal="test"),
        history=history,
        agent_id="test-agent",
    )


def _step(action: str, reasoning: str = "", step_number: int = 1) -> StepOutput:
    return StepOutput(step_number=step_number, action=action, reasoning=reasoning)


def _step_with_tool(tool_name: str, step_number: int = 1) -> StepOutput:
    return StepOutput(
        step_number=step_number,
        action=f"call {tool_name}",
        reasoning="using tool",
        tool_calls=[ToolCall(tool_name=tool_name, params={"x": 1}, success=True)],
    )


class TestLoopDetectorProbe:
    """Tests for LoopDetectorProbe."""

    @pytest.fixture()
    def probe(self) -> LoopDetectorProbe:
        return LoopDetectorProbe(window_size=10, repeat_threshold=3)

    # ------------------------------------------------------------------
    # Passing cases
    # ------------------------------------------------------------------

    async def test_unique_actions_pass(self, probe: LoopDetectorProbe) -> None:
        """Unique actions in history should pass."""
        history = [_step(f"action_{i}", step_number=i + 1) for i in range(5)]
        current = _step("action_new", step_number=6)
        result = await probe.evaluate(current, _ctx(history))
        assert result.verdict == "passed"
        assert result.confidence == 1.0

    async def test_below_threshold_passes(self, probe: LoopDetectorProbe) -> None:
        """Repeated action below threshold should pass."""
        repeated = _step("do the same thing", "because", step_number=1)
        history = [repeated, repeated]  # 2 occurrences in history
        current = repeated  # 3rd total but threshold is 3 — equal to threshold
        # Window = history[-10:] + [current] = 3 items → count == 3 → hard_fail
        result = await probe.evaluate(current, _ctx(history))
        assert result.verdict == "hard_fail"  # exactly at threshold

    async def test_one_repeat_passes(self, probe: LoopDetectorProbe) -> None:
        """Two occurrences (below threshold of 3) should pass."""
        repeated = _step("do the same thing", "because")
        history = [repeated]  # 1 in history
        current = repeated   # 2 total → below threshold
        result = await probe.evaluate(current, _ctx(history))
        assert result.verdict == "passed"

    async def test_empty_history_passes(self, probe: LoopDetectorProbe) -> None:
        """First step with no history should always pass."""
        result = await probe.evaluate(_step("first action"), _ctx([]))
        assert result.verdict == "passed"

    # ------------------------------------------------------------------
    # Loop detection — action fingerprint
    # ------------------------------------------------------------------

    async def test_repeated_action_triggers_hard_fail(
        self, probe: LoopDetectorProbe
    ) -> None:
        """Three identical action+reasoning combos triggers hard_fail."""
        repeated = _step("fetch data", "need to fetch", step_number=1)
        history = [repeated, repeated]  # 2 in history
        current = repeated  # 3rd → triggers
        result = await probe.evaluate(current, _ctx(history))
        assert result.verdict == "hard_fail"
        assert "loop_pattern" in result.reason.lower() or "loop" in result.reason.lower()
        assert not result.should_converge

    async def test_loop_reason_contains_pattern(
        self, probe: LoopDetectorProbe
    ) -> None:
        """The hard_fail reason should reference the loop pattern."""
        step = _step("A" * 120, "r")
        history = [step, step]
        result = await probe.evaluate(step, _ctx(history))
        assert result.verdict == "hard_fail"
        assert "A" in result.reason  # truncated pattern included

    # ------------------------------------------------------------------
    # Loop detection — tool-call fingerprint
    # ------------------------------------------------------------------

    async def test_repeated_tool_calls_triggers_hard_fail(
        self, probe: LoopDetectorProbe
    ) -> None:
        """Identical tool-call sequences trigger hard_fail."""
        step = _step_with_tool("search")
        history = [step, step]
        result = await probe.evaluate(step, _ctx(history))
        assert result.verdict == "hard_fail"

    async def test_different_tool_params_dont_trigger(
        self, probe: LoopDetectorProbe
    ) -> None:
        """Different tool params should have different fingerprints."""
        def step_with_param(val: int) -> StepOutput:
            return StepOutput(
                step_number=val,
                action=f"search {val}",
                reasoning="different",
                tool_calls=[ToolCall(tool_name="search", params={"q": str(val)}, success=True)],
            )

        history = [step_with_param(i + 1) for i in range(5)]
        current = step_with_param(99)
        result = await probe.evaluate(current, _ctx(history))
        assert result.verdict == "passed"

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    async def test_window_limits_history(self) -> None:
        """Loop outside the window should not trigger detection."""
        probe = LoopDetectorProbe(window_size=5, repeat_threshold=3)
        repeated = _step("old action", "old reason")
        # 10 repetitions but all outside the window of 5
        old_history = [repeated] * 10
        unique_history = [_step(f"unique_{i}") for i in range(5)]
        current = repeated  # window = unique_history[-5:] + [current] = 6 items, only 1 repeat
        result = await probe.evaluate(current, _ctx(old_history + unique_history))
        # Window is last 5 from history (all unique) + current (repeated) = 1 occurrence
        assert result.verdict == "passed"
