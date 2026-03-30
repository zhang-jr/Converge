"""Unit tests for LLMQualityProbe."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from core.state.models import DesiredState, LoopContext, StepOutput, ToolCall
from probes.llm_quality_probe import LLMQualityProbe
from probes.quality_probe import DefaultQualityProbe, ProbeResult, QualityProbe


def _ctx(goal: str = "Read the file and report its line count") -> LoopContext:
    return LoopContext(
        desired_state=DesiredState(goal=goal),
        current_step=1,
        history=[],
        state_snapshot={},
        agent_id="test-agent",
        trace_id="test-trace",
    )


def _step(
    action: str = "Calling tools: bash",
    reasoning: str = "Running wc -l to count lines.",
    tool_calls: list[ToolCall] | None = None,
    result: object = None,
    step_number: int = 1,
) -> StepOutput:
    return StepOutput(
        step_number=step_number,
        action=action,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
        result=result,
    )


class TestLLMQualityProbe:
    """Tests for LLMQualityProbe (mock LLM — litellm not imported)."""

    @pytest.fixture()
    def probe(self) -> LLMQualityProbe:
        return LLMQualityProbe()

    async def test_converges_when_llm_says_achieved(self, probe: LLMQualityProbe) -> None:
        """Mock LLM returns goal_achieved=True → should_converge=True."""
        step = _step(
            tool_calls=[ToolCall(tool_name="bash", success=True, result="131 file.py")],
            result={"status": "completed", "tool_results": [{"tool": "bash", "success": True}]},
        )
        result = await probe.evaluate(step, _ctx())
        assert result.should_converge is True
        assert result.confidence >= 0.8

    async def test_continues_when_not_achieved(self, probe: LLMQualityProbe) -> None:
        """When LLM says goal not achieved → should_converge=False."""
        mock_response = json.dumps({
            "goal_achieved": False,
            "confidence": 0.4,
            "reasoning": "The file was read but line count was not reported.",
        })
        with patch.object(probe, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            step = _step(
                tool_calls=[ToolCall(tool_name="read_file", success=True, result="contents")],
                result={"status": "completed"},
            )
            result = await probe.evaluate(step, _ctx())
        assert result.should_converge is False

    async def test_fallback_on_llm_failure(self, probe: LLMQualityProbe) -> None:
        """When _call_llm raises, delegates to fallback probe."""
        with patch.object(probe, "_call_llm", new_callable=AsyncMock, side_effect=RuntimeError("timeout")):
            step = _step(
                tool_calls=[ToolCall(tool_name="bash", success=True)],
                result={"status": "completed"},
            )
            result = await probe.evaluate(step, _ctx())
        # Fallback is DefaultQualityProbe — should return a valid ProbeResult
        assert result.verdict in ("passed", "soft_fail", "hard_fail")

    async def test_tool_failure_short_circuits(self, probe: LLMQualityProbe) -> None:
        """When tools fail, delegates to fallback without calling evaluation LLM."""
        step = _step(
            tool_calls=[ToolCall(tool_name="bash", success=False, error="command not found")],
            result={"status": "partial_failure"},
        )
        # Patch _call_llm to track whether it's called
        with patch.object(probe, "_call_llm", new_callable=AsyncMock) as mock_llm:
            result = await probe.evaluate(step, _ctx())
        mock_llm.assert_not_called()
        assert result.verdict == "soft_fail"

    async def test_thinking_error_short_circuits(self, probe: LLMQualityProbe) -> None:
        """When LLM call itself failed (action starts with Error, no tools),
        delegates to fallback."""
        step = _step(
            action="Error: connection refused",
            reasoning="Exception: ConnectionError",
            tool_calls=[],
        )
        with patch.object(probe, "_call_llm", new_callable=AsyncMock) as mock_llm:
            result = await probe.evaluate(step, _ctx())
        mock_llm.assert_not_called()

    async def test_handles_malformed_json(self, probe: LLMQualityProbe) -> None:
        """When LLM returns non-JSON, falls back gracefully."""
        with patch.object(
            probe, "_call_llm",
            new_callable=AsyncMock,
            return_value="I'm not sure if the goal is achieved.",
        ):
            step = _step(
                tool_calls=[ToolCall(tool_name="bash", success=True)],
                result={"status": "completed"},
            )
            result = await probe.evaluate(step, _ctx())
        # _parse_json returns {} → goal_achieved=False, confidence=0.7
        assert result.should_converge is False

    async def test_custom_fallback_probe(self) -> None:
        """Custom fallback probe is used when LLM fails."""

        class AlwaysConvergeProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="passed",
                    confidence=0.99,
                    reason="Always converge",
                    should_converge=True,
                )

        probe = LLMQualityProbe(fallback_probe=AlwaysConvergeProbe())
        with patch.object(probe, "_call_llm", new_callable=AsyncMock, side_effect=RuntimeError):
            step = _step(
                tool_calls=[ToolCall(tool_name="bash", success=True)],
                result={"status": "completed"},
            )
            result = await probe.evaluate(step, _ctx())
        assert result.should_converge is True
        assert result.confidence == 0.99

    async def test_probe_name(self, probe: LLMQualityProbe) -> None:
        assert probe.name == "LLMQualityProbe"
