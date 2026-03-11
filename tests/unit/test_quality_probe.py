"""Unit tests for QualityProbe implementations."""

import pytest

from core.state.models import DesiredState, LoopContext, StepOutput, ToolCall
from probes.quality_probe import (
    CompositeQualityProbe,
    ConfidenceThresholdProbe,
    DefaultQualityProbe,
    ProbeResult,
    QualityProbe,
)


@pytest.fixture
def default_probe():
    """Create a default quality probe."""
    return DefaultQualityProbe()


@pytest.fixture
def basic_context():
    """Create a basic loop context."""
    return LoopContext(
        desired_state=DesiredState(goal="Complete the task"),
        current_step=1,
        history=[],
        state_snapshot={},
        agent_id="test-agent",
        trace_id="test-trace",
    )


class TestProbeResult:
    """Tests for ProbeResult model."""

    def test_create_passed_result(self):
        """Test creating a passed result."""
        result = ProbeResult(
            verdict="passed",
            confidence=0.9,
            reason="All checks passed",
            should_converge=False,
        )
        assert result.verdict == "passed"
        assert result.confidence == 0.9
        assert result.should_converge is False

    def test_create_converged_result(self):
        """Test creating a converged result."""
        result = ProbeResult(
            verdict="passed",
            confidence=0.95,
            reason="Goal achieved",
            should_converge=True,
        )
        assert result.should_converge is True

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ProbeResult(
            verdict="soft_fail",
            confidence=0.5,
            reason="Some issues",
            should_converge=False,
            suggestions=["Try again"],
        )
        d = result.to_dict()
        assert d["verdict"] == "soft_fail"
        assert d["suggestions"] == ["Try again"]


class TestDefaultQualityProbe:
    """Tests for DefaultQualityProbe."""

    async def test_passed_on_successful_step(
        self,
        default_probe: DefaultQualityProbe,
        basic_context: LoopContext,
    ):
        """Test passed verdict on successful step."""
        step = StepOutput(
            step_number=1,
            action="Analyzed the problem",
            result="Analysis complete",
        )

        result = await default_probe.evaluate(step, basic_context)

        assert result.verdict == "passed"
        assert result.confidence > 0

    async def test_soft_fail_on_tool_error(
        self,
        default_probe: DefaultQualityProbe,
        basic_context: LoopContext,
    ):
        """Test soft_fail when tool execution fails."""
        step = StepOutput(
            step_number=1,
            action="Tried to read file",
            tool_calls=[
                ToolCall(
                    tool_name="read_file",
                    params={"path": "/missing"},
                    success=False,
                    error="File not found",
                )
            ],
        )

        result = await default_probe.evaluate(step, basic_context)

        assert result.verdict == "soft_fail"
        assert "File not found" in result.reason

    async def test_hard_fail_on_consecutive_failures(
        self,
        basic_context: LoopContext,
    ):
        """Test hard_fail after too many consecutive failures."""
        probe = DefaultQualityProbe(max_consecutive_failures=2)

        failed_step = StepOutput(
            step_number=1,
            action="Failed action",
            tool_calls=[
                ToolCall(tool_name="test", success=False, error="Error")
            ],
        )

        basic_context.history = [failed_step, failed_step]

        step = StepOutput(
            step_number=3,
            action="Another failure",
            tool_calls=[
                ToolCall(tool_name="test", success=False, error="Error")
            ],
        )

        result = await probe.evaluate(step, basic_context)

        assert result.verdict == "hard_fail"
        assert "consecutive" in result.reason.lower()

    async def test_convergence_detection(
        self,
        default_probe: DefaultQualityProbe,
        basic_context: LoopContext,
    ):
        """Test detection of goal completion."""
        step = StepOutput(
            step_number=1,
            action="Task completed successfully",
            result="All done",
        )

        result = await default_probe.evaluate(step, basic_context)

        assert result.should_converge is True

    async def test_custom_convergence_keywords(
        self,
        basic_context: LoopContext,
    ):
        """Test custom convergence keywords."""
        probe = DefaultQualityProbe(
            convergence_keywords=["mission accomplished"]
        )

        step = StepOutput(
            step_number=1,
            action="Mission accomplished!",
            result="Task result",
        )

        result = await probe.evaluate(step, basic_context)

        assert result.should_converge is True


class TestCompositeQualityProbe:
    """Tests for CompositeQualityProbe."""

    async def test_all_pass_strategy(self, basic_context: LoopContext):
        """Test all_pass strategy requires all probes to pass."""

        class AlwaysPassProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="passed", confidence=0.9, reason="OK"
                )

        class AlwaysFailProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="soft_fail", confidence=0.3, reason="Not OK"
                )

        composite = CompositeQualityProbe(
            [AlwaysPassProbe(), AlwaysFailProbe()],
            strategy="all_pass",
        )

        step = StepOutput(step_number=1, action="Test")
        result = await composite.evaluate(step, basic_context)

        assert result.verdict == "soft_fail"

    async def test_any_pass_strategy(self, basic_context: LoopContext):
        """Test any_pass strategy passes if any probe passes."""

        class AlwaysPassProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="passed", confidence=0.9, reason="OK"
                )

        class AlwaysFailProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="soft_fail", confidence=0.3, reason="Not OK"
                )

        composite = CompositeQualityProbe(
            [AlwaysPassProbe(), AlwaysFailProbe()],
            strategy="any_pass",
        )

        step = StepOutput(step_number=1, action="Test")
        result = await composite.evaluate(step, basic_context)

        assert result.verdict == "passed"

    async def test_hard_fail_propagates(self, basic_context: LoopContext):
        """Test that hard_fail always propagates."""

        class AlwaysPassProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="passed", confidence=0.9, reason="OK"
                )

        class HardFailProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="hard_fail", confidence=0.1, reason="Critical"
                )

        composite = CompositeQualityProbe(
            [AlwaysPassProbe(), HardFailProbe()],
            strategy="any_pass",
        )

        step = StepOutput(step_number=1, action="Test")
        result = await composite.evaluate(step, basic_context)

        assert result.verdict == "hard_fail"


class TestConfidenceThresholdProbe:
    """Tests for ConfidenceThresholdProbe."""

    async def test_pass_above_threshold(self, basic_context: LoopContext):
        """Test pass when confidence above threshold."""

        class HighConfidenceProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="passed", confidence=0.9, reason="OK"
                )

        probe = ConfidenceThresholdProbe(
            HighConfidenceProbe(), threshold=0.7
        )

        step = StepOutput(step_number=1, action="Test")
        result = await probe.evaluate(step, basic_context)

        assert result.verdict == "passed"
        assert result.confidence == 0.9

    async def test_soft_fail_below_threshold(
        self, basic_context: LoopContext
    ):
        """Test soft_fail when confidence below threshold."""

        class LowConfidenceProbe(QualityProbe):
            async def evaluate(self, step, context):
                return ProbeResult(
                    verdict="passed", confidence=0.5, reason="OK"
                )

        probe = ConfidenceThresholdProbe(
            LowConfidenceProbe(), threshold=0.7
        )

        step = StepOutput(step_number=1, action="Test")
        result = await probe.evaluate(step, basic_context)

        assert result.verdict == "soft_fail"
        assert "below threshold" in result.reason.lower()
