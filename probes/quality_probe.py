"""Quality Probe system for evaluating agent outputs.

QualityProbe determines whether the reconcile loop should continue,
converge (goal achieved), or fail (unrecoverable error).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from core.state.models import ConvergenceCriterion

if TYPE_CHECKING:
    from core.state.models import LoopContext, StepOutput


class ProbeResult(BaseModel):
    """Result of a quality probe evaluation.

    Attributes:
        verdict: The probe's verdict on the step output.
            - passed: Step output is acceptable, continue or converge
            - soft_fail: Minor issues, may continue with caution
            - hard_fail: Unacceptable output, must stop
        confidence: Confidence score from 0.0 to 1.0.
        reason: Human-readable explanation of the verdict.
        should_converge: True if the goal has been achieved.
        suggestions: Optional suggestions for improvement.
    """

    verdict: Literal["passed", "soft_fail", "hard_fail"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    should_converge: bool = False
    suggestions: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reason": self.reason,
            "should_converge": self.should_converge,
            "suggestions": self.suggestions,
        }


class QualityProbe(ABC):
    """Abstract base class for quality probes.

    Quality probes evaluate each step of the reconcile loop to determine:
    1. Whether the output is acceptable
    2. Whether the goal has been achieved (should_converge)
    3. Whether to request human intervention

    Implementations can use various strategies:
    - LLM-based evaluation
    - Rule-based checks
    - Output validation
    - Goal matching
    """

    @property
    def name(self) -> str:
        """Name of this probe."""
        return self.__class__.__name__

    @abstractmethod
    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate the quality of a step output.

        Args:
            step_output: The output from the current reconcile step.
            context: Full context of the reconcile loop.

        Returns:
            ProbeResult with verdict and convergence decision.
        """
        ...


class DefaultQualityProbe(QualityProbe):
    """Default quality probe implementation.

    Uses simple heuristics for evaluation:
    - Checks for tool execution errors
    - Evaluates based on step result content
    - Looks for goal completion indicators

    For production use, consider implementing an LLM-based probe.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        convergence_keywords: list[str] | None = None,
    ) -> None:
        """Initialize the default quality probe.

        Args:
            max_consecutive_failures: Max failed steps before hard_fail.
            convergence_keywords: Keywords indicating goal completion.
        """
        self._max_consecutive_failures = max_consecutive_failures
        self._convergence_keywords = convergence_keywords or [
            "completed",
            "done",
            "finished",
            "success",
            "achieved",
            "goal reached",
        ]

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate step output quality."""
        failed_tools = [tc for tc in step_output.tool_calls if not tc.success]
        if failed_tools:
            consecutive_failures = self._count_consecutive_failures(context)
            if consecutive_failures >= self._max_consecutive_failures:
                return ProbeResult(
                    verdict="hard_fail",
                    confidence=0.9,
                    reason=f"Too many consecutive tool failures ({consecutive_failures})",
                    should_converge=False,
                )
            return ProbeResult(
                verdict="soft_fail",
                confidence=0.6,
                reason=f"Tool execution failed: {failed_tools[0].error}",
                should_converge=False,
                suggestions=["Retry with different parameters", "Try alternative approach"],
            )

        if self._check_convergence(step_output, context):
            return ProbeResult(
                verdict="passed",
                confidence=0.85,
                reason="Goal appears to be achieved",
                should_converge=True,
            )

        return ProbeResult(
            verdict="passed",
            confidence=0.7,
            reason="Step completed successfully, continuing toward goal",
            should_converge=False,
        )

    def _count_consecutive_failures(self, context: LoopContext) -> int:
        """Count consecutive failed steps from history."""
        count = 0
        for step in reversed(context.history):
            has_failure = any(not tc.success for tc in step.tool_calls)
            if has_failure:
                count += 1
            else:
                break
        return count

    def _check_convergence(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> bool:
        """Check if the goal appears to be achieved."""
        result_str = str(step_output.result).lower() if step_output.result else ""
        action_str = step_output.action.lower()

        for keyword in self._convergence_keywords:
            if keyword in result_str or keyword in action_str:
                return True

        return False


class CompositeQualityProbe(QualityProbe):
    """Combines multiple probes with configurable aggregation.

    Useful for running multiple evaluation strategies together.
    """

    def __init__(
        self,
        probes: list[QualityProbe],
        strategy: Literal["all_pass", "any_pass", "majority"] = "all_pass",
    ) -> None:
        """Initialize composite probe.

        Args:
            probes: List of probes to evaluate.
            strategy: How to aggregate results.
                - all_pass: All probes must pass
                - any_pass: At least one probe must pass
                - majority: More than half must pass
        """
        self._probes = probes
        self._strategy = strategy

    @property
    def name(self) -> str:
        return f"CompositeProbe({self._strategy})"

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate using all probes and aggregate results."""
        results = []
        for probe in self._probes:
            result = await probe.evaluate(step_output, context)
            results.append(result)

        passed = [r for r in results if r.verdict == "passed"]
        hard_fails = [r for r in results if r.verdict == "hard_fail"]

        if hard_fails:
            first_fail = hard_fails[0]
            return ProbeResult(
                verdict="hard_fail",
                confidence=first_fail.confidence,
                reason=f"Hard fail from {len(hard_fails)} probe(s): {first_fail.reason}",
                should_converge=False,
            )

        if self._strategy == "all_pass":
            all_passed = len(passed) == len(results)
        elif self._strategy == "any_pass":
            all_passed = len(passed) > 0
        else:
            all_passed = len(passed) > len(results) / 2

        if all_passed:
            any_converge = any(r.should_converge for r in results)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            return ProbeResult(
                verdict="passed",
                confidence=avg_confidence,
                reason=f"{len(passed)}/{len(results)} probes passed",
                should_converge=any_converge,
            )

        soft_fails = [r for r in results if r.verdict == "soft_fail"]
        if soft_fails:
            first_soft = soft_fails[0]
            return ProbeResult(
                verdict="soft_fail",
                confidence=first_soft.confidence,
                reason=f"Soft fail from {len(soft_fails)} probe(s): {first_soft.reason}",
                should_converge=False,
            )

        return ProbeResult(
            verdict="passed",
            confidence=0.5,
            reason="Composite evaluation complete",
            should_converge=False,
        )


class ConfidenceThresholdProbe(QualityProbe):
    """Probe that requires minimum confidence to pass.

    Wraps another probe and enforces a confidence threshold.
    """

    def __init__(
        self,
        inner_probe: QualityProbe,
        threshold: float = 0.7,
    ) -> None:
        """Initialize threshold probe.

        Args:
            inner_probe: The probe to wrap.
            threshold: Minimum confidence required to pass.
        """
        self._inner = inner_probe
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"ConfidenceThreshold({self._inner.name}, {self._threshold})"

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate and apply confidence threshold."""
        result = await self._inner.evaluate(step_output, context)

        if result.verdict == "passed" and result.confidence < self._threshold:
            return ProbeResult(
                verdict="soft_fail",
                confidence=result.confidence,
                reason=f"Confidence {result.confidence:.2f} below threshold {self._threshold}",
                should_converge=False,
                suggestions=["Consider human review due to low confidence"],
            )

        return result


class ConvergenceCriteriaProbe(QualityProbe):
    """Checks structured ConvergenceCriterion declarations from DesiredState.

    Iterates desired_state.convergence_criteria and checks each one.
    All criteria must pass for should_converge=True.

    Supported criterion_type values:
        - "file_exists": checks if params["path"] file exists on disk
        - "all_tests_pass": runs params.get("test_command", "pytest") as subprocess
        - "lint_clean": runs params.get("lint_command", "ruff check .") as subprocess
        - "custom_probe": always passes (checked externally)

    Args:
        run_shell_commands: Whether to actually run shell commands for
            all_tests_pass/lint_clean checks (default False for safety in tests).
    """

    def __init__(self, run_shell_commands: bool = False) -> None:
        self._run_shell_commands = run_shell_commands

    @property
    def name(self) -> str:
        """Name of this probe."""
        return "ConvergenceCriteriaProbe"

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate convergence criteria from DesiredState.

        Args:
            step_output: The output from the current reconcile step.
            context: Full context of the reconcile loop.

        Returns:
            ProbeResult indicating whether all criteria are satisfied.
        """
        criteria = context.desired_state.convergence_criteria
        if not criteria:
            # No criteria declared -> always pass, never auto-converge
            return ProbeResult(
                verdict="passed",
                confidence=0.8,
                reason="No convergence criteria declared",
                should_converge=False,
            )

        failed_criteria: list[str] = []
        for criterion in criteria:
            passed = await self._check_criterion(criterion)
            if not passed:
                failed_criteria.append(criterion.description or criterion.criterion_type)

        if not failed_criteria:
            return ProbeResult(
                verdict="passed",
                confidence=0.95,
                reason=f"All {len(criteria)} convergence criteria satisfied",
                should_converge=True,
            )

        return ProbeResult(
            verdict="soft_fail",
            confidence=0.5,
            reason=f"Criteria not yet met: {', '.join(failed_criteria)}",
            should_converge=False,
            suggestions=[f"Work on: {c}" for c in failed_criteria],
        )

    async def _check_criterion(self, criterion: ConvergenceCriterion) -> bool:
        """Check a single criterion.

        Args:
            criterion: The criterion to check.

        Returns:
            True if the criterion is satisfied.
        """
        import os

        if criterion.criterion_type == "file_exists":
            path = criterion.params.get("path", "")
            return bool(path) and os.path.exists(path)

        if criterion.criterion_type == "custom_probe":
            return True

        if not self._run_shell_commands:
            return False  # Conservative: report not-yet-met without running

        if criterion.criterion_type == "all_tests_pass":
            cmd = criterion.params.get("test_command", "pytest")
            return await self._run_command(cmd)

        if criterion.criterion_type == "lint_clean":
            cmd = criterion.params.get("lint_command", "ruff check .")
            return await self._run_command(cmd)

        return False

    async def _run_command(self, cmd: str) -> bool:
        """Run a shell command, return True if exit code 0.

        Args:
            cmd: Shell command to execute.

        Returns:
            True if the command exited with code 0.
        """
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
