"""Confidence Probe — heuristic-based confidence scoring for step outputs.

Evaluates step quality based on three weighted signals:
- Tool success rate  (weight 0.4)
- Reasoning quality  (weight 0.3)
- Step progress      (weight 0.3)

No LLM calls are made in Phase 3; all evaluation is rule-based.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from probes.quality_probe import ProbeResult, QualityProbe

if TYPE_CHECKING:
    from core.state.models import LoopContext, StepOutput


# Keywords that signal a high-quality reasoning output
_REASONING_KEYWORDS = frozenset(
    [
        "therefore",
        "because",
        "result",
        "conclude",
        "analysis",
        "determined",
        "identified",
        "found",
        "verified",
        "confirmed",
    ]
)

# Keywords in the step result that indicate goal completion
_COMPLETION_KEYWORDS = frozenset(
    [
        "completed",
        "done",
        "finished",
        "success",
        "achieved",
        "goal reached",
        "resolved",
        "accomplished",
    ]
)


class ConfidenceProbe(QualityProbe):
    """Heuristic confidence probe that does NOT call an LLM.

    Weighted signals:
    - **tool_success_rate** (0.4): ratio of successful tool calls.
    - **reasoning_quality** (0.3): reasoning length + keyword presence.
    - **step_progress** (0.3): whether new state changes were recorded.

    Thresholds:
    - confidence < hard_threshold → hard_fail
    - confidence < soft_threshold → soft_fail
    - confidence >= soft_threshold → passed

    Convergence is signalled when the step result contains completion keywords.

    Args:
        hard_threshold: Confidence below which hard_fail is returned (default 0.3).
        soft_threshold: Confidence below which soft_fail is returned (default 0.6).
        min_reasoning_length: Minimum reasoning characters to score full points (default 50).
    """

    def __init__(
        self,
        hard_threshold: float = 0.3,
        soft_threshold: float = 0.6,
        min_reasoning_length: int = 50,
    ) -> None:
        self._hard_threshold = hard_threshold
        self._soft_threshold = soft_threshold
        self._min_reasoning_length = min_reasoning_length

    @property
    def name(self) -> str:
        return (
            f"ConfidenceProbe(hard={self._hard_threshold},soft={self._soft_threshold})"
        )

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate confidence from heuristic signals.

        Args:
            step_output: The output from the current reconcile step.
            context: Full context of the reconcile loop.

        Returns:
            ProbeResult with heuristic confidence score.
        """
        tool_score = self._score_tool_success(step_output)
        reasoning_score = self._score_reasoning(step_output)
        progress_score = self._score_progress(step_output)

        confidence = (
            tool_score * 0.4
            + reasoning_score * 0.3
            + progress_score * 0.3
        )
        confidence = max(0.0, min(1.0, confidence))

        should_converge = self._check_convergence(step_output)

        if confidence < self._hard_threshold:
            return ProbeResult(
                verdict="hard_fail",
                confidence=confidence,
                reason=(
                    f"Confidence {confidence:.2f} below hard threshold {self._hard_threshold}. "
                    f"tool={tool_score:.2f} reasoning={reasoning_score:.2f} "
                    f"progress={progress_score:.2f}"
                ),
                should_converge=False,
            )

        if confidence < self._soft_threshold:
            return ProbeResult(
                verdict="soft_fail",
                confidence=confidence,
                reason=(
                    f"Confidence {confidence:.2f} below soft threshold {self._soft_threshold}. "
                    f"tool={tool_score:.2f} reasoning={reasoning_score:.2f} "
                    f"progress={progress_score:.2f}"
                ),
                should_converge=should_converge,
                suggestions=["Consider human review or alternative approach"],
            )

        return ProbeResult(
            verdict="passed",
            confidence=confidence,
            reason=(
                f"Confidence {confidence:.2f} acceptable. "
                f"tool={tool_score:.2f} reasoning={reasoning_score:.2f} "
                f"progress={progress_score:.2f}"
            ),
            should_converge=should_converge,
        )

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _score_tool_success(self, step_output: StepOutput) -> float:
        """Score based on tool call success rate.

        Returns 1.0 if no tools were called (nothing to fail).
        """
        if not step_output.tool_calls:
            return 1.0
        total = len(step_output.tool_calls)
        succeeded = sum(1 for tc in step_output.tool_calls if tc.success)
        return succeeded / total

    def _score_reasoning(self, step_output: StepOutput) -> float:
        """Score based on reasoning length and keyword presence."""
        reasoning = step_output.reasoning or ""

        # Length score: 0.0 if empty, 1.0 if >= min_reasoning_length
        length_score = min(1.0, len(reasoning) / max(1, self._min_reasoning_length))

        # Keyword score: fraction of known quality keywords present
        reasoning_lower = reasoning.lower()
        hits = sum(1 for kw in _REASONING_KEYWORDS if kw in reasoning_lower)
        keyword_score = min(1.0, hits / 3)  # saturates at 3 keywords

        return (length_score * 0.6 + keyword_score * 0.4)

    def _score_progress(self, step_output: StepOutput) -> float:
        """Score based on whether state changes were recorded."""
        if step_output.state_changes:
            return 1.0
        # Check result for any content indicating progress
        if step_output.result is not None and step_output.result != {}:
            return 0.7
        return 0.3

    def _check_convergence(self, step_output: StepOutput) -> bool:
        """Return True when the step result contains completion signal keywords."""
        result_str = str(step_output.result).lower() if step_output.result else ""
        action_str = step_output.action.lower()
        combined = result_str + " " + action_str
        return any(kw in combined for kw in _COMPLETION_KEYWORDS)
