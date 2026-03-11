"""Loop Detector Probe — detects reasoning loops in the reconcile cycle.

Uses a sliding window of action fingerprints to identify repetitive patterns
that indicate the agent is stuck. Inherits from QualityProbe.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from probes.quality_probe import ProbeResult, QualityProbe

if TYPE_CHECKING:
    from core.state.models import LoopContext, StepOutput


def _fingerprint(action: str, reasoning: str) -> str:
    """Compute a short fingerprint for an action+reasoning pair.

    Args:
        action: The action taken in the step.
        reasoning: The reasoning behind the action.

    Returns:
        Hex string (first 16 bytes of sha256).
    """
    raw = (action + reasoning).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _tool_calls_fingerprint(step_output: StepOutput) -> str:
    """Compute fingerprint for the sequence of tool calls in a step.

    Args:
        step_output: The step output to fingerprint.

    Returns:
        Hex string fingerprint.
    """
    tc_key = json.dumps(
        [
            {"name": tc.tool_name, "params": sorted(tc.params.items()) if tc.params else []}
            for tc in step_output.tool_calls
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(tc_key.encode("utf-8")).hexdigest()[:32]


class LoopDetectorProbe(QualityProbe):
    """Detects reasoning loops using a sliding window of action fingerprints.

    Two detection strategies run in parallel:
    1. Action+reasoning fingerprint repetition within the window.
    2. Tool-call sequence repetition within the window.

    If either strategy detects ``repeat_threshold`` or more occurrences of the
    same fingerprint within ``window_size`` recent steps, returns hard_fail.

    Args:
        window_size: Number of recent steps to examine (default 10).
        repeat_threshold: Occurrences of the same fingerprint to trigger detection
            (default 3).
    """

    def __init__(
        self,
        window_size: int = 10,
        repeat_threshold: int = 3,
    ) -> None:
        self._window_size = window_size
        self._repeat_threshold = repeat_threshold

    @property
    def name(self) -> str:
        return f"LoopDetectorProbe(w={self._window_size},t={self._repeat_threshold})"

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate whether the current step is part of a repetitive loop.

        Args:
            step_output: The output from the current reconcile step.
            context: Full context of the reconcile loop.

        Returns:
            ProbeResult with hard_fail if a loop is detected, passed otherwise.
        """
        # Build window: history (already executed) + current step
        window_steps = list(context.history[-self._window_size :]) + [step_output]

        # Strategy 1: action+reasoning fingerprint
        action_fps: list[str] = [
            _fingerprint(s.action, s.reasoning) for s in window_steps
        ]
        current_action_fp = action_fps[-1]
        action_count = action_fps.count(current_action_fp)

        if action_count >= self._repeat_threshold:
            pattern = step_output.action[:100]
            return ProbeResult(
                verdict="hard_fail",
                confidence=0.95,
                reason=(
                    f"Loop detected: action fingerprint repeated {action_count}x "
                    f"in last {len(window_steps)} steps. "
                    f"loop_pattern='{pattern}'"
                ),
                should_converge=False,
            )

        # Strategy 2: tool-call sequence fingerprint
        if step_output.tool_calls:
            tool_fps: list[str] = [
                _tool_calls_fingerprint(s)
                for s in window_steps
                if s.tool_calls
            ]
            if tool_fps:
                current_tool_fp = _tool_calls_fingerprint(step_output)
                tool_count = tool_fps.count(current_tool_fp)

                if tool_count >= self._repeat_threshold:
                    tool_names = [tc.tool_name for tc in step_output.tool_calls]
                    return ProbeResult(
                        verdict="hard_fail",
                        confidence=0.95,
                        reason=(
                            f"Loop detected: tool-call sequence {tool_names} repeated "
                            f"{tool_count}x in last {len(window_steps)} steps."
                        ),
                        should_converge=False,
                    )

        return ProbeResult(
            verdict="passed",
            confidence=1.0,
            reason="No loop pattern detected.",
            should_converge=False,
        )
