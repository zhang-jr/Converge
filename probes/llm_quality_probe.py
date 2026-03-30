"""LLM-based Quality Probe for semantic goal-completion evaluation.

Uses a dedicated LLM call to evaluate whether the agent's goal has been
achieved, rather than relying on keyword heuristics.  This produces more
accurate convergence decisions at the cost of an extra (small) LLM call
per step.

The probe is opt-in — ``DefaultQualityProbe`` remains the default.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from probes.quality_probe import DefaultQualityProbe, ProbeResult, QualityProbe

if TYPE_CHECKING:
    from core.state.models import LoopContext, StepOutput

_EVALUATION_SYSTEM_PROMPT = """\
You are a goal-completion evaluator.  Given the user's goal and the \
agent's most recent step (action, reasoning, and tool results), determine \
whether the goal has been fully achieved.

Rules:
1. The goal is ACHIEVED if the step results contain the information or \
outcome the user asked for, even if the agent has not explicitly said "done".
2. The goal is NOT ACHIEVED if critical information is missing, tools \
failed, or the task requires further action.
3. Focus on the SUBSTANCE of the results, not on keywords like \
"completed" or "done".

Respond with ONLY a JSON object (no markdown fences):
{"goal_achieved": true/false, "confidence": 0.0 to 1.0, \
"reasoning": "brief explanation"}"""


class LLMQualityProbe(QualityProbe):
    """Quality probe that uses an LLM to judge goal completion.

    Follows the same LLM-call pattern as ``LLMPlanner`` in
    ``core/runtime/planning.py``: resolves model via ``core.config``,
    calls ``litellm.acompletion``, falls back to mock when litellm is
    unavailable.

    Tool failures and LLM-level errors are checked first using the
    same logic as ``DefaultQualityProbe`` to avoid unnecessary
    evaluation calls.

    Args:
        model: LiteLLM model identifier (or provider alias from
            ``llm_providers.json``).  Defaults to the project-level
            ``LLM_MODEL`` from ``.env``.
        temperature: Sampling temperature for evaluation calls.
        max_tokens: Maximum tokens for the evaluation response.
        fallback_probe: Probe to delegate to when the LLM call fails
            (network error, rate limit, malformed response).
            Defaults to ``DefaultQualityProbe()``.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        fallback_probe: QualityProbe | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._fallback = fallback_probe or DefaultQualityProbe()

    @property
    def name(self) -> str:
        return "LLMQualityProbe"

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        """Evaluate step output using an LLM call.

        Short-circuits on tool / LLM-level errors (same logic as
        ``DefaultQualityProbe``) and only invokes the evaluation LLM
        when the step executed without errors.
        """
        # --- Short-circuit: LLM-level error ---
        if DefaultQualityProbe._is_thinking_error(step_output):
            return await self._fallback.evaluate(step_output, context)

        # --- Short-circuit: tool failure ---
        failed_tools = [tc for tc in step_output.tool_calls if not tc.success]
        if failed_tools:
            return await self._fallback.evaluate(step_output, context)

        # --- Semantic evaluation via LLM ---
        try:
            goal_achieved, confidence, reasoning = await self._evaluate_with_llm(
                step_output, context,
            )
        except Exception:
            # LLM call failed — delegate to heuristic fallback
            return await self._fallback.evaluate(step_output, context)

        if goal_achieved:
            return ProbeResult(
                verdict="passed",
                confidence=confidence,
                reason=f"LLM evaluation: {reasoning}",
                should_converge=True,
            )

        return ProbeResult(
            verdict="passed",
            confidence=confidence,
            reason=f"LLM evaluation: {reasoning}",
            should_converge=False,
        )

    # ------------------------------------------------------------------
    # LLM call (mirrors LLMPlanner._call_llm)
    # ------------------------------------------------------------------

    async def _evaluate_with_llm(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> tuple[bool, float, str]:
        """Call the LLM to evaluate goal completion.

        Returns:
            Tuple of (goal_achieved, confidence, reasoning).
        """
        user_msg = self._build_user_message(step_output, context)
        raw = await self._call_llm(user_msg)
        parsed = self._parse_json(raw)

        goal_achieved = bool(parsed.get("goal_achieved", False))
        confidence = float(parsed.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(parsed.get("reasoning", ""))

        return goal_achieved, confidence, reasoning

    def _build_user_message(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> str:
        """Build the user message describing the goal and step results."""
        tool_summary = "None"
        if step_output.tool_calls:
            parts = []
            for tc in step_output.tool_calls:
                result_str = str(tc.result)[:300] if tc.result else "None"
                parts.append(
                    f"  - {tc.tool_name}: success={tc.success}, result={result_str}"
                )
            tool_summary = "\n".join(parts)

        result_str = str(step_output.result)[:500] if step_output.result else "None"

        return (
            f"Goal: {context.desired_state.goal}\n\n"
            f"Step {step_output.step_number}:\n"
            f"  Action: {step_output.action}\n"
            f"  Reasoning: {step_output.reasoning}\n"
            f"  Tool calls:\n{tool_summary}\n"
            f"  Result: {result_str}"
        )

    async def _call_llm(self, user_message: str) -> str:
        """Make an evaluation-specific LLM call.

        Returns the raw text content from the LLM response.
        Falls back to mock response if litellm is unavailable.
        """
        from core.config import (
            LLM_API_BASE,
            LLM_API_KEY,
            LLM_MODEL,
            resolve_model,
        )

        messages = [
            {"role": "system", "content": _EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            import litellm

            if self._model:
                resolved = resolve_model(self._model)
                model = resolved.litellm_model
                api_base = resolved.api_base
                api_key = resolved.api_key
            else:
                model = LLM_MODEL
                api_base = LLM_API_BASE
                api_key = LLM_API_KEY

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key

            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content or ""

        except ImportError:
            return self._mock_response()

    def _mock_response(self) -> str:
        """Return a deterministic mock response for testing without LLM."""
        return json.dumps({
            "goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Mock: goal assumed achieved for testing.",
        })

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
