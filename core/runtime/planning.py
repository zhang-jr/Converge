"""LLM-powered goal decomposition and adaptive replanning.

LLMPlanner decomposes vague user goals into structured ExecutionPlans
via dedicated LLM calls, separate from the execution-phase prompt.

Key responsibilities:
- Goal clarity detection (should_clarify)
- Goal → ExecutionPlan decomposition (plan)
- Mid-execution replanning when conditions change (replan)

Integration:
    AgentReconcileLoop._planning_phase() delegates to LLMPlanner.
    AgentReconcileLoop._reflect_step() may trigger replan().
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from core.state.models import (
    DesiredState,
    ExecutionPlan,
    PlannedStep,
    StepOutput,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PLANNING_SYSTEM_PROMPT = """\
You are a task planner. Given a goal and available tools, decompose the goal \
into an ordered sequence of concrete execution steps.

Rules:
1. Each step should be an atomic operation achievable with a single tool call.
2. Steps must have clear dependencies — later steps may depend on earlier results.
3. Each step must have an explicit success criterion (expected_output).
4. Keep the plan between 2 and 10 steps. Prefer fewer steps when possible.
5. Use tool_hint to suggest which tool is most appropriate for each step.

Available tools:
{tools_desc}

Respond with ONLY a JSON object (no markdown fences) matching this schema:
{{
  "reasoning": "why you structured the plan this way",
  "steps": [
    {{
      "goal": "what this step accomplishes",
      "tool_hint": "suggested_tool_name",
      "expected_output": "what success looks like"
    }}
  ]
}}"""

_CLARIFY_SYSTEM_PROMPT = """\
You are a goal clarity assessor. Given a user's goal and available tools, \
decide whether the goal is clear enough to create an actionable plan.

A goal is CLEAR if:
- It specifies what outcome is desired
- It is achievable with the available tools
- It does not require major assumptions about user intent

A goal is VAGUE if:
- It is ambiguous about what the user actually wants
- Multiple incompatible interpretations exist
- Critical information is missing (e.g., which files, what kind of improvement)

Available tools:
{tools_desc}

Respond with ONLY a JSON object (no markdown fences):
{{
  "is_clear": true/false,
  "question": "your clarifying question if is_clear=false, else empty string",
  "reasoning": "why you think the goal is clear or vague"
}}"""

_REPLAN_SYSTEM_PROMPT = """\
You are a task replanner. Given the original plan, completed steps with their \
results, and new information discovered during execution, generate an updated \
plan for the REMAINING work.

Do NOT include already-completed steps. Only plan what still needs to be done.

Available tools:
{tools_desc}

Respond with ONLY a JSON object (no markdown fences):
{{
  "reasoning": "why the plan changed and what the new approach is",
  "steps": [
    {{
      "goal": "what this step accomplishes",
      "tool_hint": "suggested_tool_name",
      "expected_output": "what success looks like"
    }}
  ]
}}"""

_REFLECT_SYSTEM_PROMPT = """\
You are a step evaluator. Given the planned step and the actual execution \
result, assess whether:
1. The step achieved its expected outcome.
2. Any new information was discovered that changes assumptions.
3. The remaining plan should be adjusted.

Respond with ONLY a JSON object (no markdown fences):
{{
  "met_expectation": true/false,
  "reflection": "what happened vs what was expected",
  "new_information": "any discoveries that affect future steps, or empty string",
  "needs_replan": true/false
}}"""


class LLMPlanner:
    """LLM-based goal decomposition into ExecutionPlan.

    Uses dedicated LLM calls (separate from execution-phase calls) to:
    1. Assess goal clarity and request clarification if needed.
    2. Decompose goals into ordered PlannedStep sequences.
    3. Replan when execution reveals new information.

    Args:
        model: LiteLLM model string (e.g. "claude-sonnet-4-6").
            If None, uses framework default from core.config.
        temperature: LLM temperature for planning calls.
        max_tokens: Max tokens for planning LLM responses.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def should_clarify(
        self,
        desired_state: DesiredState,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Check if the goal is too vague and return a clarifying question.

        Args:
            desired_state: The user's declared goal.
            available_tools: List of tool info dicts (name + description).

        Returns:
            A clarifying question string if the goal is vague, else None.
        """
        tools_desc = self._format_tools(available_tools)
        system = _CLARIFY_SYSTEM_PROMPT.format(tools_desc=tools_desc)
        user_msg = f"Goal: {desired_state.goal}"
        if desired_state.constraints:
            user_msg += "\nConstraints:\n" + "\n".join(
                f"- {c}" for c in desired_state.constraints
            )

        response = await self._call_llm(system, user_msg)
        parsed = self._parse_json(response)
        if not parsed.get("is_clear", True):
            return parsed.get("question", "Could you clarify your goal?")
        return None

    async def plan(
        self,
        desired_state: DesiredState,
        available_tools: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        agent_id: str = "",
    ) -> ExecutionPlan:
        """Decompose a goal into an ExecutionPlan.

        Args:
            desired_state: The user's declared goal.
            available_tools: List of tool info dicts (name + description).
            context: Additional context to include in the prompt.
            agent_id: Agent ID to tag the plan with.

        Returns:
            ExecutionPlan with ordered PlannedSteps.
        """
        tools_desc = self._format_tools(available_tools)
        system = _PLANNING_SYSTEM_PROMPT.format(tools_desc=tools_desc)

        user_msg = f"Goal: {desired_state.goal}"
        if desired_state.constraints:
            user_msg += "\nConstraints:\n" + "\n".join(
                f"- {c}" for c in desired_state.constraints
            )
        if context:
            user_msg += f"\nAdditional context:\n{json.dumps(context, default=str)}"

        response = await self._call_llm(system, user_msg)
        parsed = self._parse_json(response)

        steps = [
            PlannedStep(
                step_index=i,
                goal=s.get("goal", ""),
                tool_hint=s.get("tool_hint", ""),
                expected_output=s.get("expected_output", ""),
            )
            for i, s in enumerate(parsed.get("steps", []))
        ]

        return ExecutionPlan(
            agent_id=agent_id,
            goal=desired_state.goal,
            reasoning=parsed.get("reasoning", ""),
            steps=steps,
        )

    async def replan(
        self,
        original_plan: ExecutionPlan,
        completed_steps: list[StepOutput],
        new_information: str,
        available_tools: list[dict[str, Any]] | None = None,
        agent_id: str = "",
    ) -> ExecutionPlan:
        """Generate an updated plan based on execution feedback.

        Args:
            original_plan: The plan being executed.
            completed_steps: Steps already executed with results.
            new_information: What changed or was discovered.
            available_tools: List of tool info dicts.
            agent_id: Agent ID for the new plan.

        Returns:
            New ExecutionPlan covering remaining work.
        """
        tools_desc = self._format_tools(available_tools)
        system = _REPLAN_SYSTEM_PROMPT.format(tools_desc=tools_desc)

        completed_summary = "\n".join(
            f"- Step {s.step_number}: {s.action} → {s.result}"
            for s in completed_steps
        )

        remaining_steps = [
            f"- Step {s.step_index}: {s.goal} (expected: {s.expected_output})"
            for s in original_plan.steps[len(completed_steps):]
        ]

        user_msg = (
            f"Original goal: {original_plan.goal}\n\n"
            f"Completed steps:\n{completed_summary}\n\n"
            f"Remaining planned steps:\n" + "\n".join(remaining_steps) + "\n\n"
            f"New information: {new_information}"
        )

        response = await self._call_llm(system, user_msg)
        parsed = self._parse_json(response)

        # New step indices continue from where completed steps left off
        base_index = len(completed_steps)
        steps = [
            PlannedStep(
                step_index=base_index + i,
                goal=s.get("goal", ""),
                tool_hint=s.get("tool_hint", ""),
                expected_output=s.get("expected_output", ""),
            )
            for i, s in enumerate(parsed.get("steps", []))
        ]

        return ExecutionPlan(
            agent_id=agent_id,
            goal=original_plan.goal,
            reasoning=parsed.get("reasoning", ""),
            steps=steps,
        )

    async def reflect_on_step(
        self,
        step_output: StepOutput,
        planned_step: PlannedStep | None,
    ) -> dict[str, Any]:
        """Evaluate a completed step against its plan.

        Args:
            step_output: Actual execution result.
            planned_step: The corresponding planned step (may be None).

        Returns:
            Dict with keys: met_expectation, reflection, new_information,
            needs_replan.
        """
        if planned_step is None:
            return {
                "met_expectation": True,
                "reflection": "No planned step to compare against.",
                "new_information": "",
                "needs_replan": False,
            }

        user_msg = (
            f"Planned step:\n"
            f"  Goal: {planned_step.goal}\n"
            f"  Expected output: {planned_step.expected_output}\n\n"
            f"Actual execution:\n"
            f"  Action: {step_output.action}\n"
            f"  Reasoning: {step_output.reasoning}\n"
            f"  Result: {step_output.result}"
        )

        response = await self._call_llm(_REFLECT_SYSTEM_PROMPT, user_msg)
        parsed = self._parse_json(response)

        return {
            "met_expectation": parsed.get("met_expectation", True),
            "reflection": parsed.get("reflection", ""),
            "new_information": parsed.get("new_information", ""),
            "needs_replan": parsed.get("needs_replan", False),
        }

    # ------------------------------------------------------------------
    # LLM call (mirrors Agent._call_llm but with planning-specific config)
    # ------------------------------------------------------------------

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Make a planning-specific LLM call.

        Returns the raw text content from the LLM response.
        Falls back to empty JSON object if litellm is unavailable.
        """
        from core.config import (
            LLM_API_BASE,
            LLM_API_KEY,
            LLM_MODEL,
            resolve_model,
        )

        messages = [
            {"role": "system", "content": system_prompt},
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

            t0 = time.monotonic()
            response = await litellm.acompletion(**kwargs)
            _ = (time.monotonic() - t0) * 1000.0  # latency_ms for future tracing

            return response.choices[0].message.content or ""

        except ImportError:
            # litellm not installed — return mock for testing
            return self._mock_response(system_prompt)

    def _mock_response(self, system_prompt: str) -> str:
        """Return a deterministic mock response for testing without LLM."""
        if "clarity assessor" in system_prompt:
            return json.dumps({
                "is_clear": True,
                "question": "",
                "reasoning": "Mock: goal is assumed clear.",
            })
        if "replanner" in system_prompt:
            return json.dumps({
                "reasoning": "Mock: continuing with adjusted plan.",
                "steps": [
                    {
                        "goal": "Continue toward goal",
                        "tool_hint": "",
                        "expected_output": "progress made",
                    }
                ],
            })
        if "step evaluator" in system_prompt:
            return json.dumps({
                "met_expectation": True,
                "reflection": "Mock: step met expectation.",
                "new_information": "",
                "needs_replan": False,
            })
        # Default: planning response
        return json.dumps({
            "reasoning": "Mock: breaking goal into sequential steps.",
            "steps": [
                {
                    "goal": "Analyze the problem",
                    "tool_hint": "read_file",
                    "expected_output": "problem analysis completed",
                },
                {
                    "goal": "Implement solution",
                    "tool_hint": "write_file",
                    "expected_output": "solution implemented",
                },
                {
                    "goal": "Verify result",
                    "tool_hint": "bash",
                    "expected_output": "verification passed",
                },
            ],
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]] | None) -> str:
        """Format tool info list into a readable string for prompts."""
        if not tools:
            return "(no tools specified)"
        lines = []
        for t in tools:
            name = t.get("name", "unknown")
            desc = t.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
