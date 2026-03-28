"""Unit tests for Phase 8: LLMPlanner goal decomposition and replanning.

All tests use mock LLM responses (no litellm dependency required).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.runtime.planning import LLMPlanner
from core.state.models import (
    DesiredState,
    ExecutionPlan,
    LoopContext,
    PlannedStep,
    StepOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str) -> str:
    """Helper to create a mock _call_llm return value."""
    return content


def _planning_response(
    steps: list[dict[str, str]],
    reasoning: str = "test reasoning",
) -> str:
    return json.dumps({"reasoning": reasoning, "steps": steps})


def _clarify_response(is_clear: bool, question: str = "") -> str:
    return json.dumps({
        "is_clear": is_clear,
        "question": question,
        "reasoning": "test",
    })


def _reflect_response(
    met: bool = True,
    reflection: str = "ok",
    new_info: str = "",
    needs_replan: bool = False,
) -> str:
    return json.dumps({
        "met_expectation": met,
        "reflection": reflection,
        "new_information": new_info,
        "needs_replan": needs_replan,
    })


# ---------------------------------------------------------------------------
# LLMPlanner.plan() Tests
# ---------------------------------------------------------------------------


class TestLLMPlannerPlan:
    """Tests for LLMPlanner.plan()."""

    async def test_plan_basic_decomposition(self):
        """Plan decomposes goal into PlannedSteps."""
        planner = LLMPlanner()
        response = _planning_response([
            {"goal": "Read the file", "tool_hint": "read_file", "expected_output": "file contents"},
            {"goal": "Analyze contents", "tool_hint": "", "expected_output": "analysis done"},
        ])

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            plan = await planner.plan(
                DesiredState(goal="Analyze this file"),
                available_tools=[{"name": "read_file", "description": "Read a file"}],
            )

        assert isinstance(plan, ExecutionPlan)
        assert plan.goal == "Analyze this file"
        assert len(plan.steps) == 2
        assert plan.steps[0].goal == "Read the file"
        assert plan.steps[0].tool_hint == "read_file"
        assert plan.steps[0].step_index == 0
        assert plan.steps[1].step_index == 1
        assert plan.reasoning == "test reasoning"

    async def test_plan_with_constraints(self):
        """Plan includes constraints in the LLM call."""
        planner = LLMPlanner()
        response = _planning_response([
            {"goal": "Step 1", "tool_hint": "", "expected_output": "done"},
        ])

        call_args: list[tuple] = []

        async def mock_call(system: str, user: str) -> str:
            call_args.append((system, user))
            return response

        planner._call_llm = mock_call  # type: ignore[assignment]

        await planner.plan(
            DesiredState(goal="Do something", constraints=["Be fast", "No errors"]),
        )

        assert len(call_args) == 1
        _, user_msg = call_args[0]
        assert "Be fast" in user_msg
        assert "No errors" in user_msg

    async def test_plan_with_agent_id(self):
        """Plan tags the ExecutionPlan with agent_id."""
        planner = LLMPlanner()
        response = _planning_response([
            {"goal": "Step 1", "tool_hint": "", "expected_output": "done"},
        ])

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            plan = await planner.plan(
                DesiredState(goal="Test"),
                agent_id="agent-42",
            )

        assert plan.agent_id == "agent-42"

    async def test_plan_empty_response(self):
        """Plan handles empty/malformed LLM response gracefully."""
        planner = LLMPlanner()

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value="not json"):
            plan = await planner.plan(DesiredState(goal="Test"))

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 0

    async def test_plan_mock_fallback(self):
        """Plan works with mock LLM when litellm is unavailable."""
        planner = LLMPlanner()

        # Force the ImportError path by patching litellm out
        import sys
        real_litellm = sys.modules.get("litellm")
        sys.modules["litellm"] = None  # type: ignore[assignment]
        try:
            plan = await planner.plan(DesiredState(goal="Build a feature"))
        finally:
            if real_litellm is not None:
                sys.modules["litellm"] = real_litellm
            else:
                sys.modules.pop("litellm", None)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 3  # Mock returns 3 steps
        assert plan.steps[0].goal == "Analyze the problem"


# ---------------------------------------------------------------------------
# LLMPlanner.should_clarify() Tests
# ---------------------------------------------------------------------------


class TestLLMPlannerClarify:
    """Tests for LLMPlanner.should_clarify()."""

    async def test_clear_goal_returns_none(self):
        """Clear goal returns None (no clarification needed)."""
        planner = LLMPlanner()
        response = _clarify_response(is_clear=True)

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            result = await planner.should_clarify(
                DesiredState(goal="Read file.txt and count its lines"),
            )

        assert result is None

    async def test_vague_goal_returns_question(self):
        """Vague goal returns a clarifying question."""
        planner = LLMPlanner()
        response = _clarify_response(
            is_clear=False,
            question="What kind of improvement do you mean?",
        )

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            result = await planner.should_clarify(
                DesiredState(goal="Improve this"),
            )

        assert result == "What kind of improvement do you mean?"

    async def test_clarify_with_constraints(self):
        """Constraints are included in the clarity check prompt."""
        planner = LLMPlanner()
        response = _clarify_response(is_clear=True)

        call_args: list[tuple] = []

        async def mock_call(system: str, user: str) -> str:
            call_args.append((system, user))
            return response

        planner._call_llm = mock_call  # type: ignore[assignment]

        await planner.should_clarify(
            DesiredState(goal="Fix it", constraints=["Only Python files"]),
        )

        _, user_msg = call_args[0]
        assert "Only Python files" in user_msg

    async def test_clarify_malformed_response_defaults_clear(self):
        """Malformed response defaults to clear (no clarification)."""
        planner = LLMPlanner()

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value="garbage"):
            result = await planner.should_clarify(DesiredState(goal="Test"))

        assert result is None  # is_clear defaults to True


# ---------------------------------------------------------------------------
# LLMPlanner.replan() Tests
# ---------------------------------------------------------------------------


class TestLLMPlannerReplan:
    """Tests for LLMPlanner.replan()."""

    async def test_replan_generates_remaining_steps(self):
        """Replan generates new steps for remaining work."""
        planner = LLMPlanner()
        response = _planning_response([
            {"goal": "New approach", "tool_hint": "bash", "expected_output": "fixed"},
        ], reasoning="Original approach failed, switching strategy")

        original = ExecutionPlan(
            goal="Fix the bug",
            steps=[
                PlannedStep(step_index=0, goal="Read logs", expected_output="logs read"),
                PlannedStep(step_index=1, goal="Apply fix", expected_output="fixed"),
            ],
        )
        completed = [
            StepOutput(step_number=1, action="Read logs", result={"logs": "error in auth"}),
        ]

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            new_plan = await planner.replan(
                original_plan=original,
                completed_steps=completed,
                new_information="Bug is in auth module, not where expected",
            )

        assert isinstance(new_plan, ExecutionPlan)
        assert new_plan.goal == "Fix the bug"
        assert len(new_plan.steps) == 1
        # Step index continues from completed count
        assert new_plan.steps[0].step_index == 1
        assert new_plan.steps[0].goal == "New approach"
        assert "switching strategy" in new_plan.reasoning

    async def test_replan_includes_context_in_prompt(self):
        """Replan prompt contains original plan, completed steps, and new info."""
        planner = LLMPlanner()
        response = _planning_response([])

        call_args: list[tuple] = []

        async def mock_call(system: str, user: str) -> str:
            call_args.append((system, user))
            return response

        planner._call_llm = mock_call  # type: ignore[assignment]

        await planner.replan(
            original_plan=ExecutionPlan(
                goal="Original goal",
                steps=[PlannedStep(step_index=0, goal="Step A", expected_output="A done")],
            ),
            completed_steps=[],
            new_information="Something changed",
        )

        _, user_msg = call_args[0]
        assert "Original goal" in user_msg
        assert "Something changed" in user_msg


# ---------------------------------------------------------------------------
# LLMPlanner.reflect_on_step() Tests
# ---------------------------------------------------------------------------


class TestLLMPlannerReflect:
    """Tests for LLMPlanner.reflect_on_step()."""

    async def test_reflect_met_expectation(self):
        """Reflection reports step met expectation."""
        planner = LLMPlanner()
        response = _reflect_response(met=True, reflection="Step completed as planned")

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            result = await planner.reflect_on_step(
                StepOutput(step_number=1, action="Read file", result="contents"),
                PlannedStep(step_index=0, goal="Read file", expected_output="file contents"),
            )

        assert result["met_expectation"] is True
        assert result["needs_replan"] is False

    async def test_reflect_deviation_with_new_info(self):
        """Reflection reports deviation and new information."""
        planner = LLMPlanner()
        response = _reflect_response(
            met=False,
            reflection="File not found",
            new_info="Target file was renamed to data.csv",
            needs_replan=True,
        )

        with patch.object(planner, "_call_llm", new_callable=AsyncMock, return_value=response):
            result = await planner.reflect_on_step(
                StepOutput(step_number=1, action="Read file", result={"error": "not found"}),
                PlannedStep(step_index=0, goal="Read data.txt", expected_output="file contents"),
            )

        assert result["met_expectation"] is False
        assert result["needs_replan"] is True
        assert "renamed" in result["new_information"]

    async def test_reflect_no_planned_step(self):
        """Reflection with no planned step returns default."""
        planner = LLMPlanner()
        result = await planner.reflect_on_step(
            StepOutput(step_number=1, action="Some action"),
            planned_step=None,
        )

        assert result["met_expectation"] is True
        assert result["needs_replan"] is False


# ---------------------------------------------------------------------------
# JSON Parsing Edge Cases
# ---------------------------------------------------------------------------


class TestParseJson:
    """Tests for LLMPlanner._parse_json edge cases."""

    def test_plain_json(self):
        assert LLMPlanner._parse_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        assert LLMPlanner._parse_json(text) == {"a": 1}

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"a": 1}\n```'
        assert LLMPlanner._parse_json(text) == {"a": 1}

    def test_invalid_json_returns_empty(self):
        assert LLMPlanner._parse_json("not json at all") == {}

    def test_empty_string_returns_empty(self):
        assert LLMPlanner._parse_json("") == {}


# ---------------------------------------------------------------------------
# Integration: AgentReconcileLoop with LLMPlanner
# ---------------------------------------------------------------------------


class TestAgentReconcileLoopPlanning:
    """Tests for LLMPlanner integration in AgentReconcileLoop."""

    async def test_planning_strategy_never_skips_planning(self, state_store):
        """planning_strategy='never' skips planning entirely."""
        from core.agent.agent import Agent, AgentReconcileLoop
        from core.state.models import AgentConfig
        from probes.quality_probe import DefaultQualityProbe

        config = AgentConfig(agent_id="test-agent")
        agent = Agent(config=config, state_store=state_store)
        loop = AgentReconcileLoop(
            agent=agent,
            state_store=state_store,
            quality_probe=DefaultQualityProbe(),
        )
        loop._enable_planning = False  # This is set by Agent.run() when strategy="never"

        desired = DesiredState(goal="Simple task", planning_strategy="never")
        # Planning phase should return None
        result = await loop._planning_phase(desired, LoopContext(desired_state=desired))
        assert result is None

    async def test_planning_strategy_always_generates_plan(self, state_store):
        """planning_strategy='always' generates a plan via LLMPlanner."""
        from core.agent.agent import Agent, AgentReconcileLoop
        from core.state.models import AgentConfig
        from probes.quality_probe import DefaultQualityProbe

        config = AgentConfig(agent_id="test-agent")
        agent = Agent(config=config, state_store=state_store)
        loop = AgentReconcileLoop(
            agent=agent,
            state_store=state_store,
            quality_probe=DefaultQualityProbe(),
        )

        plan_response = _planning_response([
            {"goal": "Step A", "tool_hint": "read_file", "expected_output": "done"},
            {"goal": "Step B", "tool_hint": "bash", "expected_output": "verified"},
        ])

        with patch.object(loop._planner, "_call_llm", new_callable=AsyncMock, return_value=plan_response):
            desired = DesiredState(goal="Do something complex", planning_strategy="always")
            plan = await loop._planning_phase(desired, LoopContext(desired_state=desired))

        assert plan is not None
        assert len(plan.steps) == 2

    async def test_planning_strategy_auto_checks_clarity(self, state_store):
        """planning_strategy='auto' checks clarity before planning."""
        from core.agent.agent import Agent, AgentReconcileLoop
        from core.state.models import AgentConfig
        from probes.quality_probe import DefaultQualityProbe

        config = AgentConfig(agent_id="test-agent")
        agent = Agent(config=config, state_store=state_store)
        loop = AgentReconcileLoop(
            agent=agent,
            state_store=state_store,
            quality_probe=DefaultQualityProbe(),
        )

        clarify_resp = _clarify_response(is_clear=True)
        plan_resp = _planning_response([
            {"goal": "Step 1", "tool_hint": "", "expected_output": "done"},
        ])

        call_count = 0

        async def mock_call(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if "clarity assessor" in system:
                return clarify_resp
            return plan_resp

        loop._planner._call_llm = mock_call  # type: ignore[assignment]

        desired = DesiredState(goal="Optimize performance", planning_strategy="auto")
        plan = await loop._planning_phase(desired, LoopContext(desired_state=desired))

        assert plan is not None
        assert call_count == 2  # clarity check + plan generation

    async def test_reflect_step_triggers_replan(self, state_store):
        """Consecutive deviations trigger replan via _reflect_step."""
        from core.agent.agent import Agent, AgentReconcileLoop
        from core.state.models import AgentConfig
        from probes.quality_probe import DefaultQualityProbe

        config = AgentConfig(agent_id="test-agent")
        agent = Agent(config=config, state_store=state_store)
        loop = AgentReconcileLoop(
            agent=agent,
            state_store=state_store,
            quality_probe=DefaultQualityProbe(),
        )
        loop._replan_threshold = 2

        # Set up context with an execution plan
        desired = DesiredState(goal="Test goal")
        context = LoopContext(
            desired_state=desired,
            execution_plan=ExecutionPlan(
                goal="Test goal",
                steps=[
                    PlannedStep(step_index=0, goal="S1", expected_output="done"),
                    PlannedStep(step_index=1, goal="S2", expected_output="done"),
                    PlannedStep(step_index=2, goal="S3", expected_output="done"),
                ],
            ),
            history=[],
        )

        # First deviation
        deviation_resp = _reflect_response(met=False, reflection="deviated")
        replan_resp = _planning_response([
            {"goal": "New S2", "tool_hint": "", "expected_output": "new done"},
        ], reasoning="Replanned due to deviations")

        call_count = 0

        async def mock_call(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if "replanner" in system:
                return replan_resp
            return deviation_resp

        loop._planner._call_llm = mock_call  # type: ignore[assignment]

        step = StepOutput(step_number=1, action="wrong action", result="bad")
        planned = PlannedStep(step_index=0, goal="S1", expected_output="done")

        # First deviation — no replan yet
        reflection1 = await loop._reflect_step(step, planned, context)
        assert "[REPLANNED" not in reflection1
        assert loop._consecutive_deviations == 1

        # Second deviation — triggers replan
        reflection2 = await loop._reflect_step(step, planned, context)
        assert "[REPLANNED" in reflection2
        assert loop._consecutive_deviations == 0  # Reset after replan
        assert context.execution_plan is not None
        assert context.execution_plan.steps[0].goal == "New S2"
