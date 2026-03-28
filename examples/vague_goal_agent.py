"""Vague-goal agent example (Phase 8).

Demonstrates LLM-powered goal decomposition:
1. Auto planning — vague goal is automatically decomposed into ExecutionPlan
2. Always planning — explicit planning for any goal
3. Never planning — backward-compatible behavior (no decomposition)
4. Clarification — when goal is too vague, HumanIntervention is triggered

Prerequisites:
  - Copy .env.example to .env and set ANTHROPIC_API_KEY
  - pip install litellm python-dotenv
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.declarative import AgentFramework
from core.runtime.human_intervention import CallbackHumanInterventionHandler
from core.state.models import AgentConfig, HumanDecision
from tools.code.file_tools import ReadFileTool
from tools.code.search_tools import GlobTool, GrepTool
from tools.code.shell_tools import BashTool
from tools.registry import ToolRegistry


def _make_framework() -> tuple[AgentFramework, ToolRegistry, AgentConfig]:
    """Create a framework with code tools registered."""
    registry = ToolRegistry()
    for tool in [ReadFileTool(), GlobTool(), GrepTool(), BashTool()]:
        registry.register(tool)

    config = AgentConfig(
        agent_id="vague-goal-agent",
        tools=["read_file", "glob", "grep", "bash"],
        safety_max_steps=10,
    )
    registry.grant_permissions("vague-goal-agent", config.tools)

    framework = AgentFramework(db_path=":memory:")
    framework._tool_registry = registry
    return framework, registry, config


async def scenario_auto_planning() -> None:
    """Scenario 1: Auto planning with a vague goal.

    The LLMPlanner automatically decomposes the goal into steps.
    planning_strategy="auto" is the default.
    """
    print("\n=== Scenario 1: Auto Planning (vague goal) ===")  # noqa: T201

    framework, _, config = _make_framework()
    result = await framework.run(
        goal="Analyze the project structure and summarize what this codebase does",
        agent_config=config,
        planning_strategy="auto",
    )

    print(f"  Status: {result.status}, Steps: {result.total_steps}")  # noqa: T201
    for step in result.steps:
        print(f"  Step {step.step_number}: {step.action}")  # noqa: T201
        if step.reflection:
            print(f"    Reflection: {step.reflection[:120]}")  # noqa: T201


async def scenario_always_planning() -> None:
    """Scenario 2: Forced planning even for a clear goal.

    planning_strategy="always" ensures a plan is generated.
    """
    print("\n=== Scenario 2: Always Planning (explicit goal) ===")  # noqa: T201

    framework, _, config = _make_framework()
    result = await framework.run(
        goal="Read examples/vague_goal_agent.py and count its lines using bash wc -l",
        agent_config=config,
        planning_strategy="always",
    )

    print(f"  Status: {result.status}, Steps: {result.total_steps}")  # noqa: T201
    for step in result.steps:
        print(f"  Step {step.step_number}: {step.action}")  # noqa: T201
        if step.reflection:
            print(f"    Reflection: {step.reflection[:120]}")  # noqa: T201


async def scenario_never_planning() -> None:
    """Scenario 3: No planning — backward-compatible mode.

    planning_strategy="never" behaves exactly like Phase 7.
    """
    print("\n=== Scenario 3: Never Planning (backward compatible) ===")  # noqa: T201

    framework, _, config = _make_framework()
    result = await framework.run(
        goal="Read examples/vague_goal_agent.py and report its first 5 lines",
        agent_config=config,
        planning_strategy="never",
    )

    print(f"  Status: {result.status}, Steps: {result.total_steps}")  # noqa: T201
    for step in result.steps:
        print(f"  Step {step.step_number}: {step.action}")  # noqa: T201


async def scenario_clarification() -> None:
    """Scenario 4: Goal too vague — triggers clarification.

    Uses CallbackHumanInterventionHandler to simulate a user answering
    the clarification question.
    """
    print("\n=== Scenario 4: Clarification (very vague goal) ===")  # noqa: T201

    async def auto_approve(reason, context, pending_action):
        question = (pending_action or {}).get("clarification_question", "")
        print(f"  [HUMAN] Clarification requested: {question}")  # noqa: T201
        feedback = "I want to find all TODO comments in Python files"
        print(f"  [HUMAN] User responds: {feedback}")  # noqa: T201
        return HumanDecision(
            approved=True,
            feedback=feedback,
            decision_by="simulated_human",
        )

    from core.agent.agent import Agent
    from core.state.sqlite_store import SQLiteStateStore

    framework, registry, config = _make_framework()
    store = SQLiteStateStore(":memory:")

    agent = Agent(
        config=config,
        state_store=store,
        tool_registry=registry,
        human_intervention_handler=CallbackHumanInterventionHandler(auto_approve),
    )

    from core.state.models import DesiredState

    result = await agent.run(
        DesiredState(
            goal="Help me clean up this code",
            planning_strategy="auto",
        ),
    )

    print(f"  Status: {result.status}, Steps: {result.total_steps}")  # noqa: T201
    for step in result.steps:
        print(f"  Step {step.step_number}: {step.action}")  # noqa: T201

    await store.close()


async def main() -> None:
    """Run all scenarios."""
    await scenario_auto_planning()
    await scenario_always_planning()
    await scenario_never_planning()
    await scenario_clarification()


if __name__ == "__main__":
    asyncio.run(main())
