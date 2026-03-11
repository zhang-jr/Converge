"""Declarative API for the Agent Framework.

Users describe what they want (goals), the framework handles how to achieve it.
This is the primary entry point for using the framework.
"""

from __future__ import annotations

from typing import Any

from core.runtime.agent_runtime import AgentRuntime
from core.state.models import AgentConfig, DesiredState, ReconcileResult
from core.state.sqlite_store import SQLiteStateStore
from probes.quality_probe import DefaultQualityProbe, QualityProbe
from tools.base import ToolBase
from tools.registry import ToolRegistry


class AgentFramework:
    """Main entry point for the Agent Framework.

    Provides a simple, declarative API for running agents.

    Usage:
        framework = AgentFramework()
        framework.register_tool(MyTool())

        result = await framework.run("Complete the task")

        # Or synchronously
        result = framework.run_sync("Complete the task")
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        quality_probe: QualityProbe | None = None,
    ) -> None:
        """Initialize the framework.

        Args:
            db_path: Path to SQLite database for state storage.
            quality_probe: Custom quality probe for output evaluation.
        """
        self._db_path = db_path
        self._tool_registry = ToolRegistry()
        self._quality_probe = quality_probe or DefaultQualityProbe()
        self._runtime: AgentRuntime | None = None
        self._default_config: AgentConfig | None = None

    def register_tool(self, tool: ToolBase) -> AgentFramework:
        """Register a tool for agents to use.

        Args:
            tool: The tool to register.

        Returns:
            Self for method chaining.
        """
        self._tool_registry.register(tool)
        return self

    def register_tools(self, tools: list[ToolBase]) -> AgentFramework:
        """Register multiple tools.

        Args:
            tools: List of tools to register.

        Returns:
            Self for method chaining.
        """
        for tool in tools:
            self.register_tool(tool)
        return self

    def configure_agent(
        self,
        agent_id: str = "default-agent",
        name: str = "",
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        system_prompt: str = "",
        tools: list[str] | None = None,
        safety_max_steps: int = 50,
        confidence_threshold: float = 0.7,
    ) -> AgentFramework:
        """Configure the default agent.

        Args:
            agent_id: Unique agent identifier.
            name: Human-readable name.
            model: LLM model to use.
            system_prompt: Custom system prompt.
            tools: List of tool names the agent can use.
            safety_max_steps: Maximum reconcile loop steps.
            confidence_threshold: Minimum confidence for auto-proceed.

        Returns:
            Self for method chaining.
        """
        self._default_config = AgentConfig(
            agent_id=agent_id,
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
            safety_max_steps=safety_max_steps,
            confidence_threshold=confidence_threshold,
        )

        if tools:
            self._tool_registry.grant_permissions(agent_id, tools)

        return self

    async def run(
        self,
        goal: str,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
        agent_config: AgentConfig | None = None,
    ) -> ReconcileResult:
        """Run an agent to achieve a goal.

        Args:
            goal: Natural language description of the goal.
            constraints: Optional constraints for the goal.
            context: Optional additional context.
            agent_config: Optional agent configuration override.

        Returns:
            ReconcileResult with execution details.
        """
        config = agent_config or self._default_config

        state_store = SQLiteStateStore(self._db_path)
        runtime = AgentRuntime(
            state_store=state_store,
            tool_registry=self._tool_registry,
            quality_probe=self._quality_probe,
        )

        async with runtime:
            return await runtime.run(
                goal=goal,
                agent_config=config,
                constraints=constraints,
                context=context,
            )

    def run_sync(
        self,
        goal: str,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
        agent_config: AgentConfig | None = None,
    ) -> ReconcileResult:
        """Synchronous wrapper for run().

        Args:
            goal: Natural language description of the goal.
            constraints: Optional constraints.
            context: Optional additional context.
            agent_config: Optional agent configuration override.

        Returns:
            ReconcileResult with execution details.
        """
        import asyncio
        return asyncio.run(
            self.run(goal, constraints, context, agent_config)
        )


def goal(
    description: str,
    constraints: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> DesiredState:
    """Create a DesiredState from a goal description.

    Convenience function for creating goals.

    Args:
        description: Natural language goal description.
        constraints: Optional constraints.
        context: Optional additional context.

    Returns:
        DesiredState object.

    Example:
        state = goal(
            "Find all Python files and count lines of code",
            constraints=["Only count .py files", "Exclude tests"],
        )
    """
    return DesiredState(
        goal=description,
        constraints=constraints or [],
        context=context or {},
    )


def agent(
    agent_id: str,
    tools: list[str] | None = None,
    model: str = "anthropic/claude-3-5-sonnet-20241022",
    system_prompt: str = "",
    **kwargs: Any,
) -> AgentConfig:
    """Create an agent configuration.

    Convenience function for creating agent configs.

    Args:
        agent_id: Unique agent identifier.
        tools: List of tool names the agent can use.
        model: LLM model to use.
        system_prompt: Custom system prompt.
        **kwargs: Additional AgentConfig parameters.

    Returns:
        AgentConfig object.

    Example:
        config = agent(
            "code-analyst",
            tools=["read_file", "search_code"],
            system_prompt="You are a code analysis expert.",
        )
    """
    return AgentConfig(
        agent_id=agent_id,
        tools=tools or [],
        model=model,
        system_prompt=system_prompt,
        **kwargs,
    )


async def run_agent(
    goal_text: str,
    agent_config: AgentConfig | None = None,
    tools: list[ToolBase] | None = None,
    constraints: list[str] | None = None,
    context: dict[str, Any] | None = None,
    db_path: str = ":memory:",
) -> ReconcileResult:
    """One-shot function to run an agent with a goal.

    Convenience function for quick agent execution.

    Args:
        goal_text: Natural language goal description.
        agent_config: Optional agent configuration.
        tools: Optional list of tools to register.
        constraints: Optional constraints.
        context: Optional additional context.
        db_path: Path for SQLite database.

    Returns:
        ReconcileResult with execution details.

    Example:
        result = await run_agent(
            "Analyze the codebase and find security issues",
            tools=[FileReadTool(), CodeSearchTool()],
        )
    """
    framework = AgentFramework(db_path=db_path)

    if tools:
        framework.register_tools(tools)
        if agent_config:
            tool_names = [t.name for t in tools]
            framework._tool_registry.grant_permissions(
                agent_config.agent_id,
                tool_names,
            )

    return await framework.run(
        goal=goal_text,
        constraints=constraints,
        context=context,
        agent_config=agent_config,
    )
