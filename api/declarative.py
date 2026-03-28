"""Declarative API for the Agent Framework.

Users describe what they want (goals), the framework handles how to achieve it.
This is the primary entry point for using the framework.
"""

from __future__ import annotations

from typing import Any

from core.runtime.agent_runtime import AgentRuntime
from core.state.models import AgentConfig, ConvergenceCriterion, DesiredState, ReconcileResult
from core.state.sqlite_store import SQLiteStateStore
from probes.quality_probe import DefaultQualityProbe, QualityProbe
from skills.base import SkillBase
from skills.registry import SkillRegistry
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
        self._skill_registry: SkillRegistry = SkillRegistry()
        self._loaded_skills: list[SkillBase] = []

    def register_tool(self, tool: ToolBase) -> AgentFramework:
        """Register a tool for agents to use.

        Args:
            tool: The tool to register.

        Returns:
            Self for method chaining.
        """
        self._tool_registry.register(tool)
        return self

    def load_skill(self, skill: str | SkillBase) -> AgentFramework:
        """Load a skill into the framework, registering its tools.

        When a skill is loaded:
        1. All skill tools are registered in the ToolRegistry.
        2. The skill's ``system_prompt_addon`` is appended to the agent's
           system prompt on the next ``run()`` call.
        3. The skill's ``convergence_criteria`` are merged into the
           DesiredState on the next ``run()`` call.

        Args:
            skill: A :class:`~skills.base.SkillBase` instance, or the name
                of a skill previously registered via
                ``framework._skill_registry.register()``.

        Returns:
            Self for method chaining.

        Raises:
            KeyError: If ``skill`` is a string and not found in the registry.

        Example::

            from skills.builtin.code_review import create_code_review_skill
            framework.load_skill(create_code_review_skill())
            result = await framework.run("Review the auth module")
        """
        if isinstance(skill, str):
            skill = self._skill_registry.get(skill)

        self.register_tools(skill.tools)
        self._loaded_skills.append(skill)
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
        planning_strategy: str = "auto",
    ) -> ReconcileResult:
        """Run an agent to achieve a goal.

        Args:
            goal: Natural language description of the goal.
            constraints: Optional constraints for the goal.
            context: Optional additional context.
            agent_config: Optional agent configuration override.
            planning_strategy: Controls LLM planning phase.
                ``"auto"`` (default) lets the LLM decide whether planning is
                needed.  ``"always"`` forces planning.  ``"never"`` skips
                planning entirely (backward compatible with Phase 7 behavior).

        Returns:
            ReconcileResult with execution details.
        """
        config = agent_config or self._default_config

        # Merge loaded-skill system prompt addons and convergence criteria.
        config = self._apply_skills_to_config(config)
        extra_criteria = self._collect_skill_criteria()

        state_store = SQLiteStateStore(self._db_path)
        runtime = AgentRuntime(
            state_store=state_store,
            tool_registry=self._tool_registry,
            quality_probe=self._quality_probe,
        )

        desired_state = DesiredState(
            goal=goal,
            constraints=constraints or [],
            context=context or {},
            convergence_criteria=extra_criteria,
            planning_strategy=planning_strategy,
        )

        async with runtime:
            from core.agent.agent import Agent

            agent = Agent(
                config=config,
                state_store=state_store,
                tool_registry=self._tool_registry,
                quality_probe=self._quality_probe,
            )
            return await agent.run(desired_state)

    # ------------------------------------------------------------------
    # Skill helpers (private)
    # ------------------------------------------------------------------

    def _apply_skills_to_config(
        self,
        config: AgentConfig | None,
    ) -> AgentConfig | None:
        """Return a new AgentConfig with skill system_prompt_addons merged in."""
        if not self._loaded_skills:
            return config

        addons = [s.system_prompt_addon for s in self._loaded_skills if s.system_prompt_addon]
        if not addons:
            return config

        addon_text = "\n\n".join(addons)

        if config is None:
            config = AgentConfig(
                agent_id="default-agent",
                system_prompt=addon_text,
            )
        else:
            base_prompt = config.system_prompt or ""
            separator = "\n\n" if base_prompt else ""
            config = config.model_copy(
                update={"system_prompt": base_prompt + separator + addon_text}
            )

        return config

    def _collect_skill_criteria(self) -> list[ConvergenceCriterion]:
        """Collect convergence criteria from all loaded skills."""
        criteria: list[ConvergenceCriterion] = []
        for skill in self._loaded_skills:
            criteria.extend(skill.convergence_criteria)
        return criteria

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
