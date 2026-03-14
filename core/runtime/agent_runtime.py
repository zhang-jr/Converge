"""AgentRuntime - the execution engine for agents.

Manages the event loop and provides the unified entry point for
running agents. All async operations should be coordinated through
the runtime, not via direct asyncio.run() calls.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from core.state.models import AgentConfig, DesiredState, ReconcileResult, SubTask
from core.state.sqlite_store import SQLiteStateStore
from observability.metrics import MetricsCollector, get_global_metrics
from observability.tracer import Tracer
from probes.quality_probe import DefaultQualityProbe, QualityProbe
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from core.runtime.human_intervention import HumanInterventionHandler
    from core.state.state_store import StateStore


class AgentRuntime:
    """Unified runtime for agent execution.

    Manages:
    - Event loop lifecycle
    - StateStore connections
    - Tool registry
    - Tracer configuration
    - Agent instantiation and execution

    Usage:
        runtime = AgentRuntime()
        result = await runtime.run(
            goal="Complete the task",
            agent_config=AgentConfig(agent_id="agent-1"),
        )

    Or with synchronous entry point:
        result = runtime.run_sync(
            goal="Complete the task",
            agent_config=AgentConfig(agent_id="agent-1"),
        )
    """

    def __init__(
        self,
        state_store: StateStore | None = None,
        tool_registry: ToolRegistry | None = None,
        quality_probe: QualityProbe | None = None,
        db_path: str = ":memory:",
        metrics_collector: MetricsCollector | None = None,
        human_intervention_handler: HumanInterventionHandler | None = None,
    ) -> None:
        """Initialize the agent runtime.

        Args:
            state_store: StateStore instance. If None, creates SQLiteStateStore.
            tool_registry: ToolRegistry instance. If None, creates new one.
            quality_probe: QualityProbe instance. If None, uses default.
            db_path: Path for SQLite database if creating default StateStore.
            metrics_collector: MetricsCollector for recording run metrics.
                If None, uses the global collector.
            human_intervention_handler: Handler for human-in-the-loop decisions.
                If None, all interventions auto-approve.
        """
        self._state_store = state_store
        self._owns_state_store = state_store is None
        self._db_path = db_path
        self._tool_registry = tool_registry or ToolRegistry()
        self._quality_probe = quality_probe or DefaultQualityProbe()
        self._tracers: dict[str, Tracer] = {}
        self._initialized = False
        self._metrics = metrics_collector or get_global_metrics()
        self._human_intervention_handler = human_intervention_handler

    @property
    def state_store(self) -> StateStore:
        """Get the state store."""
        if self._state_store is None:
            raise RuntimeError("Runtime not initialized. Call initialize() first.")
        return self._state_store

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the tool registry."""
        return self._tool_registry

    async def initialize(self) -> None:
        """Initialize the runtime.

        Creates StateStore if not provided.
        """
        if self._initialized:
            return

        if self._state_store is None:
            self._state_store = SQLiteStateStore(self._db_path)

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the runtime and release resources."""
        if self._owns_state_store and self._state_store is not None:
            await self._state_store.close()
            self._state_store = None

        self._tracers.clear()
        self._initialized = False

    async def __aenter__(self) -> AgentRuntime:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def get_tracer(self, agent_id: str) -> Tracer:
        """Get or create a tracer for an agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            Tracer instance for the agent.
        """
        if agent_id not in self._tracers:
            self._tracers[agent_id] = Tracer(agent_id=agent_id)
        return self._tracers[agent_id]

    async def run(
        self,
        goal: str,
        agent_config: AgentConfig | None = None,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReconcileResult:
        """Run an agent to achieve a goal.

        This is the main entry point for running agents.

        Args:
            goal: Natural language description of the goal.
            agent_config: Agent configuration. If None, uses defaults.
            constraints: Optional constraints for the goal.
            context: Optional additional context.

        Returns:
            ReconcileResult with execution details.
        """
        await self.initialize()

        config = agent_config or AgentConfig(agent_id="default-agent")

        desired_state = DesiredState(
            goal=goal,
            constraints=constraints or [],
            context=context or {},
        )

        from core.agent.agent import Agent

        agent = Agent(
            config=config,
            runtime=self,
            state_store=self.state_store,
            tool_registry=self._tool_registry,
            quality_probe=self._quality_probe,
            tracer=self.get_tracer(config.agent_id),
            human_intervention_handler=self._human_intervention_handler,
        )

        result = await agent.run(desired_state)
        self._metrics.record_run(result, agent_id=config.agent_id)
        return result

    def run_sync(
        self,
        goal: str,
        agent_config: AgentConfig | None = None,
        constraints: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReconcileResult:
        """Synchronous wrapper for run().

        Creates an event loop if needed and runs the agent.

        Args:
            goal: Natural language description of the goal.
            agent_config: Agent configuration.
            constraints: Optional constraints.
            context: Optional additional context.

        Returns:
            ReconcileResult with execution details.
        """
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call run_sync() from within an async context. "
                "Use 'await runtime.run()' instead."
            )
        except RuntimeError:
            pass

        async def _run() -> ReconcileResult:
            async with self:
                return await self.run(
                    goal=goal,
                    agent_config=agent_config,
                    constraints=constraints,
                    context=context,
                )

        return asyncio.run(_run())

    async def run_multiple(
        self,
        goals: list[dict[str, Any]],
        parallel: bool = False,
    ) -> list[ReconcileResult]:
        """Run multiple agents with different goals.

        Args:
            goals: List of goal configurations, each containing:
                - goal: The goal string
                - agent_config: Optional AgentConfig
                - constraints: Optional constraints
                - context: Optional context
            parallel: Whether to run agents in parallel.

        Returns:
            List of ReconcileResult for each goal.
        """
        await self.initialize()

        async def run_single(goal_config: dict[str, Any]) -> ReconcileResult:
            return await self.run(
                goal=goal_config["goal"],
                agent_config=goal_config.get("agent_config"),
                constraints=goal_config.get("constraints"),
                context=goal_config.get("context"),
            )

        if parallel:
            tasks = [run_single(g) for g in goals]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for goal_config in goals:
                result = await run_single(goal_config)
                results.append(result)
            return results

    async def run_subtask(self, subtask: SubTask) -> SubTask:
        """Execute a SubTask by running an Agent.

        Updates subtask.status to running/completed/failed and stores
        result in subtask.output.

        Args:
            subtask: The SubTask to execute.

        Returns:
            Updated SubTask with status and output populated.
        """
        subtask.status = "running"
        try:
            result = await self.run(
                goal=subtask.goal,
                context={"parent_task_id": subtask.parent_task_id, "task_id": subtask.task_id},
            )
            subtask.status = "completed" if result.converged else "failed"
            subtask.output = {
                "converged": result.converged,
                "total_steps": result.total_steps,
                "status": result.status,
            }
            if not result.converged:
                subtask.error = result.error
        except Exception as e:
            subtask.status = "failed"
            subtask.error = str(e)
        finally:
            from datetime import datetime

            subtask.completed_at = datetime.utcnow()
        return subtask

    async def run_subtasks(
        self,
        subtasks: list[SubTask],
        parallel: bool = False,
    ) -> list[SubTask]:
        """Execute multiple SubTasks.

        Uses sequential or parallel execution based on the parallel flag.

        Args:
            subtasks: List of SubTask instances to execute.
            parallel: Whether to run subtasks in parallel (default sequential).

        Returns:
            List of completed SubTask instances.
        """
        await self.initialize()
        if parallel:
            return list(await asyncio.gather(*[self.run_subtask(st) for st in subtasks]))
        results: list[SubTask] = []
        for st in subtasks:
            completed = await self.run_subtask(st)
            results.append(completed)
        return results
