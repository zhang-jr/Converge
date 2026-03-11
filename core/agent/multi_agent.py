"""Multi-Agent Orchestration.

Provides patterns for coordinating multiple agents toward a shared goal.
Three patterns are supported, analogous to common distributed systems patterns:

- pipeline: Agents run in sequence; each agent's output feeds the next.
  Useful when tasks have a natural processing pipeline (e.g., research → draft → review).

- supervisor: A lead agent breaks the goal into subtasks, dispatches them to
  specialist agents, then synthesizes their results. Useful for complex goals
  that benefit from specialization.

- pool: A dispatcher routes sub-tasks to whichever agent is available.
  Useful when the same task type needs parallel execution.

All patterns share state via a common StateStore, with each agent optionally
scoped to its own namespace prefix for isolation.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from core.state.models import AgentConfig, DesiredState, ReconcileResult
from core.workflow.workflow import (
    WorkflowExecutionResult,
    WorkflowSpec,
    WorkflowStep,
)
from errors.exceptions import AgentFrameworkError

if TYPE_CHECKING:
    from core.runtime.agent_runtime import AgentRuntime
    from core.workflow.controller import WorkflowController


class MultiAgentConfig(BaseModel):
    """Configuration for a multi-agent orchestration run.

    Attributes:
        orchestrator_id: Unique identifier for this orchestrator.
        name: Human-readable name.
        pattern: Execution pattern (pipeline / supervisor / pool).
        agents: Configurations for sub-agents.
        shared_state: If True, all agents share the same StateStore namespace.
        metadata: Arbitrary metadata.
    """

    orchestrator_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    pattern: Literal["pipeline", "supervisor", "pool"]
    agents: list[AgentConfig]
    shared_state: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubTaskSpec(BaseModel):
    """A sub-task dispatched to a specific agent.

    Attributes:
        task_id: Unique identifier for this sub-task.
        agent_config: The agent to execute this task.
        goal: The sub-goal for this agent.
        constraints: Constraints for the agent.
        context: Context from prior steps.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_config: AgentConfig
    goal: str
    constraints: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class MultiAgentResult(BaseModel):
    """Result of a multi-agent orchestration run.

    Attributes:
        orchestrator_id: The orchestrator that produced this result.
        pattern: The execution pattern used.
        agent_results: Per-agent results keyed by agent_id.
        synthesis: Final synthesized output (for supervisor pattern).
        status: Overall status.
        started_at: When execution started.
        completed_at: When execution finished.
        error: Error message if the run failed.
        total_agents: Number of agents in this run.
        completed_agents: Number of agents that converged.
    """

    orchestrator_id: str
    pattern: Literal["pipeline", "supervisor", "pool"]
    agent_results: dict[str, ReconcileResult] = Field(default_factory=dict)
    synthesis: str = ""
    status: Literal["completed", "failed", "partial"]
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None
    total_agents: int = 0
    completed_agents: int = 0

    @property
    def duration_ms(self) -> float | None:
        """Total duration in milliseconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000


class MultiAgentOrchestrator:
    """Orchestrates multiple agents toward a shared goal.

    Usage (pipeline pattern):
        agents = [
            AgentConfig(agent_id="researcher", ...),
            AgentConfig(agent_id="writer", ...),
            AgentConfig(agent_id="reviewer", ...),
        ]
        config = MultiAgentConfig(
            name="Research Pipeline",
            pattern="pipeline",
            agents=agents,
        )
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(
            DesiredState(goal="Write a report on quantum computing")
        )

    Usage (supervisor pattern):
        config = MultiAgentConfig(
            name="Research Team",
            pattern="supervisor",
            agents=[
                AgentConfig(agent_id="supervisor", ...),
                AgentConfig(agent_id="specialist-1", ...),
                AgentConfig(agent_id="specialist-2", ...),
            ],
        )
        # First agent is the supervisor; others are specialists
    """

    def __init__(
        self,
        config: MultiAgentConfig,
        runtime: AgentRuntime,
    ) -> None:
        """Initialize the multi-agent orchestrator.

        Args:
            config: Orchestrator configuration.
            runtime: Shared AgentRuntime for executing agents.
        """
        self._config = config
        self._runtime = runtime

    async def run(self, desired_state: DesiredState) -> MultiAgentResult:
        """Run the multi-agent orchestration.

        Dispatches to the appropriate pattern implementation.

        Args:
            desired_state: The overall goal to achieve.

        Returns:
            MultiAgentResult with per-agent outcomes.
        """
        result = MultiAgentResult(
            orchestrator_id=self._config.orchestrator_id,
            pattern=self._config.pattern,
            total_agents=len(self._config.agents),
            status="completed",
        )

        try:
            if self._config.pattern == "pipeline":
                await self._run_pipeline(desired_state, result)
            elif self._config.pattern == "supervisor":
                await self._run_supervisor(desired_state, result)
            elif self._config.pattern == "pool":
                await self._run_pool(desired_state, result)
            else:
                raise AgentFrameworkError(
                    f"Unknown multi-agent pattern: {self._config.pattern}"
                )

            failed = result.total_agents - result.completed_agents
            if failed <= 0:
                # completed_agents >= total_agents: all configured agents converged.
                # supervisor pattern can legitimately exceed total_agents (planning +
                # synthesis phases both increment the counter for the same agent).
                result.status = "completed"
            elif result.completed_agents > 0:
                result.status = "partial"
            else:
                result.status = "failed"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
        finally:
            result.completed_at = datetime.utcnow()

        return result

    async def _run_pipeline(
        self,
        desired_state: DesiredState,
        result: MultiAgentResult,
    ) -> None:
        """Pipeline pattern: agents run in sequence, each passing context forward.

        Agent N's output is injected as context into Agent N+1's goal.
        """
        current_context: dict[str, Any] = dict(desired_state.context)

        for agent_config in self._config.agents:
            agent_result = await self._runtime.run(
                goal=desired_state.goal,
                agent_config=agent_config,
                constraints=desired_state.constraints,
                context=current_context,
            )

            result.agent_results[agent_config.agent_id] = agent_result

            if agent_result.converged:
                result.completed_agents += 1
                # Pass this agent's final state as context to the next agent
                current_context = {
                    **current_context,
                    f"{agent_config.agent_id}_output": agent_result.final_state,
                    "previous_agent": agent_config.agent_id,
                }
            else:
                # Pipeline fails fast on any agent failure
                raise AgentFrameworkError(
                    f"Pipeline agent '{agent_config.agent_id}' failed: {agent_result.error}",
                    agent_id=agent_config.agent_id,
                )

    async def _run_supervisor(
        self,
        desired_state: DesiredState,
        result: MultiAgentResult,
    ) -> None:
        """Supervisor pattern: first agent decomposes goal, others execute subtasks.

        The supervisor (first agent in config.agents) breaks the goal into
        sub-tasks and assigns them to specialist agents (remaining agents).
        The supervisor then synthesizes the results.

        In this implementation, the supervisor runs first to produce a plan
        (stored in final_state), then specialist agents run in parallel using
        the plan as context, and finally the supervisor runs again to synthesize.
        """
        if not self._config.agents:
            raise AgentFrameworkError("Supervisor pattern requires at least one agent")

        supervisor_config = self._config.agents[0]
        specialists = self._config.agents[1:]

        # Phase 1: Supervisor plans
        planning_result = await self._runtime.run(
            goal=f"Break down the following goal into subtasks for specialists: {desired_state.goal}",
            agent_config=supervisor_config,
            constraints=desired_state.constraints,
            context=desired_state.context,
        )
        result.agent_results[f"{supervisor_config.agent_id}_planning"] = planning_result
        if planning_result.converged:
            result.completed_agents += 1

        plan_context = {
            **desired_state.context,
            "supervisor_plan": planning_result.final_state,
            "overall_goal": desired_state.goal,
        }

        # Phase 2: Specialists execute in parallel
        if specialists:
            specialist_tasks = [
                asyncio.create_task(
                    self._runtime.run(
                        goal=desired_state.goal,
                        agent_config=spec_config,
                        constraints=desired_state.constraints,
                        context=plan_context,
                    )
                )
                for spec_config in specialists
            ]
            specialist_results = await asyncio.gather(
                *specialist_tasks, return_exceptions=True
            )

            specialist_outputs: dict[str, Any] = {}
            for spec_config, spec_result in zip(specialists, specialist_results):
                if isinstance(spec_result, Exception):
                    result.agent_results[spec_config.agent_id] = ReconcileResult(
                        status="failed",
                        error=str(spec_result),
                        trace_id="",
                    )
                else:
                    result.agent_results[spec_config.agent_id] = spec_result
                    if spec_result.converged:
                        result.completed_agents += 1
                        specialist_outputs[spec_config.agent_id] = spec_result.final_state

        # Phase 3: Supervisor synthesizes
        synthesis_context = {
            **plan_context,
            "specialist_outputs": specialist_outputs if specialists else {},
        }
        synthesis_result = await self._runtime.run(
            goal=f"Synthesize the specialist results into a final answer for: {desired_state.goal}",
            agent_config=supervisor_config,
            constraints=desired_state.constraints,
            context=synthesis_context,
        )
        result.agent_results[f"{supervisor_config.agent_id}_synthesis"] = synthesis_result
        if synthesis_result.converged:
            result.completed_agents += 1
            # Extract synthesis output for the result
            result.synthesis = str(synthesis_result.final_state.get("summary", ""))

    async def _run_pool(
        self,
        desired_state: DesiredState,
        result: MultiAgentResult,
    ) -> None:
        """Pool pattern: all agents run the same goal in parallel.

        Useful when you want multiple independent attempts at the same goal,
        or when agents are differentiated only by their tools/prompts.
        """
        tasks = [
            asyncio.create_task(
                self._runtime.run(
                    goal=desired_state.goal,
                    agent_config=agent_config,
                    constraints=desired_state.constraints,
                    context=desired_state.context,
                )
            )
            for agent_config in self._config.agents
        ]

        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent_config, agent_result in zip(self._config.agents, agent_results):
            if isinstance(agent_result, Exception):
                result.agent_results[agent_config.agent_id] = ReconcileResult(
                    status="failed",
                    error=str(agent_result),
                    trace_id="",
                )
            else:
                result.agent_results[agent_config.agent_id] = agent_result
                if agent_result.converged:
                    result.completed_agents += 1


def pipeline(
    agents: list[AgentConfig],
    name: str = "Pipeline",
) -> MultiAgentConfig:
    """Convenience constructor for pipeline multi-agent configs.

    Args:
        agents: Ordered list of agent configs (first to last in pipeline).
        name: Human-readable orchestrator name.

    Returns:
        MultiAgentConfig with pattern="pipeline".
    """
    return MultiAgentConfig(name=name, pattern="pipeline", agents=agents)


def supervisor(
    supervisor_config: AgentConfig,
    specialists: list[AgentConfig],
    name: str = "Supervisor",
) -> MultiAgentConfig:
    """Convenience constructor for supervisor multi-agent configs.

    Args:
        supervisor_config: The lead agent that plans and synthesizes.
        specialists: Specialist agents that handle specific subtasks.
        name: Human-readable orchestrator name.

    Returns:
        MultiAgentConfig with pattern="supervisor".
    """
    return MultiAgentConfig(
        name=name,
        pattern="supervisor",
        agents=[supervisor_config, *specialists],
    )


def pool(
    agents: list[AgentConfig],
    name: str = "Pool",
) -> MultiAgentConfig:
    """Convenience constructor for pool multi-agent configs.

    Args:
        agents: Agents in the pool (all receive the same goal).
        name: Human-readable orchestrator name.

    Returns:
        MultiAgentConfig with pattern="pool".
    """
    return MultiAgentConfig(name=name, pattern="pool", agents=agents)
