"""Agent - the core agent abstraction.

An Agent is the minimal deployment unit, combining:
- LLM calls (via LiteLLM)
- Tools (via ToolRegistry)
- Memory (state via StateStore)
- Quality evaluation (via QualityProbe)

Agents execute through the ReconcileLoop pattern.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from core.runtime.reconcile_loop import ReconcileLoop
from core.state.models import (
    AgentConfig,
    DesiredState,
    LoopContext,
    ReconcileResult,
    StepOutput,
    ToolCall,
)
from observability.tracer import Tracer
from probes.quality_probe import DefaultQualityProbe, QualityProbe

if TYPE_CHECKING:
    from core.runtime.agent_runtime import AgentRuntime
    from core.runtime.human_intervention import HumanInterventionHandler
    from core.state.state_store import StateStore
    from tools.registry import ToolRegistry


class AgentReconcileLoop(ReconcileLoop):
    """ReconcileLoop implementation for Agent execution.

    Implements the observe/diff/act cycle using LLM calls.
    """

    def __init__(
        self,
        agent: Agent,
        state_store: StateStore,
        tool_registry: ToolRegistry | None = None,
        quality_probe: QualityProbe | None = None,
        tracer: Tracer | None = None,
        human_intervention_handler: HumanInterventionHandler | None = None,
    ) -> None:
        super().__init__(
            state_store=state_store,
            tool_registry=tool_registry,
            quality_probe=quality_probe,
            tracer=tracer,
            safety_max_steps=agent.config.safety_max_steps,
            confidence_threshold=agent.config.confidence_threshold,
            agent_id=agent.config.agent_id,
            human_intervention_handler=human_intervention_handler,
        )
        self._agent = agent

    async def _observe(self, context: LoopContext) -> dict[str, Any]:
        """Observe current state including conversation history."""
        observed = await super()._observe(context)
        observed["history"] = [
            {
                "step": s.step_number,
                "action": s.action,
                "result": s.result,
            }
            for s in context.history
        ]
        return observed

    async def _diff(
        self,
        observed: dict[str, Any],
        desired: DesiredState,
    ) -> dict[str, Any] | None:
        """Compute what needs to be done to achieve the goal."""
        if context_history := observed.get("history"):
            last_step = context_history[-1] if context_history else None
            if last_step and self._check_goal_achieved(last_step, desired):
                return None

        return {
            "goal": desired.goal,
            "constraints": desired.constraints,
            "observed_state": observed,
            "context": desired.context,
        }

    def _check_goal_achieved(
        self,
        last_step: dict[str, Any],
        desired: DesiredState,
    ) -> bool:
        """Simple heuristic to check if goal might be achieved."""
        result = str(last_step.get("result", "")).lower()
        goal_keywords = ["completed", "done", "finished", "success"]
        return any(kw in result for kw in goal_keywords)

    async def _act(
        self,
        diff: dict[str, Any],
        context: LoopContext,
    ) -> StepOutput:
        """Execute agent action using LLM."""
        return await self._agent.think_and_act(diff, context)


class Agent:
    """Core Agent class.

    Combines LLM reasoning with tool execution to achieve goals.
    Executes through the ReconcileLoop pattern for reliable convergence.

    Usage:
        config = AgentConfig(agent_id="my-agent", tools=["search", "calculate"])
        agent = Agent(config, runtime, state_store, tool_registry)
        result = await agent.run(DesiredState(goal="Find the answer"))
    """

    def __init__(
        self,
        config: AgentConfig,
        runtime: AgentRuntime | None = None,
        state_store: StateStore | None = None,
        tool_registry: ToolRegistry | None = None,
        quality_probe: QualityProbe | None = None,
        tracer: Tracer | None = None,
        human_intervention_handler: HumanInterventionHandler | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration.
            runtime: AgentRuntime (optional, for dependency access).
            state_store: StateStore for state management.
            tool_registry: ToolRegistry for tool access.
            quality_probe: QualityProbe for output evaluation.
            tracer: Tracer for observability.
            human_intervention_handler: Handler for human approval decisions.
        """
        self._config = config
        self._runtime = runtime
        self._state_store = state_store
        self._tool_registry = tool_registry
        self._quality_probe = quality_probe or DefaultQualityProbe()
        self._tracer = tracer or Tracer(agent_id=config.agent_id)
        self._human_intervention_handler = human_intervention_handler
        self._llm_client: Any = None

    @property
    def config(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self._config.agent_id

    async def run(self, desired_state: DesiredState) -> ReconcileResult:
        """Run the agent to achieve the desired state.

        Args:
            desired_state: The goal to achieve.

        Returns:
            ReconcileResult with execution details.
        """
        if self._state_store is None:
            raise RuntimeError("Agent requires a StateStore")

        loop = AgentReconcileLoop(
            agent=self,
            state_store=self._state_store,
            tool_registry=self._tool_registry,
            quality_probe=self._quality_probe,
            tracer=self._tracer,
            human_intervention_handler=self._human_intervention_handler,
        )

        return await loop.run(desired_state)

    async def think_and_act(
        self,
        diff: dict[str, Any],
        context: LoopContext,
    ) -> StepOutput:
        """Process the current situation and take action.

        This is where the LLM reasoning happens. In a full implementation,
        this would call the LLM to decide what action to take.

        Args:
            diff: What needs to change to achieve the goal.
            context: Current loop context.

        Returns:
            StepOutput describing the action taken.
        """
        goal = diff.get("goal", "")
        constraints = diff.get("constraints", [])

        messages = self._build_messages(goal, constraints, context)

        try:
            response = await self._call_llm(messages)
            action, reasoning, tool_calls = self._parse_response(response)

            executed_tools = []
            for tc in tool_calls:
                result = await self._execute_tool_call(tc)
                executed_tools.append(result)

            return StepOutput(
                step_number=context.current_step,
                action=action,
                reasoning=reasoning,
                tool_calls=executed_tools,
                result=self._summarize_results(executed_tools),
            )

        except Exception as e:
            return StepOutput(
                step_number=context.current_step,
                action="Error during thinking",
                reasoning=f"Exception: {e}",
                tool_calls=[],
                result={"error": str(e)},
            )

    def _build_messages(
        self,
        goal: str,
        constraints: list[str],
        context: LoopContext,
    ) -> list[dict[str, str]]:
        """Build message list for LLM call."""
        system_prompt = self._config.system_prompt or self._default_system_prompt()

        constraint_text = ""
        if constraints:
            constraint_text = "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)

        history_text = ""
        if context.history:
            history_items = []
            for step in context.history[-5:]:
                history_items.append(f"Step {step.step_number}: {step.action}")
                if step.result:
                    history_items.append(f"  Result: {step.result}")
            history_text = "\n\nRecent history:\n" + "\n".join(history_items)

        user_message = f"Goal: {goal}{constraint_text}{history_text}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        tools_desc = ""
        if self._tool_registry and self._config.tools:
            tool_infos = []
            for tool_name in self._config.tools:
                try:
                    info = self._tool_registry.get_tool_info(tool_name)
                    tool_infos.append(f"- {tool_name}: {info.get('description', '')}")
                except KeyError:
                    pass
            if tool_infos:
                tools_desc = "\n\nAvailable tools:\n" + "\n".join(tool_infos)

        return f"""You are an AI agent working to achieve goals.

Analyze the goal and take actions to achieve it.
Think step by step and explain your reasoning.
{tools_desc}

Respond with:
1. Your reasoning about what to do
2. The action you will take
3. Any tool calls needed (as JSON)"""

    async def _call_llm(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call the LLM via LiteLLM.

        This is a placeholder that returns a mock response.
        In production, this would use litellm.acompletion().
        """
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "reasoning": "Analyzing the goal and determining next steps.",
                            "action": "Proceeding with goal analysis.",
                            "tool_calls": [],
                        }),
                    }
                }
            ]
        }

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Parse LLM response into action, reasoning, and tool calls."""
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        try:
            parsed = json.loads(content)
            return (
                parsed.get("action", "No action"),
                parsed.get("reasoning", ""),
                parsed.get("tool_calls", []),
            )
        except json.JSONDecodeError:
            return (content, "", [])

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> ToolCall:
        """Execute a single tool call."""
        tool_name = tool_call.get("name", "")
        params = tool_call.get("params", {})

        if self._tool_registry is None:
            return ToolCall(
                tool_name=tool_name,
                params=params,
                success=False,
                error="No tool registry available",
            )

        try:
            tool = self._tool_registry.get_tool(
                tool_name,
                agent_id=self._config.agent_id,
            )
            result = await tool.execute(params)
            return ToolCall(
                tool_name=tool_name,
                params=params,
                result=result.output,
                success=result.success,
                error=result.error,
            )
        except Exception as e:
            return ToolCall(
                tool_name=tool_name,
                params=params,
                success=False,
                error=str(e),
            )

    def _summarize_results(self, tool_calls: list[ToolCall]) -> Any:
        """Summarize results from tool calls."""
        if not tool_calls:
            return {"status": "no_tools_called"}

        results = []
        for tc in tool_calls:
            results.append({
                "tool": tc.tool_name,
                "success": tc.success,
                "result": tc.result if tc.success else tc.error,
            })

        all_success = all(tc.success for tc in tool_calls)
        return {
            "status": "completed" if all_success else "partial_failure",
            "tool_results": results,
        }
