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
import time
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

            llm_tokens_used: int = response.get("usage", {}).get("total_tokens", 0)
            llm_latency_ms: float = response.get("latency_ms", 0.0)

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
                llm_tokens_used=llm_tokens_used,
                llm_latency_ms=llm_latency_ms,
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

        Loads model config from environment (via core.config).
        Falls back to mock response if litellm is unavailable (for testing).

        Args:
            messages: Conversation messages in OpenAI format.

        Returns:
            Response dict with choices, usage, and latency_ms.
        """
        from core.config import LLM_API_BASE, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

        tools_schema = self._build_tools_schema()

        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": self._config.model or LLM_MODEL,
                "messages": messages,
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
            }
            if LLM_API_BASE:
                kwargs["api_base"] = LLM_API_BASE
            if tools_schema:
                kwargs["tools"] = tools_schema

            t0 = time.monotonic()
            response = await litellm.acompletion(**kwargs)
            latency_ms = (time.monotonic() - t0) * 1000.0

            # Normalize to a plain dict for consistent handling
            usage = response.usage
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in (response.choices[0].message.tool_calls or [])
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                "latency_ms": latency_ms,
            }

        except ImportError:
            # litellm not installed: return mock (useful for unit tests)
            return self._mock_llm_response()

    def _mock_llm_response(self) -> dict[str, Any]:
        """Return a deterministic mock LLM response for testing."""
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "reasoning": "Analyzing the goal and determining next steps.",
                            "action": "Proceeding with goal analysis.",
                        }),
                        "tool_calls": [],
                    }
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0.0,
        }

    def _build_tools_schema(self) -> list[dict[str, Any]]:
        """Build OpenAI-format tools schema from registered tools."""
        if self._tool_registry is None or not self._config.tools:
            return []
        schema: list[dict[str, Any]] = []
        for tool_name in self._config.tools:
            try:
                info = self._tool_registry.get_tool_info(tool_name)
                schema.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": info.get("description", ""),
                        "parameters": info.get(
                            "parameters_schema", {"type": "object", "properties": {}}
                        ),
                    },
                })
            except KeyError:
                pass
        return schema

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Parse LLM response into action, reasoning, and tool calls.

        Handles both text responses (JSON with reasoning/action) and
        LiteLLM tool_calls format.
        """
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls", []) or []

        # Parse tool calls from LiteLLM format
        tool_calls: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            arguments = func.get("arguments", "{}")
            try:
                parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (ValueError, TypeError):
                parsed_args = {}
            tool_calls.append({"name": tool_name, "params": parsed_args})

        if tool_calls:
            action = f"Calling tools: {', '.join(tc['name'] for tc in tool_calls)}"
            reasoning = content
            return action, reasoning, tool_calls

        # Text response: try JSON parse first
        try:
            parsed = json.loads(content)
            return (
                parsed.get("action", "No action"),
                parsed.get("reasoning", ""),
                parsed.get("tool_calls", []),
            )
        except (json.JSONDecodeError, AttributeError):
            return content[:200] if content else "No action", content, []

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
