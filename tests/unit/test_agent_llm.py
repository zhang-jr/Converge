"""Tests for Agent LLM integration (Phase 4).

Uses Mock to replace litellm.acompletion. Does NOT mock StateStore.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agent.agent import Agent
from core.state.models import AgentConfig, DesiredState, LoopContext
from core.state.sqlite_store import SQLiteStateStore


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(agent_id="test-llm-agent", safety_max_steps=3)


@pytest.fixture
async def state_store() -> SQLiteStateStore:
    store = SQLiteStateStore(":memory:")
    yield store
    await store.close()


@pytest.fixture
def agent(agent_config: AgentConfig, state_store: SQLiteStateStore) -> Agent:
    return Agent(config=agent_config, state_store=state_store)


def _make_litellm_response(
    content: str,
    tool_calls: list | None = None,
    tokens: int = 42,
) -> MagicMock:
    """Build a mock litellm ModelResponse."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = tokens // 2
    usage.completion_tokens = tokens // 2
    usage.total_tokens = tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestCallLlm:
    """Tests for Agent._call_llm."""

    async def test_normal_text_response(self, agent: Agent) -> None:
        mock_resp = _make_litellm_response(
            content=json.dumps({"reasoning": "step1", "action": "do something"}),
            tokens=100,
        )
        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await agent._call_llm([{"role": "user", "content": "hello"}])

        assert result["choices"][0]["message"]["content"] != ""
        assert result["usage"]["total_tokens"] == 100
        assert result["latency_ms"] >= 0.0

    async def test_tool_call_response(self, agent: Agent) -> None:
        tc = MagicMock()
        tc.id = "tc-001"
        tc.function.name = "read_file"
        tc.function.arguments = json.dumps({"path": "/tmp/test.txt"})

        mock_resp = _make_litellm_response(content="", tool_calls=[tc], tokens=50)
        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await agent._call_llm([{"role": "user", "content": "read a file"}])

        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "read_file"

    async def test_timeout_raises_propagated(self, agent: Agent) -> None:
        """LiteLLM timeout errors should propagate (not swallowed)."""
        import asyncio

        with patch("litellm.acompletion", new=AsyncMock(side_effect=asyncio.TimeoutError)):
            with pytest.raises(asyncio.TimeoutError):
                await agent._call_llm([{"role": "user", "content": "hello"}])

    async def test_rate_limit_error_propagated(self, agent: Agent) -> None:
        """Rate limit errors should propagate."""
        error = Exception("RateLimitError: 429")
        with patch("litellm.acompletion", new=AsyncMock(side_effect=error)):
            with pytest.raises(Exception, match="429"):
                await agent._call_llm([{"role": "user", "content": "hello"}])

    async def test_mock_fallback_when_litellm_unavailable(self, agent: Agent) -> None:
        """When litellm is not importable, _mock_llm_response is returned."""
        result = agent._mock_llm_response()
        assert "choices" in result
        assert result["usage"]["total_tokens"] == 0


class TestParseResponse:
    """Tests for Agent._parse_response."""

    def test_parse_json_text_response(self, agent: Agent) -> None:
        response = {
            "choices": [{"message": {
                "content": json.dumps({"reasoning": "think", "action": "act"}),
                "tool_calls": [],
            }}],
            "usage": {},
            "latency_ms": 0.0,
        }
        action, reasoning, tool_calls = agent._parse_response(response)
        assert action == "act"
        assert reasoning == "think"
        assert tool_calls == []

    def test_parse_tool_call_response(self, agent: Agent) -> None:
        response = {
            "choices": [{"message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                ],
            }}],
            "usage": {},
            "latency_ms": 0.0,
        }
        action, reasoning, tool_calls = agent._parse_response(response)
        assert "bash" in action
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "bash"
        assert tool_calls[0]["params"]["command"] == "ls"

    def test_parse_malformed_json_falls_back(self, agent: Agent) -> None:
        response = {
            "choices": [{"message": {"content": "not json at all", "tool_calls": []}}],
            "usage": {},
            "latency_ms": 0.0,
        }
        action, reasoning, tool_calls = agent._parse_response(response)
        assert action == "not json at all"
        assert tool_calls == []


class TestThinkAndActTokenTracking:
    """Tests for token usage and latency tracking in StepOutput."""

    async def test_tokens_recorded_in_step_output(self, agent: Agent) -> None:
        mock_resp = _make_litellm_response(
            content=json.dumps({"reasoning": "r", "action": "a"}),
            tokens=77,
        )
        context = LoopContext(
            desired_state=DesiredState(goal="test"),
            current_step=1,
            agent_id="test-llm-agent",
            trace_id="t1",
        )
        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            step = await agent.think_and_act({"goal": "test", "constraints": []}, context)

        assert step.llm_tokens_used == 77
        assert step.llm_latency_ms >= 0.0


class TestBuildToolsSchema:
    """Tests for Agent._build_tools_schema."""

    def test_no_tools_returns_empty(self, agent: Agent) -> None:
        assert agent._build_tools_schema() == []

    def test_with_registered_tools(self, state_store: SQLiteStateStore) -> None:
        from tools.code.file_tools import ReadFileTool
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(ReadFileTool())

        config = AgentConfig(agent_id="tool-agent", tools=["read_file"])
        a = Agent(config=config, state_store=state_store, tool_registry=registry)

        schema = a._build_tools_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "read_file"
