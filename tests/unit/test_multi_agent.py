"""Unit tests for core/agent/multi_agent.py — Multi-Agent Orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agent.multi_agent import (
    MultiAgentConfig,
    MultiAgentOrchestrator,
    MultiAgentResult,
    pipeline,
    pool,
    supervisor,
)
from core.state.models import AgentConfig, DesiredState, ReconcileResult


# =============================================================================
# Helpers
# =============================================================================


def make_agent(agent_id: str) -> AgentConfig:
    return AgentConfig(agent_id=agent_id, name=agent_id.capitalize())


def converged(final_state: dict | None = None) -> ReconcileResult:
    return ReconcileResult(
        status="converged",
        converged=True,
        final_state=final_state or {},
        trace_id="t",
    )


def failed() -> ReconcileResult:
    return ReconcileResult(
        status="failed",
        converged=False,
        error="did not converge",
        trace_id="t",
    )


def make_runtime(*results: ReconcileResult) -> MagicMock:
    """Mock runtime whose run() returns `results` in order."""
    runtime = MagicMock()
    runtime.run = AsyncMock(side_effect=list(results))
    return runtime


def make_goal(text: str = "Complete the task") -> DesiredState:
    return DesiredState(goal=text)


# =============================================================================
# Convenience Constructors
# =============================================================================


class TestConvenienceConstructors:
    """Tests for pipeline(), supervisor(), pool() helper functions."""

    def test_pipeline_constructor(self):
        """pipeline() creates a MultiAgentConfig with pattern='pipeline'."""
        agents = [make_agent("a"), make_agent("b")]
        config = pipeline(agents, name="My Pipeline")
        assert config.pattern == "pipeline"
        assert config.name == "My Pipeline"
        assert len(config.agents) == 2

    def test_supervisor_constructor(self):
        """supervisor() puts supervisor first, then specialists."""
        sup = make_agent("lead")
        specs = [make_agent("spec-1"), make_agent("spec-2")]
        config = supervisor(sup, specs, name="My Team")
        assert config.pattern == "supervisor"
        assert config.agents[0].agent_id == "lead"
        assert len(config.agents) == 3

    def test_pool_constructor(self):
        """pool() creates a MultiAgentConfig with pattern='pool'."""
        agents = [make_agent("a"), make_agent("b"), make_agent("c")]
        config = pool(agents, name="Worker Pool")
        assert config.pattern == "pool"
        assert len(config.agents) == 3

    def test_default_names(self):
        """Constructors have sensible default names."""
        assert pipeline([make_agent("a")]).name == "Pipeline"
        assert pool([make_agent("a")]).name == "Pool"
        assert supervisor(make_agent("s"), []).name == "Supervisor"


# =============================================================================
# Pipeline Pattern
# =============================================================================


class TestPipelinePattern:
    """Tests for pipeline orchestration."""

    async def test_all_agents_run_in_order(self):
        """Pipeline runs agents in declaration order."""
        call_order = []

        async def run_side_effect(goal, agent_config, **kwargs):
            call_order.append(agent_config.agent_id)
            return converged()

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        config = pipeline([make_agent("a"), make_agent("b"), make_agent("c")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.status == "completed"
        assert call_order == ["a", "b", "c"]

    async def test_completed_agents_tracked(self):
        """completed_agents counts converged runs."""
        runtime = make_runtime(converged(), converged(), converged())
        config = pipeline([make_agent("a"), make_agent("b"), make_agent("c")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.completed_agents == 3
        assert result.total_agents == 3

    async def test_output_passed_as_context(self):
        """Each agent's final_state is injected as context for the next."""
        received_contexts = []

        async def run_side_effect(goal, agent_config, context=None, **kwargs):
            received_contexts.append(context or {})
            return converged(final_state={"from": agent_config.agent_id})

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        config = pipeline([make_agent("a"), make_agent("b")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        await orchestrator.run(make_goal())

        # Agent "b" should have received agent "a"'s output as context
        assert "a_output" in received_contexts[1]
        assert received_contexts[1]["a_output"]["from"] == "a"

    async def test_pipeline_fails_fast(self):
        """Pipeline stops when any agent fails (failed result)."""
        runtime = make_runtime(converged(), failed())  # Agent "c" never called

        config = pipeline([make_agent("a"), make_agent("b"), make_agent("c")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.status == "failed"
        # Only "a" and "b" were called; "c" was never reached
        assert runtime.run.call_count == 2

    async def test_results_recorded_for_each_agent(self):
        """agent_results dict contains an entry per executed agent."""
        runtime = make_runtime(converged(), converged())
        config = pipeline([make_agent("x"), make_agent("y")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert "x" in result.agent_results
        assert "y" in result.agent_results


# =============================================================================
# Pool Pattern
# =============================================================================


class TestPoolPattern:
    """Tests for pool orchestration."""

    async def test_all_agents_run_concurrently(self):
        """Pool runs all agents and records all results."""
        runtime = make_runtime(converged(), converged(), converged())
        config = pool([make_agent("a"), make_agent("b"), make_agent("c")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.status == "completed"
        assert result.completed_agents == 3
        assert runtime.run.call_count == 3

    async def test_partial_pool_success(self):
        """When some pool agents fail, status is 'partial'."""
        runtime = make_runtime(converged(), failed(), converged())
        config = pool([make_agent("a"), make_agent("b"), make_agent("c")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.status == "partial"
        assert result.completed_agents == 2

    async def test_all_fail_gives_failed_status(self):
        """When all pool agents fail, status is 'failed'."""
        runtime = make_runtime(failed(), failed())
        config = pool([make_agent("a"), make_agent("b")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.status == "failed"
        assert result.completed_agents == 0

    async def test_agent_results_keyed_by_id(self):
        """agent_results dict is keyed by agent_id."""
        runtime = make_runtime(converged(), converged())
        config = pool([make_agent("worker-1"), make_agent("worker-2")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert "worker-1" in result.agent_results
        assert "worker-2" in result.agent_results


# =============================================================================
# Supervisor Pattern
# =============================================================================


class TestSupervisorPattern:
    """Tests for supervisor orchestration."""

    async def test_supervisor_runs_three_phases(self):
        """Supervisor executes planning → specialists → synthesis."""
        call_goals = []

        async def run_side_effect(goal, agent_config, **kwargs):
            call_goals.append(goal)
            return converged(final_state={"summary": "done"})

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        config = supervisor(
            make_agent("lead"),
            [make_agent("spec-1"), make_agent("spec-2")],
        )
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal("Build a system"))

        # 4 calls: planning, spec-1, spec-2, synthesis
        assert runtime.run.call_count == 4
        assert result.status == "completed"

    async def test_supervisor_only_no_specialists(self):
        """Supervisor with no specialists skips specialist phase."""
        runtime = make_runtime(converged(), converged())  # planning + synthesis

        config = supervisor(make_agent("lead"), specialists=[])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        # 2 calls: planning + synthesis
        assert runtime.run.call_count == 2

    async def test_synthesis_result_included(self):
        """Synthesis result is captured in MultiAgentResult.synthesis."""

        async def run_side_effect(goal, agent_config, **kwargs):
            return converged(final_state={"summary": "Final synthesis output"})

        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=run_side_effect)

        config = supervisor(make_agent("lead"), [make_agent("spec")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        assert result.synthesis == "Final synthesis output"

    async def test_specialist_failures_handled(self):
        """Specialist failures do not prevent synthesis from running."""
        # planning=ok, spec=fail, synthesis=ok
        runtime = make_runtime(converged(), failed(), converged())

        config = supervisor(make_agent("lead"), [make_agent("spec")])
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(make_goal())

        # Synthesis still ran
        assert runtime.run.call_count == 3


# =============================================================================
# MultiAgentResult
# =============================================================================


class TestMultiAgentResult:
    """Tests for MultiAgentResult properties."""

    def test_duration_ms_none_when_not_completed(self):
        """duration_ms is None while completed_at is not set."""
        result = MultiAgentResult(
            orchestrator_id="orch",
            pattern="pool",
            status="completed",
        )
        assert result.duration_ms is None

    def test_duration_ms_when_completed(self):
        """duration_ms is positive after completed_at is set."""
        from datetime import datetime, timedelta

        result = MultiAgentResult(
            orchestrator_id="orch",
            pattern="pool",
            status="completed",
        )
        result.completed_at = result.started_at + timedelta(milliseconds=500)
        assert result.duration_ms is not None
        assert result.duration_ms >= 400  # ~500ms

    def test_initial_counts_are_zero(self):
        """Initial result has zero counts."""
        result = MultiAgentResult(
            orchestrator_id="orch",
            pattern="pipeline",
            status="completed",
        )
        assert result.total_agents == 0
        assert result.completed_agents == 0


# =============================================================================
# MultiAgentConfig
# =============================================================================


class TestMultiAgentConfig:
    """Tests for MultiAgentConfig model."""

    def test_orchestrator_id_auto_generated(self):
        """orchestrator_id is auto-generated if not specified."""
        cfg = MultiAgentConfig(name="test", pattern="pool", agents=[])
        assert len(cfg.orchestrator_id) > 0

    def test_shared_state_default_true(self):
        """shared_state defaults to True."""
        cfg = MultiAgentConfig(name="test", pattern="pool", agents=[])
        assert cfg.shared_state is True

    def test_invalid_pattern_rejected(self):
        """Invalid pattern is rejected by Pydantic."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            MultiAgentConfig(name="test", pattern="invalid", agents=[])
