"""Unit tests for AgentScheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.runtime.scheduler import AgentScheduler, ScheduledTask
from core.state.models import AgentConfig, ReconcileResult


def _config(agent_id: str = "test-agent") -> AgentConfig:
    return AgentConfig(agent_id=agent_id)


def _task(goal: str = "do something", priority: int = 0, task_id: str | None = None) -> ScheduledTask:
    kwargs = dict(goal=goal, agent_config=_config(), priority=priority)
    if task_id:
        kwargs["task_id"] = task_id
    return ScheduledTask(**kwargs)


def _mock_runtime(converged: bool = True) -> MagicMock:
    """Create a mock AgentRuntime that returns a canned ReconcileResult."""
    result = ReconcileResult(
        status="converged" if converged else "failed",
        converged=converged,
        total_steps=1,
    )
    runtime = MagicMock()
    runtime.run = AsyncMock(return_value=result)
    return runtime


class TestAgentScheduler:

    # ------------------------------------------------------------------
    # submit / get_result
    # ------------------------------------------------------------------

    async def test_submit_returns_task_id(self) -> None:
        scheduler = AgentScheduler(_mock_runtime(), max_concurrent=2)
        task = _task()
        tid = await scheduler.submit(task)
        assert tid == task.task_id

    async def test_results_available_after_run(self) -> None:
        runtime = _mock_runtime()
        scheduler = AgentScheduler(runtime, max_concurrent=2)
        task = _task()
        await scheduler.submit(task)
        await scheduler.run_until_empty()
        result = await scheduler.get_result(task.task_id)
        assert result is not None
        assert result.converged is True

    async def test_none_result_before_run(self) -> None:
        scheduler = AgentScheduler(_mock_runtime(), max_concurrent=2)
        result = await scheduler.get_result("nonexistent-id")
        assert result is None

    # ------------------------------------------------------------------
    # Priority ordering
    # ------------------------------------------------------------------

    async def test_priority_ordering(self) -> None:
        """Higher-priority tasks should be started first."""
        execution_order: list[str] = []

        async def run_with_tracking(goal: str, agent_config, **_):
            execution_order.append(goal)
            return ReconcileResult(status="converged", converged=True, total_steps=1)

        runtime = MagicMock()
        runtime.run = run_with_tracking

        scheduler = AgentScheduler(runtime, max_concurrent=1)
        await scheduler.submit(_task("low", priority=0))
        await scheduler.submit(_task("high", priority=10))
        await scheduler.submit(_task("mid", priority=5))

        await scheduler.run_until_empty()

        # With max_concurrent=1, high should run before mid before low
        assert execution_order[0] == "high"
        assert execution_order[1] == "mid"
        assert execution_order[2] == "low"

    # ------------------------------------------------------------------
    # Concurrency cap
    # ------------------------------------------------------------------

    async def test_max_concurrent_respected(self) -> None:
        """No more than max_concurrent tasks should run simultaneously."""
        max_concurrent = 3
        running_count = 0
        peak_concurrent = 0
        lock = asyncio.Lock()

        async def counting_run(goal: str, agent_config, **_):
            nonlocal running_count, peak_concurrent
            async with lock:
                running_count += 1
                peak_concurrent = max(peak_concurrent, running_count)
            await asyncio.sleep(0.01)
            async with lock:
                running_count -= 1
            return ReconcileResult(status="converged", converged=True, total_steps=1)

        runtime = MagicMock()
        runtime.run = counting_run

        scheduler = AgentScheduler(runtime, max_concurrent=max_concurrent)
        for i in range(10):
            await scheduler.submit(_task(f"goal_{i}"))

        await scheduler.run_until_empty()
        assert peak_concurrent <= max_concurrent

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def test_cancel_queued_task(self) -> None:
        """Cancelled tasks should be skipped and not produce results."""
        runtime = _mock_runtime()
        scheduler = AgentScheduler(runtime, max_concurrent=1)
        task = _task()
        await scheduler.submit(task)
        cancelled = await scheduler.cancel(task.task_id)
        assert cancelled is True
        await scheduler.run_until_empty()
        result = await scheduler.get_result(task.task_id)
        assert result is None  # was skipped

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def test_context_manager(self) -> None:
        runtime = _mock_runtime()
        async with AgentScheduler(runtime, max_concurrent=2) as scheduler:
            await scheduler.submit(_task())
        # Tasks drained on exit; runtime was called
        runtime.run.assert_awaited()

    # ------------------------------------------------------------------
    # Failed tasks
    # ------------------------------------------------------------------

    async def test_failed_runtime_wrapped_in_result(self) -> None:
        runtime = MagicMock()
        runtime.run = AsyncMock(side_effect=RuntimeError("oops"))
        scheduler = AgentScheduler(runtime, max_concurrent=1)
        task = _task()
        await scheduler.submit(task)
        results = await scheduler.run_until_empty()
        assert len(results) == 1
        assert results[0].status == "failed"
        assert "oops" in results[0].error
