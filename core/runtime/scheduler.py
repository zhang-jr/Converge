"""AgentScheduler — priority queue-based concurrent task dispatcher.

Accepts ScheduledTask objects, maintains a priority queue, and dispatches
tasks to AgentRuntime subject to a max_concurrent concurrency cap.

Phase 3 implementation: simple priority + concurrency control.
Capability matching interface is declared but not evaluated (Phase 4+).

Usage::

    async with AgentScheduler(runtime, max_concurrent=3) as scheduler:
        task_id = await scheduler.submit(ScheduledTask(
            task_id="t1",
            goal="Analyse sales data",
            agent_config=AgentConfig(agent_id="analyst"),
            priority=5,
        ))
        results = await scheduler.run_until_empty()
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from core.state.models import AgentConfig, ReconcileResult
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.runtime.agent_runtime import AgentRuntime


class ScheduledTask(BaseModel):
    """A task submitted to the AgentScheduler.

    Attributes:
        task_id: Unique identifier (auto-generated if not provided).
        goal: Natural language description of the goal.
        agent_config: Agent configuration to use for this task.
        priority: Scheduling priority — higher number runs first (default 0).
        required_capabilities: Declared capability requirements (Phase 4 matching).
        constraints: Goal constraints forwarded to AgentRuntime.
        context: Extra context forwarded to AgentRuntime.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    agent_config: AgentConfig
    priority: int = 0
    required_capabilities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentScheduler:
    """Priority queue scheduler for concurrent agent task execution.

    Args:
        runtime: The AgentRuntime used to execute tasks.
        max_concurrent: Maximum number of tasks running simultaneously (default 5).
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        max_concurrent: int = 5,
    ) -> None:
        self._runtime = runtime
        self._max_concurrent = max_concurrent
        # asyncio.PriorityQueue items: (-priority, sequence, task)
        # Negative priority so the highest number sorts first (min-heap).
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, ScheduledTask]
        ] = asyncio.PriorityQueue()
        self._running: dict[str, asyncio.Task[ReconcileResult]] = {}
        self._results: dict[str, ReconcileResult] = {}
        self._cancelled: set[str] = set()
        self._sequence: int = 0  # tiebreaker for equal priority
        self._worker_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, task: ScheduledTask) -> str:
        """Submit a task for scheduling.

        Args:
            task: The task to schedule.

        Returns:
            The task_id of the submitted task.
        """
        self._sequence += 1
        # Negate priority for min-heap (highest priority → smallest key)
        await self._queue.put((-task.priority, self._sequence, task))
        return task.task_id

    async def get_result(self, task_id: str) -> ReconcileResult | None:
        """Retrieve the result for a completed task.

        Args:
            task_id: The task to look up.

        Returns:
            ReconcileResult if completed, None if still running or unknown.
        """
        return self._results.get(task_id)

    async def cancel(self, task_id: str) -> bool:
        """Request cancellation of a task.

        If the task is currently running, the underlying asyncio.Task is
        cancelled. If it is still queued, it will be skipped when dequeued.

        Args:
            task_id: The task to cancel.

        Returns:
            True if the task was found and cancellation was requested.
        """
        self._cancelled.add(task_id)
        if task_id in self._running:
            self._running[task_id].cancel()
            return True
        return task_id in self._cancelled

    async def run_until_empty(self) -> list[ReconcileResult]:
        """Drain the queue and execute all submitted tasks.

        Blocks until the queue is empty and all running tasks have finished.

        Returns:
            List of ReconcileResult in completion order.
        """
        collected: list[ReconcileResult] = []
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def run_task(task: ScheduledTask) -> None:
            async with semaphore:
                if task.task_id in self._cancelled:
                    return
                try:
                    result = await self._runtime.run(
                        goal=task.goal,
                        agent_config=task.agent_config,
                        constraints=task.constraints,
                        context=task.context,
                    )
                except Exception as e:
                    # Wrap unhandled errors into a failed ReconcileResult
                    from core.state.models import ReconcileResult as RR
                    result = RR(
                        status="failed",
                        total_steps=0,
                        converged=False,
                        error=str(e),
                    )
                self._results[task.task_id] = result
                collected.append(result)

        worker_tasks: list[asyncio.Task[None]] = []
        while not self._queue.empty():
            _, _, task = await self._queue.get()
            if task.task_id in self._cancelled:
                self._queue.task_done()
                continue
            t = asyncio.create_task(run_task(task))
            self._running[task.task_id] = t  # type: ignore[assignment]
            worker_tasks.append(t)
            self._queue.task_done()

        if worker_tasks:
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Cleanup
        for task_id in list(self._running.keys()):
            self._running.pop(task_id, None)

        return collected

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AgentScheduler:
        """Start the scheduler."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Drain remaining tasks and shut down."""
        if not self._queue.empty():
            await self.run_until_empty()

        # Cancel any still-running tasks
        for t in self._running.values():
            t.cancel()
        if self._running:
            await asyncio.gather(*self._running.values(), return_exceptions=True)
        self._running.clear()
