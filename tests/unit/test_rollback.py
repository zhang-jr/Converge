"""Unit tests for StateStore snapshot/restore and ReconcileLoop rollback.

Tests SQLiteStateStore.snapshot() / restore() / delete_snapshot() and
the enable_rollback flag in ReconcileLoop.
"""

from __future__ import annotations

import pytest

from core.state.models import DesiredState, StepOutput
from core.state.sqlite_store import SQLiteStateStore
from errors.exceptions import RollbackError


# ---------------------------------------------------------------------------
# SQLiteStateStore — snapshot / restore
# ---------------------------------------------------------------------------

class TestSQLiteStateStoreSnapshot:
    """Tests for snapshot and restore operations."""

    @pytest.fixture
    async def store(self) -> SQLiteStateStore:
        s = SQLiteStateStore(":memory:")
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_snapshot_returns_id(self, store: SQLiteStateStore) -> None:
        await store.put("k1", {"v": 1})
        snap_id = await store.snapshot()
        assert isinstance(snap_id, str)
        assert len(snap_id) > 0

    @pytest.mark.asyncio
    async def test_snapshot_then_restore_empty_prefix(self, store: SQLiteStateStore) -> None:
        await store.put("k1", {"value": "original"})
        await store.put("k2", {"value": "original2"})

        snap_id = await store.snapshot()

        # Mutate state after snapshot
        await store.put("k1", {"value": "mutated"})
        await store.put("k3", {"value": "new_key"})

        await store.restore(snap_id)

        k1 = await store.get("k1")
        k2 = await store.get("k2")
        k3 = await store.get("k3")

        assert k1 is not None and k1.value == {"value": "original"}
        assert k2 is not None and k2.value == {"value": "original2"}
        assert k3 is None  # k3 was added after snapshot → should be gone

    @pytest.mark.asyncio
    async def test_snapshot_prefix_isolation(self, store: SQLiteStateStore) -> None:
        await store.put("agent/k1", {"v": 1})
        await store.put("agent/k2", {"v": 2})
        await store.put("other/k3", {"v": 3})

        snap_id = await store.snapshot("agent/")

        # Mutate agent/ keys and other/ key
        await store.put("agent/k1", {"v": 99})
        await store.put("other/k3", {"v": 77})

        await store.restore(snap_id)

        # agent/ keys restored, other/ key NOT touched by restore
        k1 = await store.get("agent/k1")
        k3 = await store.get("other/k3")

        assert k1 is not None and k1.value == {"v": 1}
        assert k3 is not None and k3.value == {"v": 77}  # not in snapshot scope

    @pytest.mark.asyncio
    async def test_restore_unknown_snapshot_raises(self, store: SQLiteStateStore) -> None:
        with pytest.raises(RollbackError):
            await store.restore("nonexistent-snapshot-id")

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, store: SQLiteStateStore) -> None:
        await store.put("k", {"v": 1})
        snap_id = await store.snapshot()

        snapshots_before = await store.list_snapshots()
        assert snap_id in snapshots_before

        await store.delete_snapshot(snap_id)

        snapshots_after = await store.list_snapshots()
        assert snap_id not in snapshots_after

    @pytest.mark.asyncio
    async def test_multiple_snapshots_independent(self, store: SQLiteStateStore) -> None:
        await store.put("k", {"v": 1})
        snap1 = await store.snapshot()

        await store.put("k", {"v": 2})
        snap2 = await store.snapshot()

        # Restore to snap1 — should give v=1
        await store.restore(snap1)
        entry = await store.get("k")
        assert entry is not None and entry.value == {"v": 1}

        # Restore to snap2 — should give v=2
        await store.restore(snap2)
        entry = await store.get("k")
        assert entry is not None and entry.value == {"v": 2}

    @pytest.mark.asyncio
    async def test_snapshot_empty_store(self, store: SQLiteStateStore) -> None:
        snap_id = await store.snapshot()
        assert isinstance(snap_id, str)

        # Restore empty snapshot should clear the store
        await store.put("k", {"v": 1})
        await store.restore(snap_id)
        entries = await store.list()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_list_snapshots(self, store: SQLiteStateStore) -> None:
        await store.put("k", {"v": 1})
        snap1 = await store.snapshot()
        snap2 = await store.snapshot()

        all_snaps = await store.list_snapshots()
        assert snap1 in all_snaps
        assert snap2 in all_snaps


# ---------------------------------------------------------------------------
# ReconcileLoop — enable_rollback
# ---------------------------------------------------------------------------

class TestReconcileLoopRollback:
    """Tests for rollback integration in ReconcileLoop."""

    @pytest.mark.asyncio
    async def test_rollback_restores_state_on_failure(self) -> None:
        """On failure, state should be restored to pre-act snapshot."""
        from core.runtime.reconcile_loop import SimpleReconcileLoop

        store = SQLiteStateStore(":memory:")
        await store.put("task/state", {"status": "initial"})

        act_count = 0

        async def act_callback(diff, context, loop):
            nonlocal act_count
            act_count += 1
            # Mutate state then raise an error
            await store.put("task/state", {"status": "in_progress"})
            raise RuntimeError("Simulated failure during act")

        loop = SimpleReconcileLoop(
            state_store=store,
            act_callback=act_callback,
            safety_max_steps=3,
            enable_rollback=True,
        )

        with pytest.raises(RuntimeError, match="Simulated failure"):
            await loop.run(DesiredState(goal="test rollback"))

        # State should be rolled back to "initial" (the pre-act snapshot)
        entry = await store.get("task/state")
        assert entry is not None
        assert entry.value == {"status": "initial"}

        await store.close()

    @pytest.mark.asyncio
    async def test_no_rollback_state_persists_on_failure(self) -> None:
        """Without rollback enabled, mutated state persists after failure."""
        from core.runtime.reconcile_loop import SimpleReconcileLoop

        store = SQLiteStateStore(":memory:")
        await store.put("task/state", {"status": "initial"})

        async def act_callback(diff, context, loop):
            await store.put("task/state", {"status": "in_progress"})
            raise RuntimeError("Simulated failure")

        loop = SimpleReconcileLoop(
            state_store=store,
            act_callback=act_callback,
            safety_max_steps=1,
            enable_rollback=False,
        )

        with pytest.raises(RuntimeError):
            await loop.run(DesiredState(goal="test no rollback"))

        entry = await store.get("task/state")
        assert entry is not None
        # State NOT rolled back — shows the mutated value
        assert entry.value == {"status": "in_progress"}

        await store.close()

    @pytest.mark.asyncio
    async def test_successful_run_does_not_rollback(self) -> None:
        """A successful run should not undo its state changes."""
        from probes.quality_probe import DefaultQualityProbe, ProbeResult
        from core.runtime.reconcile_loop import SimpleReconcileLoop

        store = SQLiteStateStore(":memory:")
        await store.put("task/result", {"done": False})

        async def act_callback(diff, context, loop):
            await store.put("task/result", {"done": True})
            return StepOutput(
                step_number=context.current_step,
                action="mark_done",
                reasoning="task completed",
                result={"done": True},
            )

        class ConvergingProbe(DefaultQualityProbe):
            async def evaluate(self, step_output, context):
                return ProbeResult(
                    verdict="passed",
                    confidence=0.95,
                    reason="done",
                    should_converge=True,
                )

        loop = SimpleReconcileLoop(
            state_store=store,
            act_callback=act_callback,
            quality_probe=ConvergingProbe(),
            enable_rollback=True,
        )

        result = await loop.run(DesiredState(goal="test converge"))

        assert result.converged
        entry = await store.get("task/result")
        assert entry is not None
        assert entry.value == {"done": True}

        await store.close()

    @pytest.mark.asyncio
    async def test_step_callback_called_after_each_step(self) -> None:
        """step_callback must be invoked for every completed step."""
        from probes.quality_probe import DefaultQualityProbe, ProbeResult
        from core.runtime.reconcile_loop import SimpleReconcileLoop

        store = SQLiteStateStore(":memory:")
        received_steps: list[StepOutput] = []

        async def step_cb(step: StepOutput) -> None:
            received_steps.append(step)

        step_counter = 0

        class ThreeStepProbe(DefaultQualityProbe):
            async def evaluate(self, step_output, context):
                return ProbeResult(
                    verdict="passed",
                    confidence=0.9,
                    reason="ok",
                    should_converge=context.current_step >= 3,
                )

        async def act_callback(diff, context, loop):
            return StepOutput(
                step_number=context.current_step,
                action=f"step_{context.current_step}",
                reasoning="working",
            )

        loop = SimpleReconcileLoop(
            state_store=store,
            act_callback=act_callback,
            quality_probe=ThreeStepProbe(),
            step_callback=step_cb,
        )

        result = await loop.run(DesiredState(goal="test callback"))

        assert result.converged
        assert len(received_steps) == 3
        assert received_steps[0].step_number == 1
        assert received_steps[2].step_number == 3

        await store.close()
