"""Unit tests for memory/scratchpad.py — AgentScratchpad."""

from __future__ import annotations

import pytest

from memory.scratchpad import AgentScratchpad


class TestAgentScratchpadCRUD:
    """Tests for basic set/get/delete operations."""

    def test_set_and_get(self):
        """set() stores a value that get() retrieves."""
        pad = AgentScratchpad()
        pad.set("todo", ["step 1", "step 2"])
        assert pad.get("todo") == ["step 1", "step 2"]

    def test_get_missing_returns_default(self):
        """get() returns None when key is absent."""
        pad = AgentScratchpad()
        assert pad.get("missing") is None

    def test_get_missing_custom_default(self):
        """get() returns the supplied default for missing keys."""
        pad = AgentScratchpad()
        assert pad.get("x", "fallback") == "fallback"

    def test_overwrite_value(self):
        """set() on an existing key replaces the value."""
        pad = AgentScratchpad()
        pad.set("step", 1)
        pad.set("step", 2)
        assert pad.get("step") == 2

    def test_delete_existing_key(self):
        """delete() removes the key and returns True."""
        pad = AgentScratchpad()
        pad.set("k", "v")
        assert pad.delete("k") is True
        assert pad.get("k") is None

    def test_delete_missing_key_returns_false(self):
        """delete() returns False when key is not found."""
        pad = AgentScratchpad()
        assert pad.delete("nonexistent") is False

    def test_clear_empties_all(self):
        """clear() removes all entries."""
        pad = AgentScratchpad()
        pad.set("a", 1)
        pad.set("b", 2)
        pad.clear()
        assert len(pad) == 0
        assert pad.get("a") is None


class TestAgentScratchpadInspection:
    """Tests for keys(), to_dict(), len(), contains."""

    def test_keys_empty(self):
        """keys() returns empty list when scratchpad is empty."""
        assert AgentScratchpad().keys() == []

    def test_keys_after_set(self):
        """keys() returns all set keys (order not guaranteed)."""
        pad = AgentScratchpad()
        pad.set("x", 1)
        pad.set("y", 2)
        assert set(pad.keys()) == {"x", "y"}

    def test_to_dict_shallow_copy(self):
        """to_dict() returns a copy, mutations don't affect scratchpad."""
        pad = AgentScratchpad()
        pad.set("count", 0)
        d = pad.to_dict()
        d["count"] = 99
        assert pad.get("count") == 0  # original unaffected

    def test_len(self):
        """__len__ reflects number of entries."""
        pad = AgentScratchpad()
        assert len(pad) == 0
        pad.set("a", 1)
        assert len(pad) == 1
        pad.delete("a")
        assert len(pad) == 0

    def test_contains_existing(self):
        """__contains__ returns True for stored keys."""
        pad = AgentScratchpad()
        pad.set("flag", True)
        assert "flag" in pad

    def test_contains_missing(self):
        """__contains__ returns False for absent keys."""
        pad = AgentScratchpad()
        assert "flag" not in pad

    def test_repr(self):
        """__repr__ mentions item count."""
        pad = AgentScratchpad()
        pad.set("k", "v")
        assert "1 item" in repr(pad)

    def test_supports_any_python_value(self):
        """Scratchpad can store any Python value without serialization."""
        pad = AgentScratchpad()

        class CustomObj:
            pass

        obj = CustomObj()
        pad.set("obj", obj)
        assert pad.get("obj") is obj  # identity preserved


class TestAgentScratchpadLifecycle:
    """Tests simulating the Agent.run() lifecycle."""

    def test_clear_resets_between_runs(self):
        """Calling clear() simulates a new run starting fresh."""
        pad = AgentScratchpad()

        # Run 1
        pad.set("run_1_data", "important")

        # Simulate new run: Agent clears scratchpad
        pad.clear()

        # Run 2 starts clean
        assert len(pad) == 0
        assert pad.get("run_1_data") is None

    def test_multiple_set_and_clear_cycles(self):
        """Repeated set/clear cycles work correctly."""
        pad = AgentScratchpad()

        for i in range(5):
            pad.set("i", i)
            assert pad.get("i") == i
            pad.clear()
            assert len(pad) == 0


class TestAgentScratchpadIntegrationWithAgent:
    """Tests verifying Agent integrates the scratchpad correctly."""

    async def test_agent_exposes_scratchpad_property(self):
        """Agent.scratchpad returns an AgentScratchpad instance."""
        from core.agent.agent import Agent
        from core.state.models import AgentConfig

        config = AgentConfig(agent_id="test-agent")
        agent = Agent(config)
        assert isinstance(agent.scratchpad, AgentScratchpad)

    async def test_agent_run_clears_scratchpad(self):
        """Agent.run() clears the scratchpad at start of each call."""
        from unittest.mock import AsyncMock, patch

        from core.agent.agent import Agent
        from core.state.models import AgentConfig, DesiredState, ReconcileResult
        from core.state.sqlite_store import SQLiteStateStore

        config = AgentConfig(agent_id="test-agent")
        state_store = SQLiteStateStore(":memory:")
        agent = Agent(config, state_store=state_store)

        # Pre-populate scratchpad (simulating leftover from previous run)
        agent.scratchpad.set("leftover", "should be gone")

        # Patch AgentReconcileLoop.run to avoid actual LLM calls
        dummy_result = ReconcileResult(
            run_id="r1",
            agent_id="test-agent",
            status="converged",
            steps_completed=0,
        )

        with patch(
            "core.agent.agent.AgentReconcileLoop.run",
            new_callable=AsyncMock,
            return_value=dummy_result,
        ):
            await agent.run(DesiredState(goal="test goal"))

        # Scratchpad should have been cleared at the start of run()
        assert "leftover" not in agent.scratchpad
