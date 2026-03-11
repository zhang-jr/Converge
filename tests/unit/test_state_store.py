"""Unit tests for StateStore implementations."""

import asyncio

import pytest

from core.state.models import StateEntry
from core.state.sqlite_store import SQLiteStateStore
from errors.exceptions import VersionConflictError


@pytest.fixture
async def store():
    """Create a fresh in-memory SQLite store for each test."""
    store = SQLiteStateStore(":memory:")
    yield store
    await store.close()


class TestSQLiteStateStore:
    """Tests for SQLiteStateStore."""

    async def test_put_and_get(self, store: SQLiteStateStore):
        """Test basic put and get operations."""
        entry = await store.put("test-key", {"name": "test", "value": 42})

        assert entry.key == "test-key"
        assert entry.value == {"name": "test", "value": 42}
        assert entry.version == 1

        retrieved = await store.get("test-key")
        assert retrieved is not None
        assert retrieved.key == "test-key"
        assert retrieved.value == {"name": "test", "value": 42}

    async def test_get_nonexistent(self, store: SQLiteStateStore):
        """Test get returns None for nonexistent keys."""
        result = await store.get("nonexistent")
        assert result is None

    async def test_put_updates_version(self, store: SQLiteStateStore):
        """Test that put increments version on update."""
        await store.put("key", {"v": 1})
        entry1 = await store.get("key")
        assert entry1 is not None
        assert entry1.version == 1

        await store.put("key", {"v": 2})
        entry2 = await store.get("key")
        assert entry2 is not None
        assert entry2.version == 2
        assert entry2.value == {"v": 2}

    async def test_optimistic_lock_success(self, store: SQLiteStateStore):
        """Test optimistic lock with correct version."""
        await store.put("key", {"v": 1})

        entry = await store.put("key", {"v": 2}, expected_version=1)
        assert entry.version == 2
        assert entry.value == {"v": 2}

    async def test_optimistic_lock_conflict(self, store: SQLiteStateStore):
        """Test optimistic lock raises on version mismatch."""
        await store.put("key", {"v": 1})
        await store.put("key", {"v": 2})

        with pytest.raises(VersionConflictError) as exc_info:
            await store.put("key", {"v": 3}, expected_version=1)

        assert exc_info.value.expected_version == 1
        assert exc_info.value.actual_version == 2

    async def test_optimistic_lock_nonexistent(self, store: SQLiteStateStore):
        """Test optimistic lock on nonexistent key."""
        with pytest.raises(VersionConflictError):
            await store.put("nonexistent", {"v": 1}, expected_version=1)

    async def test_delete_existing(self, store: SQLiteStateStore):
        """Test delete removes existing entry."""
        await store.put("key", {"v": 1})
        result = await store.delete("key")

        assert result is True
        assert await store.get("key") is None

    async def test_delete_nonexistent(self, store: SQLiteStateStore):
        """Test delete returns False for nonexistent key."""
        result = await store.delete("nonexistent")
        assert result is False

    async def test_list_all(self, store: SQLiteStateStore):
        """Test list returns all entries."""
        await store.put("a", {"v": 1})
        await store.put("b", {"v": 2})
        await store.put("c", {"v": 3})

        entries = await store.list()
        assert len(entries) == 3
        keys = [e.key for e in entries]
        assert sorted(keys) == ["a", "b", "c"]

    async def test_list_with_prefix(self, store: SQLiteStateStore):
        """Test list filters by prefix."""
        await store.put("agent/1/state", {"v": 1})
        await store.put("agent/2/state", {"v": 2})
        await store.put("config/setting", {"v": 3})

        entries = await store.list("agent/")
        assert len(entries) == 2
        keys = [e.key for e in entries]
        assert "agent/1/state" in keys
        assert "agent/2/state" in keys
        assert "config/setting" not in keys

    async def test_list_empty(self, store: SQLiteStateStore):
        """Test list returns empty list when no entries."""
        entries = await store.list()
        assert entries == []

    async def test_updated_by_tracking(self, store: SQLiteStateStore):
        """Test that updated_by is tracked correctly."""
        entry = await store.put("key", {"v": 1}, updated_by="agent-1")
        assert entry.updated_by == "agent-1"

        retrieved = await store.get("key")
        assert retrieved is not None
        assert retrieved.updated_by == "agent-1"

    async def test_watch_receives_events(self, store: SQLiteStateStore):
        """Test that watch receives state change events."""
        events = []

        async def collect_events():
            async for event in store.watch():
                events.append(event)
                if len(events) >= 3:
                    break

        task = asyncio.create_task(collect_events())

        await asyncio.sleep(0.01)
        await store.put("key1", {"v": 1})
        await store.put("key2", {"v": 2})
        await store.put("key1", {"v": 3})

        await asyncio.wait_for(task, timeout=1.0)

        assert len(events) == 3
        assert events[0].key == "key1"
        assert events[0].change_type == "created"
        assert events[1].key == "key2"
        assert events[1].change_type == "created"
        assert events[2].key == "key1"
        assert events[2].change_type == "updated"

    async def test_watch_with_prefix_filter(self, store: SQLiteStateStore):
        """Test that watch filters by prefix."""
        events = []

        async def collect_events():
            async for event in store.watch("agent/"):
                events.append(event)
                if len(events) >= 2:
                    break

        task = asyncio.create_task(collect_events())

        await asyncio.sleep(0.01)
        await store.put("config/x", {"v": 1})
        await store.put("agent/1", {"v": 2})
        await store.put("other/y", {"v": 3})
        await store.put("agent/2", {"v": 4})

        await asyncio.wait_for(task, timeout=1.0)

        assert len(events) == 2
        assert all(e.key.startswith("agent/") for e in events)

    async def test_clear(self, store: SQLiteStateStore):
        """Test clear removes all entries."""
        await store.put("a", {"v": 1})
        await store.put("b", {"v": 2})

        await store.clear()

        entries = await store.list()
        assert entries == []

    async def test_context_manager(self):
        """Test async context manager."""
        async with SQLiteStateStore(":memory:") as store:
            await store.put("key", {"v": 1})
            entry = await store.get("key")
            assert entry is not None

    async def test_concurrent_writes(self, store: SQLiteStateStore):
        """Test concurrent writes work correctly."""
        async def write(key: str, value: int):
            await store.put(key, {"v": value})

        await asyncio.gather(
            write("k1", 1),
            write("k2", 2),
            write("k3", 3),
        )

        entries = await store.list()
        assert len(entries) == 3
