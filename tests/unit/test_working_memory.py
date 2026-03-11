"""Unit tests for memory/working.py — Working memory."""

import time

import pytest

from memory.working import WorkingMemory


@pytest.fixture
def mem() -> WorkingMemory:
    """Fresh WorkingMemory with generous capacity."""
    return WorkingMemory(max_entries=10)


# =============================================================================
# Basic Get / Set / Delete
# =============================================================================


class TestBasicOperations:
    """Tests for set(), get(), delete()."""

    def test_set_and_get(self, mem: WorkingMemory):
        """set() stores a value; get() retrieves it."""
        mem.set("key", {"data": 42})
        assert mem.get("key") == {"data": 42}

    def test_get_missing_returns_default(self, mem: WorkingMemory):
        """get() returns the default value for absent keys."""
        assert mem.get("missing") is None
        assert mem.get("missing", "fallback") == "fallback"

    def test_set_overwrites(self, mem: WorkingMemory):
        """set() on an existing key overwrites its value."""
        mem.set("key", "original")
        mem.set("key", "updated")
        assert mem.get("key") == "updated"

    def test_delete_existing(self, mem: WorkingMemory):
        """delete() removes a key and returns True."""
        mem.set("key", "value")
        assert mem.delete("key") is True
        assert mem.get("key") is None

    def test_delete_missing_returns_false(self, mem: WorkingMemory):
        """delete() returns False for absent keys."""
        assert mem.delete("ghost") is False

    def test_has_existing(self, mem: WorkingMemory):
        """has() returns True for a present key."""
        mem.set("key", "value")
        assert mem.has("key") is True

    def test_has_missing(self, mem: WorkingMemory):
        """has() returns False for an absent key."""
        assert mem.has("ghost") is False

    def test_has_returns_true_for_none_value(self, mem: WorkingMemory):
        """has() returns True even when the stored value is None."""
        mem.set("key", None)
        assert mem.has("key") is True

    def test_stores_arbitrary_types(self, mem: WorkingMemory):
        """set() can store any Python object."""
        mem.set("list", [1, 2, 3])
        mem.set("dict", {"a": 1})
        mem.set("int", 42)
        mem.set("none", None)

        assert mem.get("list") == [1, 2, 3]
        assert mem.get("dict") == {"a": 1}
        assert mem.get("int") == 42
        assert mem.get("none") is None


# =============================================================================
# TTL
# =============================================================================


class TestTTL:
    """Tests for TTL-based expiry."""

    def test_no_ttl_never_expires(self, mem: WorkingMemory):
        """Entries without TTL (-1) never expire."""
        mem.set("key", "value", ttl_seconds=-1.0)
        time.sleep(0.01)
        assert mem.get("key") == "value"

    def test_expired_entry_returns_default(self, mem: WorkingMemory):
        """get() returns default when TTL has elapsed."""
        mem.set("key", "value", ttl_seconds=0.001)
        time.sleep(0.05)
        assert mem.get("key") is None

    def test_expired_entry_removed_on_access(self, mem: WorkingMemory):
        """get() removes the expired entry from storage."""
        mem.set("key", "value", ttl_seconds=0.001)
        time.sleep(0.05)
        mem.get("key")
        assert mem.size() == 0

    def test_has_returns_false_for_expired(self, mem: WorkingMemory):
        """has() returns False for expired entries."""
        mem.set("key", "value", ttl_seconds=0.001)
        time.sleep(0.05)
        assert mem.has("key") is False

    def test_prune_expired_removes_stale_entries(self, mem: WorkingMemory):
        """prune_expired() removes expired entries and returns count."""
        mem.set("live", "ok", ttl_seconds=60.0)
        mem.set("dead1", "bye", ttl_seconds=0.001)
        mem.set("dead2", "bye", ttl_seconds=0.001)
        time.sleep(0.05)

        removed = mem.prune_expired()
        assert removed == 2
        assert mem.get("live") == "ok"


# =============================================================================
# LRU Eviction
# =============================================================================


class TestLRUEviction:
    """Tests for LRU capacity enforcement."""

    def test_evicts_oldest_on_overflow(self):
        """When capacity is exceeded, the LRU entry is evicted."""
        mem = WorkingMemory(max_entries=3)
        mem.set("a", 1)
        mem.set("b", 2)
        mem.set("c", 3)
        mem.set("d", 4)  # Should evict "a"

        assert mem.get("a") is None
        assert mem.get("b") == 2
        assert mem.get("c") == 3
        assert mem.get("d") == 4

    def test_get_promotes_to_mru(self):
        """Accessing a key promotes it to the MRU position."""
        mem = WorkingMemory(max_entries=3)
        mem.set("a", 1)
        mem.set("b", 2)
        mem.set("c", 3)

        # Access "a" to promote it
        mem.get("a")

        # Adding "d" should evict "b" (now the LRU), not "a"
        mem.set("d", 4)

        assert mem.get("a") == 1  # Promoted, not evicted
        assert mem.get("b") is None  # Evicted
        assert mem.get("c") == 3
        assert mem.get("d") == 4

    def test_overwrite_promotes_to_mru(self):
        """Re-setting a key promotes it to MRU."""
        mem = WorkingMemory(max_entries=3)
        mem.set("a", 1)
        mem.set("b", 2)
        mem.set("c", 3)

        # Re-set "a" to promote it
        mem.set("a", 99)

        # "d" should evict "b"
        mem.set("d", 4)

        assert mem.get("a") == 99  # Not evicted
        assert mem.get("b") is None  # Evicted

    def test_size_stays_within_limit(self):
        """size() never exceeds max_entries."""
        mem = WorkingMemory(max_entries=5)
        for i in range(20):
            mem.set(f"key-{i}", i)
        assert mem.size() <= 5


# =============================================================================
# Tags
# =============================================================================


class TestTags:
    """Tests for tag-based retrieval."""

    def test_get_by_tag_returns_matching(self, mem: WorkingMemory):
        """get_by_tag() returns all entries with the specified tag."""
        mem.set("plan", {"steps": []}, tags=["planning"])
        mem.set("result", {"data": 1}, tags=["output"])
        mem.set("summary", {"text": "done"}, tags=["planning", "output"])

        planning = mem.get_by_tag("planning")
        assert "plan" in planning
        assert "summary" in planning
        assert "result" not in planning

    def test_get_by_tag_skips_expired(self, mem: WorkingMemory):
        """get_by_tag() excludes expired entries."""
        mem.set("live", 1, ttl_seconds=60.0, tags=["group"])
        mem.set("dead", 2, ttl_seconds=0.001, tags=["group"])
        time.sleep(0.05)

        result = mem.get_by_tag("group")
        assert "live" in result
        assert "dead" not in result

    def test_get_by_tag_empty_for_no_match(self, mem: WorkingMemory):
        """get_by_tag() returns empty dict when no entries have the tag."""
        mem.set("key", "value", tags=["other"])
        assert mem.get_by_tag("nonexistent") == {}


# =============================================================================
# Update
# =============================================================================


class TestUpdate:
    """Tests for update() method."""

    def test_update_existing_key(self, mem: WorkingMemory):
        """update() changes the value of an existing key."""
        mem.set("key", "original")
        result = mem.update("key", "updated")
        assert result is True
        assert mem.get("key") == "updated"

    def test_update_missing_key_returns_false(self, mem: WorkingMemory):
        """update() returns False for absent keys."""
        assert mem.update("ghost", "value") is False

    def test_update_expired_key_returns_false(self, mem: WorkingMemory):
        """update() returns False for expired entries."""
        mem.set("key", "value", ttl_seconds=0.001)
        time.sleep(0.05)
        assert mem.update("key", "new") is False


# =============================================================================
# Clear and Snapshot
# =============================================================================


class TestClearAndSnapshot:
    """Tests for clear() and snapshot()."""

    def test_clear_removes_all(self, mem: WorkingMemory):
        """clear() empties the memory store."""
        mem.set("a", 1)
        mem.set("b", 2)
        mem.clear()
        assert mem.size() == 0

    def test_snapshot_returns_all_live(self, mem: WorkingMemory):
        """snapshot() returns a dict of all non-expired entries."""
        mem.set("a", 1)
        mem.set("b", 2)
        snap = mem.snapshot()
        assert snap == {"a": 1, "b": 2}

    def test_snapshot_excludes_expired(self, mem: WorkingMemory):
        """snapshot() excludes expired entries."""
        mem.set("live", "ok", ttl_seconds=60.0)
        mem.set("dead", "bye", ttl_seconds=0.001)
        time.sleep(0.05)

        snap = mem.snapshot()
        assert "live" in snap
        assert "dead" not in snap
